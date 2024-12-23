import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
import yaml

from jaxrl_m.data.text_processing import text_processors

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.calvin_dataset import CalvinDataset, glob_to_path_list
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders

from s3_save import S3SyncCallback

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("wandb_proj_name", "jaxrl_m_calvin_gcbc", "Experiment name.")
flags.DEFINE_string("s3_save_uri", "", "Experiment name.")
flags.DEFINE_integer("debug", 0, "Debug config")
flags.DEFINE_integer("save_to_s3", 1, "")
flags.DEFINE_integer("seed", None, "")
flags.DEFINE_integer("log_to_wandb", 1, "")
flags.DEFINE_string("algo", "", "Experiment name.")
flags.DEFINE_string("description", "", "Experiment name.")
flags.DEFINE_string("checkpoint_path", "", "Experiment name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "calvin_dataset_config",
    None,
    "File path to the CALVIN dataset configuration.",
    lock_config=False,
)


def save_dict_as_yaml(savepath, data):
    with open(savepath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

import cv2
def save_video(output_video_file, frames):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    fps = 30  # Adjust the frame rate as needed

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)

    if FLAGS.seed is not None:
        FLAGS.config.seed = FLAGS.seed

    if FLAGS.debug:
        FLAGS.config.batch_size = 24
        FLAGS.config.num_val_batches = 2
        FLAGS.config.num_steps = 100
        FLAGS.config.log_interval = 20
        FLAGS.config.eval_interval = 90
        FLAGS.config.save_interval = 80
        FLAGS.wandb_proj_name = "el_trasho"

    
    assert FLAGS.config.batch_size % num_devices == 0

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": FLAGS.wandb_proj_name + f"_{FLAGS.config.dataset_name}",
            "exp_descriptor": f"{FLAGS.algo}_{FLAGS.description}",
            "seed":FLAGS.config.seed,
        }
    )

    
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        # debug=FLAGS.debug,
        debug=not FLAGS.log_to_wandb,
    )

    save_dir = tf.io.gfile.join(FLAGS.config.save_dir, FLAGS.wandb_proj_name, f"{FLAGS.config.dataset_name}", f"{FLAGS.algo}", f"{FLAGS.description}", f"seed_{FLAGS.config.seed}", f"{wandb_logger.config.unique_identifier}")
    os.makedirs(save_dir, exist_ok=True)
    s3_sync_callback = S3SyncCallback(os.path.abspath(save_dir), os.path.join(FLAGS.s3_save_uri, FLAGS.wandb_proj_name, f"{FLAGS.config.dataset_name}", f"{FLAGS.algo}", f"{FLAGS.description}", f"seed_{FLAGS.config.seed}", f"{wandb_logger.config.unique_identifier}"))  

    save_dict_as_yaml(os.path.join(save_dir, "config.yaml"), FLAGS.config.to_dict())
    if FLAGS.save_to_s3:
        s3_sync_callback.upload_base_savedir()

    # load datasets
    assert type(FLAGS.calvin_dataset_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.config.data_path, exclude=FLAGS.calvin_dataset_config.exclude
        )
        for path in FLAGS.calvin_dataset_config.include
    ]

    train_paths = [task_paths[0]]
    val_paths = [task_paths[1]]

    train_data = CalvinDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        num_devices=num_devices, 
        train=True,
        action_proprio_metadata=FLAGS.calvin_dataset_config.action_proprio_metadata,
        sample_weights=FLAGS.calvin_dataset_config.sample_weights,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = CalvinDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        action_proprio_metadata=FLAGS.calvin_dataset_config.action_proprio_metadata,
        train=False,
        **FLAGS.config.dataset_kwargs,
    )

    def process_text(batch): 
        if text_processor is None:
            batch["goals"].pop("language")
        else:
            batch["goals"]["language"] = text_processor.encode(
                [s for s in batch["goals"]["language"]]
            )

            
        return batch

    if FLAGS.config.language_conditioned:
        assert FLAGS.config.encoder == "resnetv1-34-bridge-film", f"FLAGS.config.encoder: {FLAGS.config.encoder}"
        text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)
        train_data_iter = map(process_text, train_data.tf_dataset.as_numpy_iterator())
    else:
        assert FLAGS.config.encoder == "resnetv1-34-bridge", f"FLAGS.config.encoder: {FLAGS.config.encoder}"
        train_data_iter = train_data.iterator()

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )


    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding) 

    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )

    if FLAGS.checkpoint_path is not None:
        agent = checkpoints.restore_checkpoint(FLAGS.checkpoint_path, target=agent)


        
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = next(train_data_iter)

        if FLAGS.config.dataset_kwargs.goal_relabeling_strategy == "delta_goals_with_generated_encode_decode":
            assert np.max(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"]) <= 1, f'batch["uses_generated_goal"]: {batch["uses_generated_goal"]}, batch["uses_encode_decode_goal"]: {batch["uses_encode_decode_goal"]}, batch["uses_noised_encode_decode_goal"]: {batch["uses_noised_encode_decode_goal"]}'
            
            generated_goal_mask = batch["uses_generated_goal"]
            encode_decode_mask = batch["uses_encode_decode_goal"]
            noised_encode_decode_mask = batch["uses_noised_encode_decode_goal"]
            real_goal_mask = np.logical_not(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"])
            assert np.array_equal(generated_goal_mask + encode_decode_mask + noised_encode_decode_mask + real_goal_mask, np.ones_like(generated_goal_mask)), f"generated_goal_mask: {generated_goal_mask}, encode_decode_mask: {encode_decode_mask}, encode_decode_mask: {encode_decode_mask}, noised_encode_decode_mask: {noised_encode_decode_mask}"

        batch = shard_batch(batch, sharding) 
        timer.tock("dataset")
        timer.tick("train")

        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0  or (i + 1) == 1000:
            logging.info("Evaluating...")
            timer.tick("val")
            metrics = []

            if FLAGS.config.language_conditioned:
                val_data_iter = map(process_text, val_data.tf_dataset.as_numpy_iterator())
            else:
                val_data_iter = val_data.iterator()

            for _, batch in zip(tqdm.trange(FLAGS.config.num_val_batches), val_data_iter):
                rng, val_rng = jax.random.split(rng)
                m = agent.get_debug_metrics(batch, seed=val_rng)
                metrics.append(m)

            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0  or (i + 1) == 1000:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

            if FLAGS.save_to_s3:
                s3_sync_callback.on_train_epoch_end(i + 1)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)

            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


def add_generated_goals_to_batch(batch, frac_generated=0.5):

    N = batch["goals"]["image"].shape[0]
    assert batch["generated_goals"].shape[0] == N
    n_generated_goals = batch["generated_goals"].shape[1]

    selected_generated_idxs = np.random.choice(n_generated_goals, size=N)
    generated_goals = batch["generated_goals"][np.arange(N), selected_generated_idxs, ...]

    random_idxs = np.arange(N)
    np.random.shuffle(random_idxs)
    random_idxs = random_idxs[:int(N * frac_generated)]

    batch["goals"]["image"] = batch["goals"]["image"].copy() # Need to do this to overcome the "ValueError: assignment destination is read-only" error
    batch["goals"]["image"][random_idxs] = generated_goals[random_idxs]
    del batch["generated_goals"]
    return batch

if __name__ == "__main__":
    app.run(main)
