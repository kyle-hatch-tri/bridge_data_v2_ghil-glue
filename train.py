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

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list
from jaxrl_m.data.calvin_dataset import CalvinDataset
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors

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
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
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
    

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")


    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": FLAGS.wandb_proj_name + f"_bridge",
            "exp_descriptor": f"{FLAGS.algo}_{FLAGS.description}",
            "seed":FLAGS.config.seed,
        }
    )

    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=not FLAGS.log_to_wandb,
    )

    save_dir = tf.io.gfile.join(FLAGS.config.save_dir, FLAGS.wandb_proj_name, f"bridge", f"{FLAGS.algo}", f"{FLAGS.description}", f"seed_{FLAGS.config.seed}", f"{wandb_logger.config.unique_identifier}")
    os.makedirs(save_dir, exist_ok=True)
    s3_sync_callback = S3SyncCallback(os.path.abspath(save_dir), os.path.join(FLAGS.s3_save_uri, FLAGS.wandb_proj_name, f"bridge", f"{FLAGS.algo}", f"{FLAGS.description}", f"seed_{FLAGS.config.seed}", f"{wandb_logger.config.unique_identifier}"))  

    save_dict_as_yaml(os.path.join(save_dir, "config.yaml"), FLAGS.config.to_dict())
    if FLAGS.save_to_s3:
        s3_sync_callback.upload_base_savedir()

    # load datasets
    assert type(FLAGS.bridgedata_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.config.data_path, exclude=FLAGS.bridgedata_config.exclude
        )
        for path in FLAGS.bridgedata_config.include
    ]

    train_paths = [
        [os.path.join(path, "train/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]
    val_paths = [
        [os.path.join(path, "val/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]

    train_data = BridgeDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True,
        action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
        sample_weights=FLAGS.bridgedata_config.sample_weights,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = BridgeDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
        train=False,
        **FLAGS.config.dataset_kwargs,
    )

    if FLAGS.config.get("text_processor") is None:
        text_processor = None
    else:
        text_processor = text_processors[FLAGS.config.text_processor](
            **FLAGS.config.text_processor_kwargs
        )

    def process_text(batch):
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode([s for s in batch["goals"]["language"]])
        return batch
    
    train_data_iter = map(
        shard_fn, map(process_text, train_data.tf_dataset.as_numpy_iterator())
    )

    if FLAGS.config.language_conditioned:
        assert FLAGS.config.encoder == "resnetv1-34-bridge-film", f"FLAGS.config.encoder: {FLAGS.config.encoder}"
    else:
        assert FLAGS.config.encoder == "resnetv1-34-bridge", f"FLAGS.config.encoder: {FLAGS.config.encoder}"

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

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
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0 or (i + 1) == 1000:
            logging.info("Evaluating...")
            timer.tick("val")
            metrics = []
            val_data_iter = map(shard_fn, map(process_text, val_data.iterator()))
            for _, batch in zip(tqdm.trange(FLAGS.config.num_val_batches), val_data_iter):
                rng, val_rng = jax.random.split(rng)
                metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0 or (i + 1) == 1000:
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


if __name__ == "__main__":
    app.run(main)
