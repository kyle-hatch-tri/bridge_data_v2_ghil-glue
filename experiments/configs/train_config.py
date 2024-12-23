from ml_collections import ConfigDict


def get_config(config_string):
    dataset, algo, variant = config_string.split("_")
    variant = variant.split("-")

    base_real_config = dict(
        batch_size=256,
        num_val_batches=8,
        num_steps=600_000, 
        log_interval=1000,
        eval_interval=50_000,
        save_interval=50_000,
        save_dir="<path to save dir>",
        data_path="<path_to_data_dir>",
        dataset_name=dataset,
        resume_path=None,
        seed=42,
    )

    base_data_config = dict(
        shuffle_buffer_size=25_000,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    possible_structures = {
        "gciql": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=dict(
                    network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        state_dependent_std=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    ),
                    learning_rate=3e-4,
                    discount=0.98,
                    expectile=0.7,
                    temperature=1.0,
                    target_update_rate=0.002,
                    shared_encoder=True,
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    negative_proportion=0.1,
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.1),
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "gcbc": ConfigDict(
            dict(
                agent="gc_bc",
                agent_kwargs=dict(
                    network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "lcbc": ConfigDict(
            dict(
                agent="lc_bc",
                agent_kwargs=dict(
                    network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    load_language=True,
                    skip_unlabeled=True,
                    **base_data_config,
                ),
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "gcdiffusion": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="delta_goals",
                    goal_relabeling_kwargs=dict(goal_delta=[0, 24]),
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=4,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "lcdiffusion": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=32,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    language_conditioned=True,
                    early_goal_concat=False,
                    shared_goal_encoder=False,
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=20,
                    action_samples=1,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="delta_goals",
                    goal_relabeling_kwargs=dict(goal_delta=[0, 20]), # This value doesn't matter since it is always the same language instruction
                    load_language=True,
                    skip_unlabeled=True,
                    relabel_actions=True,
                    obs_horizon=1,
                    act_pred_horizon=4,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "lcgcprogressvf": ConfigDict(
            dict(
                agent="lcgc_progress_vf",
                agent_kwargs = dict(
                    network_kwargs=dict(
                        dropout_rate=0.1,
                        hidden_dims=[256, 256],
                        use_layer_norm=True,
                    ),
                    early_goal_concat=False,
                    shared_goal_encoder=False,
                    use_proprio=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,

                    frac_pos=0.5,
                    frac_neg_wrong_lang=0.2,
                    frac_neg_reverse_direction=0.2,
                    frac_neg_wrong_goalimg=0.1,

                    loss_fn="bce",
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="delta_goals2",
                    goal_relabeling_kwargs=dict(goal_delta=[16, 24]),
                    relabel_actions=True,
                    load_language=True,
                    skip_unlabeled=True,
                    obs_horizon=None,
                    act_pred_horizon=None,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=True, act="swish"
                ),
                **base_real_config,
            )
        ),
        "contrastiverltd": ConfigDict(
            dict(
                agent="stable_contrastive_rl",
                agent_kwargs=dict(
                    critic_network_kwargs=dict(
                        hidden_dims=(256, 256, 256), use_layer_norm=True
                    ),
                    critic_kwargs=dict(init_final=1e-12, repr_dim=16, twin_q=True),
                    policy_network_kwargs=dict(
                        hidden_dims=(256, 256, 256), dropout_rate=0.1
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=False,
                        state_dependent_std=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    ),
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    actor_decay_steps=int(2e6),
                    use_td=True,
                    gcbc_coef=0.20,
                    discount=0.98,
                    temperature=1.0,
                    target_update_rate=0.002,
                    shared_encoder=False,
                    early_goal_concat=False,
                    shared_goal_encoder=True,
                    use_proprio=False,
                ),
                dataset_kwargs=dict(
                    goal_relabeling_strategy="uniform",
                    goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    relabel_actions=True,
                    **base_data_config,
                ),
                encoder="resnetv1-34-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg", add_spatial_coordinates=False, act="swish"
                ),
                **base_real_config,
            )
        ),
    }

    

    config = possible_structures[algo]

    if "lc" in algo:
        assert algo[:2] == "lc"
        config["language_conditioned"] = True 
        config["encoder"] = "resnetv1-34-bridge-film"
        config["text_processor"] = "muse_embedding"
        config["text_processor_kwargs"] = dict()
    else:
        config["language_conditioned"] = False 

    if "auggoaldiff" in variant:
        config["dataset_kwargs"]["augment_next_obs_goal_differently"] = True 

    # this arg is currently not implemented for bridge data loader
    if "noactnorm" in variant:
        config["dataset_kwargs"]["normalize_actions"] = False

    if "goaldelta20short" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [0, 24]


    if "goaldelta20long" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [16, 24]
        config["dataset_kwargs"]["goal_relabeling_strategy"] = "delta_goals2"

    if "unipi" in variant:
        assert algo == "gcbc"


        config["num_steps"] = 500_000
        config["eval_interval"] = 25_000
        config["save_interval"] = 25_000

        config["agent_kwargs"]["network_kwargs"] = dict(
                                            hidden_dims=(256, 256, 256),
                                            dropout_rate=0.1,
                                        )
        config["agent_kwargs"]["policy_kwargs"] = dict(
                    tanh_squash_distribution=False,
                    fixed_std=[1, 1, 1, 1, 1, 1, 1],
                    state_dependent_std=False,
                )
        config["agent_kwargs"]["decay_steps"]= int(2e6)

        

        config["dataset_kwargs"]["goal_relabeling_strategy"] = "delta_goals"
        config["dataset_kwargs"]["goal_relabeling_kwargs"] = dict(goal_delta=[1, 1])
        config["dataset_kwargs"]["relabel_actions"] = True
        config["dataset_kwargs"]["act_pred_horizon"] = None 
        config["dataset_kwargs"]["obs_horizon"] = None 



    for batch_size in [1024, 2048, 4096, 8192]:
        if f"b{batch_size}" in variant:
            config["batch_size"] = batch_size

    return config


if __name__ == "__main__":
    config = get_config("bridge_gcbc_auggoaldiff-auggoaldiff-unipi")
    print(config)