#!/bin/bash

python3 -u calvin_gcbc.py \
--calvin_dataset_config=experiments/configs/susie/calvin/configs/gcbc_data_config.py:all \
--config=experiments/configs/susie/calvin/configs/gcbc_train_config.py:calvin_gcdiffusion_noactnorm-auggoaldiff \
--algo=gcdiffusion \
--description=auggoaldiff \
--debug=0 \
--log_to_wandb=0 \
--save_to_s3=0 \
--wandb_proj_name=susie_low_level \
--seed=42
