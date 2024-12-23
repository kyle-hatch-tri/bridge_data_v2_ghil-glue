#!/bin/bash

python3 -u train.py \
--config=experiments/configs/train_config.py:bridge_gcdiffusion_auggoaldiff \
--bridgedata_config=experiments/configs/data_config.py:all \
--algo=gcdiffusion \
--description=auggoaldiff \
--debug=0 \
--log_to_wandb=0 \
--save_to_s3=0 \
--wandb_proj_name=susie_low_level \
--seed=0