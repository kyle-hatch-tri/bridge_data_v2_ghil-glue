#!/bin/bash

echo "WANDB_API_KEY: $WANDB_API_KEY"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "CONFIG: $CONFIG"
echo "CALVIN_DATASET_CONFIG: $CALVIN_DATASET_CONFIG"
echo "ALGO: $ALGO"
echo "DESCRIPTION: $DESCRIPTION"
echo "DEBUG: $DEBUG"

if [[ "$DEBUG" == "1" ]]; then 
    WANDB_PROJ_NAME="el_trasho"
fi 


echo "WANDB_PROJ_NAME: $WANDB_PROJ_NAME"
echo "SAVE_TO_S3: $SAVE_TO_S3"
echo "S3_SAVE_URI: $S3_SAVE_URI"
echo "LOG_TO_WANDB: $LOG_TO_WANDB"
echo "SEEDS: $SEEDS"

IFS=',' read -r -a seeds <<< "$SEEDS"

num_gpus=$(nvidia-smi -L | wc -l)
num_seeds=${#seeds[@]}
echo "num_seeds: $num_seeds"
echo "num_gpus: $num_gpus"

if [[ ( "$num_seeds" > $num_gpus ) ]]; then
    echo "Too many seeds"
        exit 125
fi 


current_timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
mkdir -p results/stdouts/$current_timestamp




for i in "${!seeds[@]}"
do
    echo "seeds[$i]: ${seeds[i]}"

    if [[ ( "$num_seeds" > 1 ) ]]; then
        if [[ ( "$num_gpus" == 8 ) ]]; then
            if [[ ( "$i" == 0 ) ]]; then
                export CUDA_VISIBLE_DEVICES="0,1"
            elif [[ ( "$i" == 1 ) ]]; then
                export CUDA_VISIBLE_DEVICES="2,3"
            elif [[ ( "$i" == 2 ) ]]; then
                export CUDA_VISIBLE_DEVICES="4,5"
            elif [[ ( "$i" == 3 ) ]]; then
                export CUDA_VISIBLE_DEVICES="6,7"
            else
                echo "Too many seeds pt. 2"
                exit 125
            fi
        elif [[ ( "$num_gpus" == 3 ) ]]; then
            if [[ ( "$i" == 0 ) ]]; then
                    export CUDA_VISIBLE_DEVICES="0"
                elif [[ ( "$i" == 1 ) ]]; then
                    export CUDA_VISIBLE_DEVICES="1"
                elif [[ ( "$i" == 2 ) ]]; then
                    export CUDA_VISIBLE_DEVICES="2"
                else
                    echo "Too many seeds pt. 3"
                    exit 125
                fi
        else 
            echo "Can't handle this number of gpus"
            exit 125
        fi
    fi 

    
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    python3 -u calvin_gcbc.py \
    --calvin_dataset_config=$CALVIN_DATASET_CONFIG \
    --config=$CONFIG \
    --algo=$ALGO \
    --description=$DESCRIPTION \
    --debug=$DEBUG \
    --log_to_wandb=$LOG_TO_WANDB \
    --save_to_s3=$SAVE_TO_S3 \
    --s3_save_uri=$S3_SAVE_URI \
    --wandb_proj_name=$WANDB_PROJ_NAME \
    --seed=${seeds[i]} \
    2>&1 | tee "results/stdouts/$current_timestamp/stdout_and_sterr_$i.txt" &
done 
wait 


aws s3 sync "results/stdouts/$current_timestamp" "$S3_SAVE_URI/$WANDB_PROJ_NAME/stdouts/$current_timestamp"