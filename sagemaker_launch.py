import argparse
import os
from datetime import datetime

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput


def get_job_name(base):
    now = datetime.now()
    now_ms_str = f'{now.microsecond // 1000:03d}'
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    job_name = '_'.join([base, date_str])
    return job_name


def launch(args):
    if args.wandb_api_key is None:
        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        assert wandb_api_key is not None, 'Please provide wandb api key either via --wandb-api-key or env variable WANDB_API_KEY.'
        args.wandb_api_key = wandb_api_key

    if args.local:
        assert args.instance_count == 1, f'Local mode requires 1 instance, now get {args.instance_count}.'
        assert args.input_source not in {'lustre'}
        args.sagemaker_session = sagemaker.LocalSession()
    else:
        assert args.input_source not in {'local'}    
        args.sagemaker_session = sagemaker.Session()


    dataset = args.config.split(":")[-1].split("_")[0]
    if "calvin" in dataset:
        dataset_dir = "calvin_data_processed"
        dataset_path = "calvin_data_processed"
    elif dataset == "bridge":
        dataset_dir = "bridgev2_processed"
        dataset_path = "bridgev2_processed"
    else:
        raise ValueError(f"Unsupported dataset: \"{dataset}\".")

    if args.input_source == 'local':
        input_mode = 'File'
        training_inputs = {dataset_dir:f'file://<path_to_data_dir>/{dataset_path}'}
        if dataset == "calvin":
            training_inputs[dataset_dir] += "/goal_conditioned"
        elif dataset == "calvinlcbc":
            training_inputs[dataset_dir] += "/language_conditioned"


    elif args.input_source == 'lustre':
        directory_path = f"/<lustreid>/<path_to_data_dir>//{dataset_path}"
        if dataset == "calvin":
            directory_path += "/goal_conditioned"
        elif dataset == "calvinlcbc":
            directory_path += "/language_conditioned"

        input_mode = 'File'
        train_fs = FileSystemInput(
            file_system_id='fs-<file_system_id>', 
            file_system_type='FSxLustre',
            directory_path=directory_path, 
            file_system_access_mode='ro'
        )

        training_inputs = {dataset_dir: train_fs}


    elif args.input_source == 's3':
        input_mode = 'FastFile'
        training_inputs = {dataset_dir:f's3://<path_to_data_dir>/{dataset_path}/'}
        if dataset == "calvin":
            training_inputs[dataset_dir] += "goal_conditioned/"
        elif dataset == "calvinlcbc":
            training_inputs[dataset_dir] += "language_conditioned/"
    else:
        raise ValueError(f'Invalid input source {args.input_source}')


    role = '<sagemaker_role>'
    role_name = role.split(['/'][-1])

    session = boto3.session.Session()
    region = session.region_name

    config = os.path.join('/opt/ml/code/', args.config)
    calvin_dataset_config = os.path.join('/opt/ml/code/', args.calvin_dataset_config)

    hyperparameters = {}


    subnets = [
        'subnet-<subnet_id1>',
        'subnet-<subnet_id2>',
    ]

    security_group_ids = [
        'sg-<security_group_id1>', 
        'sg-<security_group_id2>', 
        'sg-<security_group_id3>',
    ]

    job_name = get_job_name(args.base_job_name)

    if args.local:
        image_uri = f'{args.base_job_name}:latest' 
    else:
        image_uri = f'<docker_image_uri>/{args.base_job_name}:latest'
    
    output_path = os.path.join(f's3://<s3_save_uri>/{args.user}/bridge_data_v2/', job_name)

    checkpoint_s3_uri = None if args.local else output_path
    checkpoint_local_path = None if args.local else '/opt/ml/checkpoints'
    code_location = output_path

    base_job_name = args.base_job_name.replace("_", "-")
    instance_count = args.instance_count
    sagemaker_session = args.sagemaker_session

    instance_type = 'local_gpu' if args.local else args.instance_type 
    keep_alive_period_in_seconds = 0
    max_run = 60 * 60 * 24 * 5

    environment = {
        'WANDB_API_KEY': args.wandb_api_key,
        'WANDB_ENTITY': "tri",

        'CONFIG':config,
        "CALVIN_DATASET_CONFIG":calvin_dataset_config,
        "ALGO":args.algo,
        "DESCRIPTION":args.description,
        "DEBUG":args.debug,
        "WANDB_PROJ_NAME":args.wandb_proj_name,
        "SAVE_TO_S3":args.save_to_s3,
        "S3_SAVE_URI":args.s3_save_uri,
        "LOG_TO_WANDB":args.log_to_wandb,
        "SEEDS":",".join([str(seed) for seed in args.seeds]),
    }



    distribution = {
        'smdistributed': {
            'dataparallel': {
                    'enabled': False,
            },
        },
    }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'AWS region:                     {region}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    print(f'Configuration file:             {config}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print('#############################################################')
    print()
    print()

    sagemaker_job_tags = [
        {
            "Key": "XXXX",
            "Value": "XXXX"
        },
        {
            "Key": "XXXX",
            "Value": "XXXX"
        }
    ]

    if "bridge" in dataset:
        entry_point = "sagemaker_launch_bridge_gcbc.sh"
    else:
        entry_point = "sagemaker_launch_calvin_gcbc.sh"

    estimator = TensorFlow(
        base_job_name=base_job_name,
        entry_point=entry_point,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        environment=environment,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        max_run=max_run,
        input_mode=input_mode,
        job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=code_location,
        distribution=distribution,
        tags=sagemaker_job_tags,
    )

    estimator.fit(inputs=training_inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--calvin_dataset_config', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--debug', type=str, default="0")
    parser.add_argument('--save_to_s3', type=str, default="1")
    parser.add_argument('--log_to_wandb', type=str, default="1")
    parser.add_argument('--seeds', type=int, nargs="+", default="42")
    parser.add_argument('--base-job-name', type=str, required=True)
    parser.add_argument('--user', type=str, required=True, help='supported users under the IT-predefined bucket.')
    parser.add_argument('--wandb_proj_name', type=str, default=None)
    parser.add_argument('--s3_save_uri', type=str, default="s3://<s3_save_uri>")
    parser.add_argument('--wandb-api-key', type=str, default=None)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='lustre')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--instance_type', type=str, default="ml.p4de.24xlarge"),
    args = parser.parse_args()

    launch(args)
