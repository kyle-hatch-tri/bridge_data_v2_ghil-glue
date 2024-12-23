# Train Low-Level Policies and Subgoal Classifiers for GHIL-Glue

This fork of the [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) repo contains code for training low-level policies and subgoal classifiers for [GHIL-Glue](https://github.com/kyle-hatch-tri/ghil-glue) on both the [BridgeData V2 dataset](https://rail-berkeley.github.io/bridgedata/) and the [CALVIN dataset](https://github.com/mees/calvin).


## Installation

The dependencies for this codebase can be installed in a conda environment:

```bash
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
pip install -r requirements.txt
```
For GPU:
```bash
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax. See these [instructions](https://github.com/rail-berkeley/bridge_data_v2?tab=readme-ov-file#environment) from the original [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) repo for troubleshooting installation issues.



## Download checkpoints

The trained low-level policy and subgoal classifier checkpoints can be downloaded from https://huggingface.co/kyle-hatch-tri/ghil-glue-checkpoints


## Data

### Instructions for downloaded and processing the Bridge data:

1. Download the raw Bridge data from [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/).
2. Process the raw data into numpy format:
  ```
  python3 data_processing/bridgedata_raw_to_numpy.py \
  --input_path <path_to_raw_bridge_v2_data> \
  --output_path <path_to_numpy_bridge_v2_data> \
  --depth 5 \
  --im_size 200
  ```
3. Process the numpy data into tfrecord format:
```
python3 data_processing/bridgedata_numpy_to_tfrecord.py \
--input_path <path_to_numpy_bridge_v2_data> \
--output_path <path_to_tfrecord_bridge_v2_data> \
--depth 5
```
See these [instructions](https://github.com/rail-berkeley/bridge_data_v2?tab=readme-ov-file#data) from the original [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) repo for troubleshooting downloading and processing the BridgeV2 data.


### Instructions for downloaded and processing the CALVIN data:
1. Download the raw CALVIN data following the [instructions](https://github.com/mees/calvin?tab=readme-ov-file#computer--quick-start) from the original [CALVIN](https://github.com/mees/calvin) repo.
2. Process the raw data into tfrecord format using the scripts found in `experiments/configs/susie/calvin/dataset_conversion_scripts`.


## Training

See `launch_training_local_calvin.sh` and `launch_training_local_bridge.sh` for examples of launching local training runs to reproduce our low-level policy and subgoal classifier training.

