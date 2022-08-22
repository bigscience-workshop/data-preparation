#!/bin/bash
#SBATCH --job-name=build_tokenization_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/build_tokenization_dataset/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

TRAINING_DATA_PREPARATION_REPO=$WORK/code/big_science/data-preparation/preprocessing/tokenizer/python_script
pushd $TRAINING_DATA_PREPARATION_REPO

DATASET_RATIOS_FILE=$TRAINING_DATA_PREPARATION_REPO/tokenizer_dataset_ratios.json

SAVE_DATASET_PATH=$six_ALL_CCFRSCRATCH/tokenizer/dataset/tokenization_dataset_v3
mkdir -p $SAVE_DATASET_PATH

export HF_DATASETS_OFFLINE=1

python aggregate_datasets.py \
    --dataset_ratios_path $DATASET_RATIOS_FILE \
    --load_num_proc 80 \
    --shard_max_size 10_000_000_000 \
    --save_path $SAVE_DATASET_PATH \
    --save_num_proc 80 \
    --save_batch_size 100
