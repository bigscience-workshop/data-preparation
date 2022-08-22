#!/bin/bash
#SBATCH --job-name=download_all_catalogue_datasets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/format_catalogue_meta/%x-%j.out           # output file name
#SBATCH --array=0-253   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

TRAINING_DATA_PREPARATION_REPO=$WORK/code/big_science/data-preparation/preprocessing/tokenizer/python_script
pushd $TRAINING_DATA_PREPARATION_REPO

DATASET_RATIOS_FILE=$TRAINING_DATA_PREPARATION_REPO/tokenizer_dataset_ratios.json

export HF_DATASETS_OFFLINE=1

cd ..
python -m format_meta \
    --dataset_ratios_path $DATASET_RATIOS_FILE \
    --dataset_index $SLURM_ARRAY_TASK_ID \
    --num_proc 1 \
