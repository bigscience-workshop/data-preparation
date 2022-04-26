#!/bin/bash
#SBATCH --job-name=pseudo_crawl_preprocess_v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uty16tp/code/big_science/logs/preprocess-on-subshards/%x-%j.out           # output file name
#SBATCH --array=0-99   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/cc

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-divide-in-subshards/bigscience-catalogue-data/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--"$SLURM_ARRAY_TASK_COUNT"
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-preprocessed/bigscience-catalogue-data
SAVE_DATASET_PATH=$SAVE_DATASET_DIR/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--"$SLURM_ARRAY_TASK_COUNT"
echo $DATASET_PATH

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python -m cc_pseudo_crawl.python_scripts.preprocess_dataset \
    --dataset-path $DATASET_PATH \
    --num-proc 80 \
    --save-path $SAVE_DATASET_PATH \
    --use-datasets-caching \
    --flavor seed
