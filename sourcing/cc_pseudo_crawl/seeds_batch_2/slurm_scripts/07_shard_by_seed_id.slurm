#!/bin/bash
#SBATCH --job-name=pseudo_crawl_shard_by_id
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 04:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_2/logs/shard_by_seed_id/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=0-99   #TODO set correct number
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/cc

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-preprocessed-text-extracted/bigscience-catalogue-data/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--100
SAVE_DATASET_PREFIX_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets-shard-by-seed-id/bigscience-catalogue-data/pseudo_crawl_seed--"$SLURM_ARRAY_TASK_ID"--100

mkdir -p $(dirname $SAVE_DATASET_PREFIX_PATH)

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/shard_by_seed_id.py \
    --dataset-path $DATASET_PATH \
    --num-proc 4 \
    --save-prefix-path $SAVE_DATASET_PREFIX_PATH
