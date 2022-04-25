#!/bin/bash
#SBATCH --job-name=pseudo_crawl_download_v1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_2/logs/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=0-99 #TODO set correct number
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/cc
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_2/datasets

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

mkdir -p $SAVE_DATASET_DIR

# TODO run this offline
# aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc-{FLAVOR}/ $CC_INDEX_FOLDER/cc-{FLAVOR}/

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/download_warc.py \
    --dataset bigscience-catalogue-data/pseudo_crawl_seed \
    --cc-index-folder $CC_INDEX_FOLDER \
    --save-dir $SAVE_DATASET_DIR \
    --num-proc 4 \
    --shard-id $SLURM_ARRAY_TASK_ID \
    --num-shards $SLURM_ARRAY_TASK_COUNT \
    --flavor "seeds_batch_2" \
    --use-datasets-caching
