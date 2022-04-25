#!/bin/bash
#SBATCH --job-name=pseudo_crawl_extract_lang_tag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_2/logs/extract_lang_tag_V5/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=35,341,297
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

DATASET_PATH=/gpfsscratch/rech/six/urd43gx/crawl/shards/shard_"$SLURM_ARRAY_TASK_ID"
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/fasttext_annotation/seeds_batch_1/datasets-lang-annotation/bigscience-catalogue-data
SAVE_DATASET_PATH=$SAVE_DATASET_DIR/seed_id="$SLURM_ARRAY_TASK_ID"
echo $DATASET_PATH

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python -m cc_pseudo_crawl.language_annotation.python_scripts.detect_html_lang_attrib \
    --dataset-path $DATASET_PATH \
    --num-proc 40 \
    --save-path $SAVE_DATASET_PATH \
    --use-datasets-caching
