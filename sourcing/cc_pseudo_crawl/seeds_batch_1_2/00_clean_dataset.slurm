#!/bin/bash
#SBATCH --job-name=pseudo_crawl_clean_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_1_2/logs/clean_dataset-v2/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=0-613
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/hub/pseudo_crawl
SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1_2/datasets-clean/bigscience-catalogue-data
echo $DATASET_PATH

SEED_ID=$(python cc_pseudo_crawl/python_scripts/load_all_seed_ids.py \
  --seed-paths "$DATA_SOURCING_REPO"/cc_pseudo_crawl/seeds_batch_1/sourcing_sheet_seeds/seeds.csv,"$DATA_SOURCING_REPO"/cc_pseudo_crawl/seeds_batch_2/sourcing_sheet_seeds/seeds.csv \
  --seed-index $SLURM_ARRAY_TASK_ID \
)

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/pseudo_crawl_seed_to_lm_dset_v2.py \
    --seed-id $SEED_ID \
    --save-dir $SAVE_DATASET_DIR \
    --pseudo_crawl_path $DATASET_PATH \
    --batch-size 10 \
    --save-batch-size 10 \
    --num-proc 10 \
    --min-chars 32 \
    --n-records 10000 \
    --pourcentage-threshold 0.01 \
    --min-repetition-threshold 10
