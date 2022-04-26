#!/bin/bash
#SBATCH --job-name=pseudo_crawl_deduplicate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/six/commun/pseudo_crawl/seeds_batch_1_2/logs/deduplicate-on-clean-v2/%x-%j.out           # output file name #TODO change path if necessary
#SBATCH --array=0-613
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1_2/datasets-clean/bigscience-catalogue-data

SEED_ID=$(python cc_pseudo_crawl/python_scripts/load_all_seed_ids.py \
  --seed-paths "$DATA_SOURCING_REPO"/cc_pseudo_crawl/seeds_batch_1/sourcing_sheet_seeds/seeds.csv,"$DATA_SOURCING_REPO"/cc_pseudo_crawl/seeds_batch_2/sourcing_sheet_seeds/seeds.csv \
  --seed-index $SLURM_ARRAY_TASK_ID \
)

SAVE_DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1_2/datasets-deduplicate-on-clean-v2/bigscience-catalogue-data/lm_change_lang_id_seed_id_${SEED_ID}_pseudocrawl_change_name
echo $DATASET_PATH

mkdir -p $SAVE_DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/exact_deduplicates.py \
    --seed-id $SEED_ID \
    --save-dir $SAVE_DATASET_DIR \
    --pseudo_crawl_path $DATASET_PATH \
    --batch-size 1000 \
    --save-batch-size 1000 \
    --num-proc 8
