#!/bin/bash
#SBATCH --job-name=pseudo_crawl_shard_and_compress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/fix_compress_cpu_p1_2/%x-%j.out           # output file name
#SBATCH --array=0-604   # TODO: modify according to the number of models you want to evaluated
#SBATCH --mail-type=ALL
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CC_INDEX_FOLDER=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/cc

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

SEED_ID=$(python cc_pseudo_crawl/seeds_batch_1/load_all_seed_ids.py --seed-path "$DATA_SOURCING_REPO"/cc_pseudo_crawl/seeds_batch_1/sourcing_sheet_seeds/ --seed-index $SLURM_ARRAY_TASK_ID)
# SEED_IDS=(
# 689
# 510
# 550
# )
# SEED_ID=${SEED_IDS[$SLURM_ARRAY_TASK_ID]}

# NODES_PER_SEED=15

# INDEX_SLICE=$(python -c "print($SLURM_ARRAY_TASK_ID % $NODES_PER_SEED)")
# SEED_ID=${SEED_IDS[$(python -c "print($SLURM_ARRAY_TASK_ID // $NODES_PER_SEED)")]}

echo "Sharding and compressing seed id ${SEED_ID}"

DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-seeds/bigscience-catalogue-data/pseudo_crawl_seed--seed-id--"$SEED_ID"
SAVE_DATASET_PATH=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-compressed-shards/bigscience-catalogue-data/seed_id="$SEED_ID"

mkdir -p $SAVE_DATASET_PATH

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python cc_pseudo_crawl/python_scripts/shard_and_compress.py \
    --dataset-path $DATASET_PATH \
    --max-size 10_000_000_000 \
    --num-proc 4 \
    --save-num-proc 10 \
    --save-path $SAVE_DATASET_PATH \
    --save-batch-size 10
    # --index-slice $INDEX_SLICE
    # --total-number-slice $NODES_PER_SEED
