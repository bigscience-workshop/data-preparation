#!/bin/bash
#SBATCH --job-name=convert_all_catalogue_to_jsonl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/convert_all_catalogue_to_jsonl/%x-%j.out           # output file name
#SBATCH --array=0-509   # TODO: modify according to the number of models you want to evaluated
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

TRAINING_DATA_PREPARATION_REPO=$WORK/code/big_science/data-preparation/preprocessing/training
pushd $TRAINING_DATA_PREPARATION_REPO

# ===== GET DATASET ======

DATASET_ID=$SLURM_ARRAY_TASK_ID
DATASET_NAME=$(python 03_convert_to_jsonl/get_dataset_name.py --dataset-csv-path 01a_catalogue_cleaning_and_filtering/data/training.csv --index $DATASET_ID)

echo "Converting $DATASET_NAME to json"

NUM_SHARDS=8

DATASET_PATH=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/clean/$DATASET_NAME/final
SAVE_PREFIX=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/cleaned_jsonl_"$NUM_SHARDS"_shards/$DATASET_NAME/data
mkdir -p $(dirname $SAVE_PREFIX)

echo "Downloading ${DATASET_NAME}"
export HF_DATASETS_OFFLINE=1

module load git-lfs
module load unrar
python -c "
from datasets import load_dataset
from pathlib import Path

ds = load_from_disk(\"$DATASET_PATH\", split=\"train\")

for shard_id in range($NUM_SHARDS):
    filepath = Path(f\"${SAVE_PREFIX}_{shard_id}_$NUM_SHARDS.jsonl\")
    if filepath.exists():
        continue
    shard = ds.shard(num_shards=$NUM_SHARDS,index=shard_id)
    tempfilepath = Path(f\"{str(filepath.absolute())}.tmp\")
    shard.to_json(tempfilepath, num_proc=10)
    tempfilepath.rename(filepath)
"
