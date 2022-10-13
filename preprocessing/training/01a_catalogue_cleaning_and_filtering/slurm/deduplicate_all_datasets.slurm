#!/bin/bash
#SBATCH --job-name=deduplicate_all_catalogue_datasets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/deduplicate_all_catalogue_datasets/%x-%j.out           # output file name
#SBATCH --array=0-516
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

WORKING_DIR=$WORK/code/big_science/data-preparation/preprocessing/training/01a_catalogue_cleaning_and_filtering
pushd $WORKING_DIR

# =======  GET DATASET AND FUNCTIONS ======

DATASET_ID=$SLURM_ARRAY_TASK_ID
DATASET_TRAINING_CSV=$WORKING_DIR/data/training.csv
readarray -t DATASET_PATH_AND_FUNCTIONS < <(python -c "
import pandas as pd

data = pd.read_csv(\"${DATASET_TRAINING_CSV}\")
dataset = data.iloc[$DATASET_ID]
print(dataset[\"dataset_name\"])

if \"pseudocrawl\" in dataset[\"dataset_name\"]:
    list_of_dedups=[\"dedup_document_on_url\", \"dedup_document\", \"dedup_pseudocrawl_newspapers\", \"filter_remove_empty_docs\"]
else:
    list_of_dedups=[\"dedup_document\", \"dedup_template_soft\", \"filter_remove_empty_docs\"]
print(\" \".join(list_of_dedups))

if pd.notnull(dataset[\"--preprocessings\"]):
    print(dataset[\"--preprocessings\"])
else:
    print()
")
DATASET_PATH=${DATASET_PATH_AND_FUNCTIONS[0]}
PREPROCESSINGS=${DATASET_PATH_AND_FUNCTIONS[1]}
OFFICIAL_PREPROCESSINGS=${DATASET_PATH_AND_FUNCTIONS[2]}

if [[ $OFFICIAL_PREPROCESSINGS == KILL ]]
then
    echo "Not supposed to preprocess $DATASET_PATH"
    exit 0
fi

# ====== OBTAIN DATASET_NAME =======

if [[ $DATASET_PATH == /gpfswork/rech/six/uty16tp/dataset/tokenization/* ]]
then
    if [[ $DATASET_PATH == */data ]]
    then
        DATASET_NAME=${DATASET_PATH:48:-5}
    else
        DATASET_NAME=${DATASET_PATH:48}
    fi
else
    DATASET_NAME=$DATASET_PATH
fi

# ====== RUN PYTHON SCRIPT =======

BASE_PATH=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/dedup/$DATASET_NAME
SAVE_PATH=$BASE_PATH/final
CHECKS_SAVE_PATH=$BASE_PATH/checks
LOGS_PATH=$BASE_PATH/logs.txt

mkdir -p $(dirname $SAVE_PATH)
mkdir -p $CHECKS_SAVE_PATH

export HF_DATASETS_OFFLINE=1

python clean.py \
    --dataset-path $DATASET_PATH \
    --preprocessings $PREPROCESSINGS \
    --save-path $SAVE_PATH \
    --checks-save-path $CHECKS_SAVE_PATH \
    --num-proc 10 \
    --batch-size 100 \
    --sampling-size-map-checks 1000 \
    --sampling-size-filter-checks 1000 \
    2>&1 | tee $LOGS_PATH
