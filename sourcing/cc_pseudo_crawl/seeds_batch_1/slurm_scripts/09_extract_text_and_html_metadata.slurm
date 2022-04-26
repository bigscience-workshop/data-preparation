#!/bin/bash
#SBATCH --job-name=pseudo_crawl_preprocess_extract_text             # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 2:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/pseudo_crawl_preprocess_extract_text-v3/%j-%x.out   # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/pseudo_crawl_preprocess_extract_text-v3/%j-%x.err    # error file name
#SBATCH --account=six@cpu                                               # account
#SBATCH --array=0-99
#SBATCH --partition=cpu_p1

set -x -e

source $HOME/start-modelling-metadata-user

cd $WORK/repos/sync/metadata/

# We are on an offline partition
export HF_DATASETS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH

METADATA_TO_INCLUDE='["html"]'

DATASET_FILES_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-preprocessed/bigscience-catalogue-data
OUT_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-preprocessed-text-extracted/bigscience-catalogue-data

mkdir -p $OUT_DIR

MAP_BATCH_SIZE=1
PREPROCESSING_NUM_WORKERS=80
NUM_FILES_TO_PROCESS=1
SAVE_BATCH_SIZE=1000

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

echo "Args:"
echo "    task_id=${SLURM_ARRAY_TASK_ID}"
echo "    map_batch_size=$MAP_BATCH_SIZE"
echo "    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS"
echo "    out_dir=$OUT_DIR"
echo "    dataset_name=$DATASET_FILES_DIR"
echo "    metadata_to_include=$METADATA_TO_INCLUDE"

python -m cc_pseudo_crawl.python_scripts.extract_text.extract_text_and_html_metadata\
    task_id=${SLURM_ARRAY_TASK_ID}\
    out_dir=$OUT_DIR \
    dataset_name=$DATASET_FILES_DIR \
    metadata_to_include="$METADATA_TO_INCLUDE" \
    map_batch_size=$MAP_BATCH_SIZE \
    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS\
    num_files_to_process=$NUM_FILES_TO_PROCESS\
    save_batch_size=$SAVE_BATCH_SIZE\
    project_name=pseudo_crawl_extract\
    use_load_from_disk=true
