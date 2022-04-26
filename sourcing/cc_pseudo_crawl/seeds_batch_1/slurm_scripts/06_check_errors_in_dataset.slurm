#!/bin/bash
#SBATCH --job-name=pseudo_crawl_check_erros_in_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uty16tp/code/big_science/logs/re_dowload/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

DATA_SOURCING_REPO=$WORK/code/big_science/data-preparation/sourcing
pushd $DATA_SOURCING_REPO

DATASET_DIR=$six_ALL_CCFRSCRATCH/pseudo_crawl/seeds_batch_1/datasets-redownload/bigscience-catalogue-data
echo $DATASET_DIR

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

python -m cc_pseudo_crawl.python_scripts.check_erros_in_dataset \
    --dataset-dir $DATASET_DIR
