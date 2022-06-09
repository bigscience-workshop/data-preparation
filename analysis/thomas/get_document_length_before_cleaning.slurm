#!/bin/bash
#SBATCH --job-name=get_document_length_before_cleaning    #   job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40                                               # number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 06:00:00                                                 # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out   # output file name
#SBATCH --account=six@cpu                                               # account

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

DATA_PREPARATION_PATH=/gpfswork/rech/six/uty16tp/code/big_science/data-preparation
TRAINING_CSV_PATH=$DATA_PREPARATION_PATH/preprocessing/training/training_v2.csv

pushd $DATA_PREPARATION_PATH/analysis/thomas

python get_document_length_before_cleaning.py
    --num-proc 40
    --training-csv-path
