#!/bin/bash
#SBATCH --job-name=download_dataset    #   job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10                                               # number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 06:00:00                                                 # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out   # output file name
#SBATCH --account=six@cpu                                               # account
#SBATCH --partition=compil

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

module load git-lfs
module load unrar

DATA_PREPARATION_PATH=/gpfswork/rech/six/uty16tp/code/big_science/data-preparation

pushd $DATA_PREPARATION_PATH/analysis/thomas

python download_dataset.py