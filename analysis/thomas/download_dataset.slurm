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

  #SBATCH --account=six@a100
  #SBATCH --constraint=a100
  #SBAYCJ --partition=gpu_p5
  #SBATCH --gres=gpu:1

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling

pushd $DATA_PREPARATION_PATH/analysis/thomas

python download_dataset.py