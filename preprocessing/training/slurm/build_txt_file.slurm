#!/bin/bash
#SBATCH --job-name=build_tokenization_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/build_tokenization_dataset/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate thomas_data_tooling # Debug deepspeed temporarily

CATALOGUE_DATA_REPO=$WORK/code/big_science/catalogue_data

export HF_DATASETS_OFFLINE=1
pushd $CATALOGUE_DATA_REPO

python build_txt_file.py \
  --dataset-path /gpfsscratch/rech/six/commun/tokenizer/dataset/tokenization_dataset_v2 \
  --save-file /gpfsscratch/rech/six/commun/tokenizer/dataset/tokenization_dataset_v2/data.txt