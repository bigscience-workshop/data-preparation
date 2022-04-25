conda activate dedup-dataset

TOKENIZER_DATA_PREPARATION_REPO=~/code/big_science/data-preparation/preprocessing/tokenizer
pushd $TOKENIZER_DATA_PREPARATION_REPO

DATASET_PATH=~/data/tokenization_dataset/alpha-subset-12M
SAVE_DATASET_DIR=~/data/tokenization_dataset/alpha-subset-12M-dedup-lines

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=~/to_delete

python python_script/dedup_lines.py \
    --save-dir $SAVE_DATASET_DIR \
    --dataset_dir $DATASET_PATH \
    --batch-size 100 \
    --num-proc 3 \
    --min-chars 0 \
    --n-records 1000000 \
    --min-repetition-threshold 0
