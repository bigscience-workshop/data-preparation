conda activate dedup-dataset

TOKENIZER_DATA_PREPARATION_REPO=~/code/big_science/data-preparation/preprocessing/tokenizer
pushd $TOKENIZER_DATA_PREPARATION_REPO

DATASET_PATH=~/data/tokenization_dataset/alpha_arrow
SAVE_DATASET_DIR=~/data/tokenization_dataset/alpha_arrow-dedup

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=~/to_delete

python python_script/ram_dedup_lines.py \
    --save-dir $SAVE_DATASET_DIR \
    --dataset_dir $DATASET_PATH \
    --num-proc 1 \
    --batch-size 6000000 \
    --load-from-disk
