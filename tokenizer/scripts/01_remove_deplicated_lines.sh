conda activate dedup-dataset

DATA_TOOLING_REPO=/home/lucile/code/data_tooling

DATASET_PATH=/home/lucile/data/tokenization_dataset/alpha-subset-12M
SAVE_DATASET_DIR=/home/lucile/data/tokenization_dataset/alpha-subset-12M-dedup-lines

pushd $DATA_TOOLING_REPO

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/home/lucile/to_delete

python tokenizer/python_script/dedup_lines.py \
    --save-dir $SAVE_DATASET_DIR \
    --dataset_dir $DATASET_PATH \
    --batch-size 100 \
    --num-proc 3 \
    --min-chars 0 \
    --n-records 1000000 \
    --min-repetition-threshold 0
