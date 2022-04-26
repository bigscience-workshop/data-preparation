conda activate dedup-dataset

TOKENIZER_DATA_PREPARATION_REPO=~/code/big_science/data-preparation/preprocessing/tokenizer
pushd $TOKENIZER_DATA_PREPARATION_REPO

DATASET_PATH=~/data/tokenization_dataset/alpha_v2_dedup
SAVE_DATASET_DIR=~/data/tokenization_dataset/alpha_v2_dedup_lines_and_article

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=~/to_delete

python python_script/dedup_exact_article.py \
    --save-dir $SAVE_DATASET_DIR \
    --dataset_dir $DATASET_PATH \
    --num-proc 8 \
    --batch-size 6000000
