conda activate dedup-dataset

DATA_TOOLING_REPO=/home/lucile/code/data_tooling

DATASET_PATH=/home/lucile/data/tokenization_dataset/alpha_v2
SAVE_DATASET_DIR=/home/lucile/data/tokenization_dataset/alpha_v2_dedup

pushd $DATA_TOOLING_REPO

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/home/lucile/to_delete

python tokenizer/python_script/ram_dedup_lines.py \
    --save-dir $SAVE_DATASET_DIR \
    --dataset_dir $DATASET_PATH \
    --num-proc 1 \
    --batch-size 6000000
