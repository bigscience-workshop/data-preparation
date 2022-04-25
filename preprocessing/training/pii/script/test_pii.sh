conda activate pii

TRAINING_DATA_PREPARATION_REPO=~/code/big_science/data-preparation/preprocessing/training
pushd $TRAINING_DATA_PREPARATION_REPO

python pii/pii_processor.py \
    --save-to-json \
    --dataset-path ~/data_pii/pre_processing \
    --dataset-name lm_eu_oscar.jsonl \
    --save-path ~/data_pii/post_processing/lm_eu_oscar.jsonl \
    --save-check-path ~/data_pii/post_processing_checks/lm_eu_oscar \
    --num-proc 3 \
    --batch-size 1000 \
    --save-batch-size 10000 \
    --check-sampling-size 10000 \
    --check-only-modified