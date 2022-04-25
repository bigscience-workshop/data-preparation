conda activate datacatalog

TRAINING_DATA_PREPARATION_REPO=$WORK/code/big_science/data-preparation/preprocessing/training
pushd $TRAINING_DATA_PREPARATION_REPO

python clean.py \
    --dataset-path bigscience-catalogue-lm-data/lm_en_wikinews_filtered \
    --preprocessings filter_wiki_user_titles \
    --save-path ~/data/result_filtering_cleaning/lm_en_wikinews_filtered.jsonl \
    --num-proc 4 \
    --batch-size 100