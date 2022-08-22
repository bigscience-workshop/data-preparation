conda activate datacatalog

WORKING_DIR=$WORK/code/big_science/data-preparation/preprocessing/catalogue_cleaning/01_quality_improvement_processing_pipeline
pushd $WORKING_DIR

python clean.py \
    --dataset-path bigscience-catalogue-lm-data/lm_en_wikinews_filtered \
    --preprocessings filter_wiki_user_titles \
    --save-path ~/data/result_filtering_cleaning/lm_en_wikinews_filtered.jsonl \
    --num-proc 4 \
    --batch-size 100