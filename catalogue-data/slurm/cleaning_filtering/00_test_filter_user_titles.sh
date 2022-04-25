conda activate datacatalog

CATALOGUE_DATA_REPO="/home/lucile/code/catalogue_data"

cd $CATALOGUE_DATA_REPO

python clean.py \
    --dataset-path bigscience-catalogue-lm-data/lm_en_wikinews_filtered \
    --preprocessings filter_wiki_user_titles \
    --save-path /home/lucile/data/result_filtering_cleaning/lm_en_wikinews_filtered.jsonl \
    --num-proc 4 \
    --batch-size 100