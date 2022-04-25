conda activate pii

CATALOGUE_DATA_REPO="/home/lucile/catalogue_data"

cd $CATALOGUE_DATA_REPO

python pii/pii_processor.py \
    --save-to-json \
    --dataset-path /home/lucile/data_pii/pre_processing \
    --dataset-name lm_eu_oscar.jsonl \
    --save-path /home/lucile/data_pii/post_processing/lm_eu_oscar.jsonl \
    --save-check-path /home/lucile/data_pii/post_processing_checks/lm_eu_oscar \
    --num-proc 3 \
    --batch-size 1000 \
    --save-batch-size 10000 \
    --check-sampling-size 10000 \
    --check-only-modified