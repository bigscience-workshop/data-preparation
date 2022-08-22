#  Processing Pipeline for Quality Improvement on Crowdsourced Datasets

This folder contains the pipeline that has been used to apply cleaning, filtering or deduplication functions to each dataset.

# Example usage

```python
python clean.py \
    --dataset-path bigscience-catalogue-lm-data/lm_en_wikinews_filtered \
    --preprocessings filter_wiki_user_titles \
    --save-path ~/data/result_filtering_cleaning/lm_en_wikinews_filtered.jsonl \
    --num-proc 4 \
    --batch-size 100
```

# Folder organization

- clean.py: main script
- clean_helpers: folder containing all the processings functions that can be applied to a dataset
- slurm: The actual bash scripts we ran to get our dataset
- data: the mapping between each dataset and the processings scripts that have been applied to it
- streamlit-app: an application allowing to visualize the modifications made by the pipeline (see the space [bigscience-catalogue-lm-data/process-pipeline-visualizer](https://huggingface.co/spaces/bigscience-catalogue-lm-data/process-pipeline-visualizer) )

## Necessary downloads for cleaning

In order to work, the pipeline needs to have some additional software/data available, which you can download according to the following instructions

### Stanza

```python
import stanza

for lang in {"ar", "ca", "eu", "id", "vi", "zh-hans", "zh-hant"}:
    stanza.download(lang, logging_level="WARNING")
```

### Indic NLP library

```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
export INDIC_RESOURCES_PATH=<PATH_TO_REPO>
```

### NLTK
import nltk
nltk.download("punkt")
