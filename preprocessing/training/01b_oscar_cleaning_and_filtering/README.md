## Data Filtering and Data Deduplication of the BigScience Corpus

This is the data filtering code used to clean the Oscar subset of the ROOTS dataset.

The supported languages are defined in the file [languages_id.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/languages_id.py).


### Filtering

#### 0. Understand the filtering pipeline

Take a look at the pdf [explanation filtering pipeline](https://drive.google.com/file/d/1cCJ8sWE88TRLDAa3eHLmXO4JlkR2QzLY/view?usp=sharing) for an explanation of the filtering pipeline.

#### 1. Define the lists of stop words and flagged words, and check how the anonymization and the normalization of texts are done

You might want to redefine the lists of stop words (closed class words) and flagged words for robustness or ethical reasons in the files [stopwords.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/stopwords.py) and [flagged_words.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/flagged_words.py).

Less importantly, you can also check how the anonymization and the normalization of texts are done in the files [anonymization.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/anonymization.py) and [normalization.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/normalization.py) (if applicable, default is not to use both).

#### 2. Download everything you need

To run the filtering code, it is necessary to download the dataset on which the filtering will take place, but also the necessary models, which are the Fasttext model for language identification (download [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)) and the Sentencepiece and KenLM models for tokenization and calculation of perplexity scores (download with the file [download_sentencepiece_kenlm_models.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/download_sentencepiece_kenlm_models.py)).

#### 3. Choose the filtering parameters

The filtering parameters for each language are to be specified in the file [parameters_filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/parameters_filtering.py). It is strongly recommended to look at the data and use the visualization code in the directory [visualization](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/01b_oscar_cleaning_and_filtering/visualization) to choose these parameters.

#### 4. Run the filtering

Run the filtering with the file [main_filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/main_filtering.py), specifying the dataset used and the links to the downloaded models. The different filters are coded in the file [filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/filtering.py).

Some common issues:
- OSCAR-v2 metadata can cause cryptic Arrow bugs. The `remove_meta` flag will take care of this and/or space issues
- Too-long documents can cause hangs. Use `max_len_prefilter` to remove outliers. 
- Memory issues can arise, causing hard-to-debug hangs if a process dies silently. Reducing the number of processes will help in this case.
- If you dataset is very large, you may have space issues in the saving stage. In this case, you will find an equivalent `.arrow` file in your `datasets` cache (typically the last-modified file in `.cache/huggingface/datasets/<dataset_name>/....`) anyway. The saving stage is mostly for better clarity and to avoid manipulating the `datasets` cache. 

#### 5. Do the deduplication

Do the deduplication, which is detailed in the sub folder [deduplicate](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/01b_oscar_cleaning_and_filtering/deduplicate).
