## Data Filtering and Data Deduplication of the BigScience Corpus

This is the data filtering code used to build the BigScience Corpus.

The supported languages are defined in the file [languages_id.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/languages_id.py).


### Filtering

#### 0. Understand the filtering pipeline

Take a look at the pdf [explanation_filtering_pipeline.pdf](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/explanation_filtering_pipeline.pdf) for an explanation of the filtering pipeline.

#### 1. Define the lists of stop words and flagged words, and check how the anonymization and the normalization of texts are done

You might want to redefine the lists of stop words (closed class words) and flagged words for robustness or ethical reasons in the files [stopwords.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/stopwords.py) and [flagged_words.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/flagged_words.py).

Less importantly, you can also check how the anonymization and the normalization of texts are done in the files [anonymization.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/anonymization.py) and [normalization.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/normalization.py) (if applicable, default is not to use both).

#### 2. Download everything you need

To run the filtering code, it is necessary to download the dataset on which the filtering will take place, but also the necessary models, which are the Fasttext model for language identification (download [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)) and the Sentencepiece and KenLM models for tokenization and calculation of perplexity scores (download with the file [download_sentencepiece_kenlm_models.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/download_sentencepiece_kenlm_models.py)).

#### 3. Choose the filtering parameters

The filtering parameters for each language are to be specified in the file [parameters_filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/parameters_filtering.py). It is strongly recommended to look at the data and use the visualization code in the directory [visualization](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/filtering/visualization) to choose these parameters.

#### 4. Run the filtering

Run the filtering with the file [main_filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/main_filtering.py), specifying the dataset used and the links to the downloaded models. The different filters are coded in the file [filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/filtering/filtering.py).

#### 5. Do the deduplication

Do the deduplication, which is detailed in the sub folder [deduplicate](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/filtering/deduplicate).
