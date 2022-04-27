## Big Science - Automated Classification & Dataset Curation - AC/DC

This is the data filtering code for BigScience.

See [this document](https://docs.google.com/document/d/1bx7lzAIWALH2IX5PLAiRfkHr3025dC-ZYkEmq4zB2gI/edit) for more details.

The supported languages are defined in the file [languages_id.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/languages_id.py).


### Filtering

#### 0. Understand the filtering pipeline

Take a look at the pdf [explanation_filtering_pipeline.pdf](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/explanation_filtering_pipeline.pdf) for an explanation of the filtering pipeline.

#### 1. Define the lists of stop words and flagged words, and check how the anonymization and the normalization of texts are done

You might want to redefine the lists of stop words and flagged words for robustness or ethical reasons in the files [stopwords.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/stopwords.py) and [flagged_words.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/flagged_words.py).

Less importantly, you can also check how the anonymization and the normalization of texts are done in the files [anonymization.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py) and [normalization.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/normalization.py) (if applicable, default is to use the anonymization and not to use the normalization).

#### 2. Download everything you need

To run the filtering code, it is necessary to download the dataset on which the filtering will take place, but also the necessary models, which are the Fasttext model for language identification (download [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)) and the Sentencepiece and KenLM models for tokenization and calculation of perplexity scores (download with the file [download_sentencepiece_kenlm_models.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/download_sentencepiece_kenlm_models.py)).

#### 3. Choose the filtering parameters

The filtering parameters for each language are to be specified in the file [parameters_filtering.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/parameters_filtering.py). It is strongly recommended to look at the data and use the visualization code in the directory [visualization](https://github.com/bigscience-workshop/data_tooling/tree/master/ac_dc/visualization) to choose these parameters.

#### 4. Run the filtering

Run the filtering with the file [main_filtering.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/main_filtering.py), specifying the dataset used and the links to the downloaded models. The different filters are coded in the file [filtering.py](https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/filtering.py).

#### 5. Do the deduplication

Do the deduplication, which is detailed in the sub folder `ac_dc/deduplicate`.
