# Data prepation

This repository contains all the tools and code used to build the ROOTS dataset produced by the BigScience initiative to train the BLOOM models as well as a reduced version to train the tokenizer.

## General pipeline for the preparation of the ROOTS dataset
![diagram_preprocessing_roots](https://user-images.githubusercontent.com/55560583/186378062-2c4691b1-67a8-4fd9-a7c4-a14a98cbc9d1.jpg)

More detail on the process, including the specifics of the *cleaning*, *filtering*, and *deduplication* operations, can be found in Sections 2 "(Crowd)Sourcing a Language Resource Catalogue" and 3 "Processing OSCAR" of our paper [on the ROOTS dataset creation](https://openreview.net/forum?id=UoEw6KigkUn).

## Key ressources
### [Code for making the Pseudo-Crawl dataset](sourcing/cc_pseudo_crawl)

### [Filtering library used to filter OSCAR](preprocessing/training/01b_oscar_cleaning_and_filtering)

### [Code used to run preprocessing pipeline on crowdsourced dataset](preprocessing/training/01a_catalogue_cleaning_and_filtering)

### [Code used for the tokenizer's training dataset](preprocessing/tokenizer)

### [Code used for making the analysis and plots for the paper](analysis)
