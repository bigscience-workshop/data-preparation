# Data prepation

This repository contains all the tools and code used to build the ROOTS dataset produced by the BigScience initiative to train the BLOOM models as well as a reduced version to train the tokenizer.

## General pipeline for the preparation of the ROOTS dataset
![diagram_preprocessing_roots](https://raw.githubusercontent.com/bigscience-workshop/data-preparation/main/roots_pipeline.png)

More detail on the process, including the specifics of the *cleaning*, *filtering*, and *deduplication* operations, can be found in Sections 2 "(Crowd)Sourcing a Language Resource Catalogue" and 3 "Processing OSCAR" of our paper [on the ROOTS dataset creation](https://openreview.net/forum?id=UoEw6KigkUn).

## Key resources
### [Code for making the Pseudo-Crawl dataset](sourcing/cc_pseudo_crawl)

### [Filtering library used to filter OSCAR](preprocessing/training/01b_oscar_cleaning_and_filtering)

### [Code used to run preprocessing pipeline on crowdsourced dataset](preprocessing/training/01a_catalogue_cleaning_and_filtering)

### [Code used for the tokenizer's training dataset](preprocessing/tokenizer)

### [Code used for making the analysis and plots for the paper](analysis)

## Citation
```
@inproceedings{
bigscience-roots:2022,
title={The BigScience {ROOTS} Corpus: A 1.6{TB} Composite Multilingual Dataset},
author={Hugo Lauren{\c{c}}on and Lucile Saulnier and Thomas Wang and Christopher Akiki and Albert Villanova del Moral and Teven Le Scao and Leandro Von Werra and Chenghao Mou and Eduardo Gonz{\'a}lez Ponferrada and Huu Nguyen and J{\"o}rg Frohberg and Mario {\v{S}}a{\v{s}}ko and Quentin Lhoest and Angelina McMillan-Major and G{\'e}rard Dupont and Stella Biderman and Anna Rogers and Loubna Ben allal and Francesco De Toni and Giada Pistilli and Olivier Nguyen and Somaieh Nikpoor and Maraim Masoud and Pierre Colombo and Javier de la Rosa and Paulo Villegas and Tristan Thrush and Shayne Longpre and Sebastian Nagel and Leon Weber and Manuel Romero Mu{\~n}oz and Jian Zhu and Daniel Van Strien and Zaid Alyafeai and Khalid Almubarak and Vu Minh Chien and Itziar Gonzalez-Dios and Aitor Soroa and Kyle Lo and Manan Dey and Pedro Ortiz Suarez and Aaron Gokaslan and Shamik Bose and David Ifeoluwa Adelani and Long Phan and Hieu Tran and Ian Yu and Suhas Pai and Jenny Chim and Violette Lepercq and Suzana Ilic and Margaret Mitchell and Sasha Luccioni and Yacine Jernite},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022},
url={https://openreview.net/forum?id=UoEw6KigkUn}
}
```
