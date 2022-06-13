## 1. Download all the datasets from the hub

This step aims to download all the datasets constituting the BigScience corpus.

For this, we used a slurm script [`slurm-scrips/download_ds.slurm`](slurm-scrips/download_ds.slurm). If you do not have slurm, you can still take inspiration from it to download all the datasets. [`python-scripts/get_list_of_datasets.py`](python-scripts/get_list_of_datasets.py) was used to get the list of all the sub datasets composing the BigScience corpus (the resulting list is hardcoded in [`slurm-scrips/download_ds.slurm`](slurm-scrips/download_ds.slurm)).

## 2. Compute the size of each document

The next step is to create a metadata dataset for which we will store only the length of the associated text for each example. 

For this, we also used a slurm script [`slurm-scrips/count_len_doc.slurm`](slurm-scrips/count_len_doc.slurm). If you do not have slurm, the downloading script is[`python-scripts/count_len_doc.py](python-scripts/count_len_doc.py).
## 3. Compute some statistics per datasets and group them by language

Example of bash command to generate the necessary statistics of the plots in step 4.
```bash
python analysis/document-sizes/python-scripts/compute_stats.py \
    --bigscience_corpus_doc_length_dir "/home/lucile_huggingface_co/data/bigscience_corpus_doc_length" \
    --statistics-pickle-file "/home/lucile_huggingface_co/data/all_data_point.pickle"
```

## 4. Produce plots

Example of bash commands to generate all the plots.

Plots per dataset and per language:
```bash
python analysis/document-sizes/python-scripts/plot_per_ds_per_lang.py \
    --save-plot-dir "/home/lucile_huggingface_co/data/boxplot_per_ds_per_lang" \
    --statistics-pickle-file "/home/lucile_huggingface_co/data/all_data_point.pickle"
```
Plot per language:
```bash
python analysis/document-sizes/python-scripts/plot_per_lang.py
    --plot-file-path "/gpfsscratch/rech/cnw/uue59kq/boxplot_per_lang.png" \
    --statistics-pickle-file "/gpfsscratch/rech/cnw/uue59kq/all_data_point.pickle"
```