# Pseudo-crawl dataset

The scripts in this folder are used to build up a text dataset from the web pages of several domain names from original common crawl WARC files.

## Context

For this pipeline, first 605 seeds - grouped under the name seeds batch 1 - were identified. After extracting text from them, a new batch of 9 seeds - grouped under the name  seeds batch 2 - was added in order to obtain more text in certain languages.  Finally, the texts extracted from these two batches of seeds were cleaned and deduplicated.

## Folder organization

All the scripts (bash/slurm) used to extract the text from batch 1 are grouped in the `seeds_batch_1` folder, all those for batch 2 in `seeds_batch_2` and finally the scripts used on both batches in `seeds_batch_1_2`.

In the `python_scripts` folder are the python scripts that are called by the scripts mentioned in the previous paragraph.

Finally, the `language_annotation` folder gathers all the scripts (bash/slurm/python) developed to identify the languages of the texts.

## Pseudo Crawl dataset creation pipeline

### Batch 1

- **Step 0**: Create a seed-to-WARC's extract mapping using the index from Common Crawl ([CC](https://commoncrawl.org/)).

  Cross the list of domains names with the web pages available on the CC dumps of 2020-2021 to obtain a mapping.

  The WARC format is an archive format that allows to concatenate in a single file several information related to a crawled web page. So in this step we want to: list all the urls belonging to our seeds that have been saved by Common Crawl and then identify where the crawl of each of these urls is stored - i.e. in which WARC file and between which offsets.

  <details>
    <summary>This mapping contains the columns:</summary>

        - 'seed_id'
        - 'url_surtkey'
        - 'url_host_tld'
        - 'url_host_registered_domain'
        - 'url_host_name', 'url'
        - 'fetch_status'
        - 'fetch_time'
        - 'warc_filename'
        - 'warc_record_offset'
        - 'warc_record_length'
        - 'fetch_redirect'
        - 'content_mime_detected'
        - 'content_languages’
    </details>

    Link to documentation to reproduce this step: #TODO isolate relevant information from `sourcing/cc_pseudo_crawl/seeds_batch_1/README.md` and probably clean `data-preparation/sourcing/cc_pseudo_crawl/seeds_batch_1/sourcing_sheet_seeds`

-  **Step 1**: Request Common Crawl to retrieve the extracts from WARC files identified in step 0

    a. Download the WARC's extracts in shards

    The idea is to divide the list of WARC's extracts to be recovered into 10 shards so that 10 different jobs can be used to recover the files listed in one of the shards.
        
    Jobs used:
    - 01_download_warc.slurm
    - 02_download_warc_trial_4.slurm
    - 03_download_warc_trial_5.slurm
    - 04_download_warc_too_big.slurm

    b. Retrying failed downloads for some WARC's extracts 

    The download of some extracts within a shard in step a. may have failed for several reasons (timeout, etc). A second script was therefore run to retry the download of only those extracts that had not been retrieved so far. 

    Job used:
    - 05_redownload_warc.slurm

    c. Verification that all WARC's extracts have been downlaoded

    Job used:
    - 06_check_errors_in_dataset.slurm

-  **Step 2**: Process the  WARC's extracts to 1) isolate the HTML code of different web pages and 2) to retrieve outgoing links from HTML web pages

    The WARC extracts obtained so far are still in a compressed format and contain several types of information. In this step, we decompressed the extracts whose crawled content corresponded to HTML code and we isolated this HTML code in text format. We then processed this HTML code in order to list all the outgoing links appearing in this code. In the end we never reused these lists of outgoing links.

    Job used:
    - 08_preprocess_warc.slurm

-  **Step 3**: Extract text and metadata from the HTML code

    In this step we extracted the text and various metadata from the HTML code using a parser developed by the modeling-metadata BigScience Working Group.

    Job used:
    - 09_extract_text_and_html_metadata.slurm

-  **Step 4**: Sharding the dataset by seed id

    In view of the size of the dataset obtained, it was desirable to store it in a sharded format. We wanted the first level of sharding to correspond to the seed identifiers, i.e. only the content corresponding to a seed could be found in a shard. We then had to use a finer sharding so that each shard in compressed format is about 1GB.

    Jobs used:
    - 10_divide_in_subshards_1000.slurm
    - 11_shard_by_seed_id.slurm
    - 12_merge_seed_shards.slurm

-  **Step 5**: Pushing a light version of the dataset onto the hub

    Job used:
    - 13_shard_and_compress.slurm


### Batch 2

The steps followed for batch 2 are identical to those followed for batch 1. The corresponding slurm scripts have a slightly different numbering because 1) we had already learned from the difficulties of batch 1 and 2) there were far fewer WARC's extracts involved


- **Step 0**: TODO

-  **Step 1**: Request Common Crawl to retrieve the extracts from WARC files identified in step 0

    a. Download the WARC's extracts in shards
        
    Job used:
    - 01_download_warc.slurm

    b. Retrying failed downloads for some WARC's extracts 

    Job used:
    - 02_redownload_warc.slurm
    - 02b_redownload_warc.slurm

    c. Verification that all WARC's extracts have been downlaoded

    Job used:
    - 03_check_errors_in_dataset.slurm

-  **Step 2**: Process the  WARC's extracts to 1) isolate the HTML code of different web pages and 2) to retrieve outgoing links from HTML web pages

    Job used:
    - 05_preprocess_warc.slurm

-  **Step 3**: Extract text and metadata from the HTML code

    Job used:
    - 06_extract_text_and_html_metadata.slurm

-  **Step 4**: Sharding the dataset by seed id

    Jobs used:
    - 07_shard_by_seed_id.slurm
    - 08_merge_seed_shards.slurm

-  **Step 5**: Pushing a light version of the dataset onto the hub
    Jobs used:
    - 09_shard_and_compress.slurm
    - 10_push_to_hub.slurm

### Batch 1 & 2

Finally, we have applied specific cleaning / deduplication steps to the pseudo-crawl

1. We have removed content in the texts that is repeated a lot between examples of the same seed id. The idea here was to remove website templates
    
    Job used:
    - 00_clean_dataset.slurm

2. We removed examples that were duplicated identically
    
    Job used:
    - 01_exact_deduplicates.slurm

3.  We deduplicated the exactly identical lines within a single example.

    python script used:
    - doc_by_doc_line_dedup.py

### Analysis scripts

In addition to the pipeline to create the dataset, we also produced some scripts to analyse the contents of the dataset. In particular:

- Several language detection methods: language_annotation folder
- A script to find out the number of examples of various types: `get_stats.py`
