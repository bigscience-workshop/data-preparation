# Code Dataset Sourcing

The code fraction of the BigScience dataset was sourced from the BigQuery [GitHub dataset](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code).

The query to create the dataset can be found in `query.sql`. After creation the dataset was preprocessed with `processing.py`. Note that there is a bug in the script that filters only for GPL licenses instead of filtering them out. There are instructions to remove the bug but it is left there for reproducibility.