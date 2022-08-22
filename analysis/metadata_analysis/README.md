# Metadata Analysis
This folder contains a tool to estimate the size per language of the catalogue datasets. It also contains the metadata and dataset names used for the catalogue part of the BLOOM training corpus. There are three files in this folder:

- `datasets.txt`: contains all the catalogue dataset names 
- `meta.jsonl`: contains the metadata of each of the catalogue datasets as created by `datasets`
- `metadata_analysis.py`: a script to anlayse the size per language based on the former to files.

_Note 1:_ The `datasets` library computes the sizes of the dataset including the meta information which might slightly inflate the values. For a more exact estimate only the `"text"` field inside the datasets should be considered.

_Note 2:_ The ground truth for the datasets was a snapshot of [this google sheet](https://docs.google.com/spreadsheets/d/1yGLp1k_XO0Q_Ozjm6DZKiWPgFZaYBIaDQ3zzncXzI3o/edit#gid=0). This data might be out of date and should be treated as such.