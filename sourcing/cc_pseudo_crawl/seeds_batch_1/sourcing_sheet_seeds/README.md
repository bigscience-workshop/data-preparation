# Pseudo-Crawl Data Sourcing Candidate Seeds Spreadsheet

Source: https://docs.google.com/spreadsheets/d/1DNLAGz--qvLh-0qQ7pMPGiNeUMgp-fRgn-8mbLagC7U/edit#gid=513216703 (timestamp 2021-11-28, reverted edits by anonymous user on record 16 - diariovasco.com), exported as [candidate_websites_for_crawling.csv](./candidate_websites_for_crawling.csv)

Steps:

1. run [cleanup-seeds](./cleanup-seeds.ipynb) to prepare a clean seed list

2. do the lookups / table join, see [general instructions](../README.md) using the crawl selector `CC-MAIN-202[01]` to restrict the join for the last 2 years

3. prepare [coverage metrics](./cc-metrics.ipynb)
