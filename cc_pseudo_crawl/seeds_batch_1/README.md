# Extracting Content from Common Crawl for Curated List of Sites

aka. "pseudo-crawls"

- tools to extract content from Common Crawl for curated list of sites
- metrics about planned and ongoing pseudo-crawls to understand their coverage (size, languages, content types, etc.)

## Preliminary Steps

- create AWS account in order to use [Athena](https://aws.amazon.com/athena/) to perform the lookups
- in Athena, create database `ccindex` and table `ccindex`, see https://commoncrawl.org/2018/03/index-to-warc-files-and-urls-in-columnar-format/
- create the database `bigscience` which holds the joined data and more
  ```sql
  CREATE DATABASE bigscience;
  ```

## Looking Up URLs per Site List

For every site list

1. create a seed table which includes the join column (host or domain name, SURT URL). See [cleanup-seeds](./sourcing_sheet_seeds/cleanup-seeds.ipynb) for an example of this and the following step.

2. export the table to a file, ideally in a columnar format (Parquet or ORC)

3. upload the seed file to S3
  ```
  aws s3 cp seeds.gz.parquet s3://bucket/path/seeds/
  ```
  Note: the S3 path must point to a bucket with write permissions granted. The path needs to be adjusted also in following commands.

3. import the seed table into Athena
  ```sql
  CREATE EXTERNAL TABLE IF NOT EXISTS bigscience.seed (
           `id` int,
           `title` string,
           `link` string,
           `url_path_prefix` string,
           `url_host_name` string,
           `url_host_registered_domain` string,
           `url_surtkey` string)
  ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
  WITH SERDEPROPERTIES (
    'serialization.format' = '1'
  ) LOCATION 's3://bucket/path/seeds/'
  TBLPROPERTIES ('has_encrypted_data'='false');
  ```

4. join the seeds table crawl by crawl with Common Crawl's index, creating a temporary table which is later used as one partition of the result table
   ```
   python3 cc_lookup_seed.py s3://bucket/path seeds "CC-MAIN-202[01]"
   ```
   This will run the join for all crawls of the year 2021 and put the join data into `s3://bucket/path/cc`.

5. finally, create a table holding the result data in order to get further metrics or prepare the content export
  ```sql
  CREATE EXTERNAL TABLE IF NOT EXISTS bigscience.cc_seed (
      seed_id                     INT,
      url_surtkey                 STRING,
      url_host_tld                STRING,
      url_host_registered_domain  STRING,
      url_host_name               STRING,
      url                         STRING,
      fetch_status              SMALLINT,
      fetch_time               TIMESTAMP,
      warc_filename               STRING,
      warc_record_offset             INT,
      warc_record_length             INT,
      fetch_redirect              STRING,
      content_mime_detected       STRING,
      content_languages           STRING)
  PARTITIONED BY (
      crawl  STRING,
      subset STRING)
  STORED AS parquet
  LOCATION 's3://bucket/path/cc/'
  TBLPROPERTIES (
    'has_encrypted_data'='false',
    'parquet.compression'='GZIP');
  ```

6. load the partitions of the join table
   ```sql
   MSCK REPAIR TABLE bigscience.cc_seed;
   ```

7. We want to run deduplication in terms of urls.
  ```sql
  CREATE TABLE bigscience.cc_seed_dedup_url
    WITH (external_location = 's3://bucket/path/cc-seed_dedup_url/',
          partitioned_by = ARRAY['subset'],
          format = 'PARQUET',
          parquet_compression = 'GZIP')
    AS
    WITH tmp AS (
        SELECT *, row_number() over (partition by url order by fetch_time desc) row
        FROM bigscience.cc_seed
    )

    SELECT
       seed_id,
       url,
       url_surtkey,
       url_host_tld,
       url_host_registered_domain,
       url_host_name,
       fetch_status,
       fetch_time,
       warc_filename,
       warc_record_offset,
       warc_record_length,
       fetch_redirect,
       content_mime_detected,
       content_languages,
       subset
    FROM tmp
    WHERE row = 1
  ```
8. Run `download_warc.py` The script will download warc files. We provide two helpers: a bash script and a slurm script for easy use

10. Run `preprocess_dataset.py`. This will help populate specific columns such as `outgoing_links`, `depth` ...

9. Send dataset in a bucket somewhere.
