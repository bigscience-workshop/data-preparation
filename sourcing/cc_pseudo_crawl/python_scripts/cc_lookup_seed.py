#!/usr/bin/python3

# iterate over monthly crawls and store
# the joined data as a partition of the result table

import logging
import re
import sys

from pyathena import connect

logging.basicConfig(
    level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

join_template = """
CREATE TABLE {db}._tmp_overlap
WITH (external_location = '{s3_location}/crawl={crawl}/',
      partitioned_by = ARRAY['subset'],
      format = 'PARQUET',
      parquet_compression = 'GZIP')
AS SELECT
       {tid}.id              AS seed_id,
       cc.url_surtkey        AS url_surtkey,
       cc.url_host_tld       AS url_host_tld,
       cc.url_host_registered_domain AS url_host_registered_domain,
       cc.url_host_name      AS url_host_name,
       cc.url                AS url,
       cc.fetch_status       AS fetch_status,
       cc.fetch_time         AS fetch_time,
       cc.warc_filename      AS warc_filename,
       cc.warc_record_offset AS warc_record_offset,
       cc.warc_record_length AS warc_record_length,
       cc.fetch_redirect     AS fetch_redirect,
       cc.content_mime_detected AS content_mime_detected,
       cc.content_languages  AS content_languages,
       cc.subset             AS subset
FROM ccindex.ccindex AS cc
  RIGHT OUTER JOIN {db}.{seed_table} AS {tid}
  ON cc.url_host_registered_domain = {tid}.url_host_registered_domain
     AND strpos(cc.url_surtkey, {tid}.url_surtkey) = 1
WHERE cc.crawl = '{crawl}'
"""

drop_tmp_table = "DROP TABLE `{db}._tmp_overlap`;"

# list of crawls
# Note: in order to get a list of released crawls:
# - query Athena
#    SHOW PARTITIONS ccindex
# - see
#    https://commoncrawl.s3.amazonaws.com/crawl-data/index.html
crawls = [
    "CC-MAIN-2013-20",
    "CC-MAIN-2013-48",
    #
    "CC-MAIN-2014-10",
    "CC-MAIN-2014-15",
    "CC-MAIN-2014-23",
    "CC-MAIN-2014-35",
    "CC-MAIN-2014-41",
    "CC-MAIN-2014-42",
    "CC-MAIN-2014-49",
    "CC-MAIN-2014-52",
    #
    "CC-MAIN-2015-06",
    "CC-MAIN-2015-11",
    "CC-MAIN-2015-14",
    "CC-MAIN-2015-18",
    "CC-MAIN-2015-22",
    "CC-MAIN-2015-27",
    "CC-MAIN-2015-32",
    "CC-MAIN-2015-35",
    "CC-MAIN-2015-40",
    "CC-MAIN-2015-48",
    #
    "CC-MAIN-2016-07",
    "CC-MAIN-2016-18",
    "CC-MAIN-2016-22",
    "CC-MAIN-2016-26",
    "CC-MAIN-2016-30",
    "CC-MAIN-2016-36",
    "CC-MAIN-2016-40",
    "CC-MAIN-2016-44",
    "CC-MAIN-2016-50",
    #
    "CC-MAIN-2017-04",
    "CC-MAIN-2017-09",
    "CC-MAIN-2017-13",
    "CC-MAIN-2017-17",
    "CC-MAIN-2017-22",
    "CC-MAIN-2017-26",
    "CC-MAIN-2017-30",
    "CC-MAIN-2017-34",
    "CC-MAIN-2017-39",
    "CC-MAIN-2017-43",
    "CC-MAIN-2017-47",
    "CC-MAIN-2017-51",
    #
    "CC-MAIN-2018-05",
    "CC-MAIN-2018-09",
    "CC-MAIN-2018-13",
    "CC-MAIN-2018-17",
    "CC-MAIN-2018-22",
    "CC-MAIN-2018-26",
    "CC-MAIN-2018-30",
    "CC-MAIN-2018-34",
    "CC-MAIN-2018-39",
    "CC-MAIN-2018-43",
    "CC-MAIN-2018-47",
    "CC-MAIN-2018-51",
    #
    "CC-MAIN-2019-04",
    "CC-MAIN-2019-09",
    "CC-MAIN-2019-13",
    "CC-MAIN-2019-18",
    "CC-MAIN-2019-22",
    "CC-MAIN-2019-26",
    "CC-MAIN-2019-30",
    "CC-MAIN-2019-35",
    "CC-MAIN-2019-39",
    "CC-MAIN-2019-43",
    "CC-MAIN-2019-47",
    "CC-MAIN-2019-51",
    #
    "CC-MAIN-2020-05",
    "CC-MAIN-2020-10",
    "CC-MAIN-2020-16",
    "CC-MAIN-2020-24",
    "CC-MAIN-2020-29",
    "CC-MAIN-2020-34",
    "CC-MAIN-2020-40",
    "CC-MAIN-2020-45",
    "CC-MAIN-2020-50",
    #
    "CC-MAIN-2021-04",
    "CC-MAIN-2021-10",
    "CC-MAIN-2021-17",
    "CC-MAIN-2021-21",
    "CC-MAIN-2021-25",
    "CC-MAIN-2021-31",
    "CC-MAIN-2021-39",
    "CC-MAIN-2021-43",
    "CC-MAIN-2021-49",
    #
]


s3_location = sys.argv[1]
s3_location = s3_location.rstrip("/")  # no trailing slash!

seed_table = sys.argv[2]

crawl_selector = re.compile(sys.argv[3], re.IGNORECASE)


crawls = filter(lambda c: crawl_selector.match(c), crawls)


cursor = connect(
    s3_staging_dir="{}/staging".format(s3_location), region_name="us-east-1"
).cursor()

for crawl in crawls:
    query = join_template.format(
        crawl=crawl,
        s3_location=f"{s3_location}/cc-{seed_table}",
        db="bigscience",
        seed_table=seed_table,
        tid="bs",
    )
    logging.info("Athena query: %s", query)

    cursor.execute(query)
    logging.info("Athena query ID %s: %s", cursor.query_id, cursor.result_set.state)
    logging.info(
        "       data_scanned_in_bytes: %d", cursor.result_set.data_scanned_in_bytes
    )
    logging.info(
        "       total_execution_time_in_millis: %d",
        cursor.result_set.total_execution_time_in_millis,
    )

    cursor.execute(drop_tmp_table.format(db="bigscience"))
    logging.info("Drop temporary table: %s", cursor.result_set.state)
