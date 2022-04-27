# Deduplication

Python script for near deduplication. Everything is configurable with a yaml file. All OSCAR 21.09 deduplication configurations can be found under `ac_dc/deduplicate/conf`.

If you are working with small datasets (datasets that fit in memory), this is a good way to start. Or you can try the [C++ version](https://github.com/ChenghaoMou/simhash) instead.

### Self near-deduplication
Cluster near duplicates in one dataset.

e.g
```bash
python self_deduplicate.py conf/self_deduplicate_gl.yaml
```

It outputs the following files
```text
#matches.tsv (diff means simhash bit difference)
id1     id2     diff
5676323 8347653 4
20899   10053778        4

#clusters.tsv (-1 means no cluster)
id      hash    cluster
0       2471784231621897202     -1
1       16314724221857303546    -1
4       10666012509495373957    -1

# text.csv (line based documents, each document has replaced the newline character with space)
document1
document2

# ids.csv (line based ids)
id0
id1
```

and an example file.

#### Configuration
```yaml
tokenization: "character"   # character, punctuation or space
window_size: 4              # size of the token window
hamming_distance: 3         # similarity threshold out of 64 bits
num_blocks: 5               # must be larger than the hamming_distance
ignore_punctuation: true    # ignore punctuation when hashing, cannot be true when punctuation is used for tokenization
lowercase: true             # lowercase the text when hashing
text_column: "text"         # column name for the text to be hashed
index_column: "id"          # column name for the index
num_proc: 80                # number of processes to run when hashing
load_dataset:
  path: null
  name: null
  split: null
  use_auth_token: false
load_from_disk:
  path: "/home/chenghao/data_tooling/ac_dc/deduplicate/cache/filtered_tot"
  gcs: null
cache: "outputs/en_cache"
output: "outputs/en"
```

To configure hamming distance and number of blocks, it is best to give this paper [[1]](#1) a read. The intuition behind it is that:
1. The smaller the hamming distance is, the faster the process will be and the stricter the results will be
2. Given the same hamming distance, the larger the number of blocks, the more passes it will need to finish, but each pass takes less time to process.
3. The time complexity is roughly $O({b \choose k} \times (n\log(n) + n \times s))$ where $b$ is the number of blocks, $k$ is the hamming distance, $n$ is the number of hashes, and $s$ is the average size of a bucket – number of hashes sharing the same prefix.

### Suffix array substring deduplication

Reference and set-up: [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)

Commands:
```bash
#!/bin/bash

lan="en"
sa_output_file="/home/chenghao/data_tooling/ac_dc/deduplicate/outputs/${lan}/sa.txt"
final_output_file="/home/chenghao/data_tooling/ac_dc/deduplicate/outputs/${lan}/substring_bytes.tsv"
text_file="/home/chenghao/data_tooling/ac_dc/deduplicate/outputs/${lan}/text.csv"
id_file="/home/chenghao/data_tooling/ac_dc/deduplicate/outputs/${lan}/ids.csv"

rm -rf ${sa_output_file} ${final_output_file}
rm -rf /tmp/dups_text.csv_*
python scripts/make_suffix_array.py ${text_file}
cargo run selfsimilar_parallel ${text_file}
cargo run collect_similar text.csv >> ${sa_output_file} # use filename only
python scripts/restore.py ${text_file} ${sa_output_file} ${id_file} >> ${final_output_file}
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp /home/chenghao/data_tooling/ac_dc/deduplicate/outputs/${lan}/{substring_bytes.tsv,sa.txt,ids.csv,text.csv} gs://commoncrawl2022/outputs/${lan}/
```

**Note**: All offsets here are byte offsets instead of string offsets.

```text
# sa.txt (byte offsets of duplicates from text.csv)
0 288
658 765
982 1246
1298 1586

# substring_bytes.tsv (restored line-based byte offsets)
# e.g. document 0 has two byte sequence duplicates: doc0[0:288] and doc0[658:765]
id      x       y
0       0       288
0       658     765
1       150     414
1       466     754
```

## Analysis

False positives in long documents: Because SimHash is essentially a BOW algorithm that long documents are more likely ended up being similar to each other, it might be a good idea to combine [[Suffix Array Substring deduplication]] when filtering long duplicates:
1. Remove short duplicates or closely-similar documents – documents with small bit difference – based on SimHash (e.g `<= 1024` or `<= 3`)
2. Remove substring duplicates based on Suffix Array

The time took for SimHash hashing and clustering is usually faster given the same data and computation resources than Suffix Array, e.g:
1. It took a few hours to hashing and clustering English data with 96 cores and 1.4TB memory, while it took more than a day just to build the suffix array;

The same substring threshold is applied both to the generation of `sa.txt` and the restoration of `substring_bytes.tsv`, meaning:
1. If a substring is shorter than $50$ bytes in `text.csv`, it will be ignored;
2. If a substring spans multiple documents with a length longer than $50$, then during document boundary restoration aka. mapping the substring back to the each documents, any document-bound sub-substring that is shorter than $50$ bytes will be further ignored;

For more details on deduplication results on different OSCAR datasets, please see [Deduplication Report](https://chenghaomou.github.io/1%20Projects/BigScience/SubProjects/Deduplication%20report)

## References
<a id="1">[1]</a> Manku, Gurmeet Singh, Arvind Jain, and Anish Das Sarma. "Detecting near-duplicates for web crawling." Proceedings of the 16th international conference on World Wide Web. 2007.
