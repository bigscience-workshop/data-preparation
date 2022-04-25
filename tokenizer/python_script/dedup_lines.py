import json
import shutil
from collections import defaultdict
import os
import argparse
import logging

import datasets
from functools import partial
import pandas as pd
from datasets import Features, load_dataset, load_from_disk
from tqdm import tqdm
from datasets.utils.logging import set_verbosity_info
from numpy.random import SeedSequence, default_rng

"""
Cleaning text:
 - run exact deduplication
"""

set_verbosity_info()
logger = logging.getLogger(__name__)

###
# seed processing and upload functions
###


META_COLUMNS = ["meta"]

# filter text to remove certain lines (e.g. menu items, copyright notice)
def filter_lines(article, skip_set, used_lines):
    # TODO discuss the strip
    lines = [line.strip() for line in article.split("\n")]
    keep = []
    skip = []
    for line in lines:
        if line in skip_set and line in used_lines:
            skip += [line]
        elif line in skip_set:
            keep += [line]
            used_lines.add(line)
        else:
            keep += [line]
    return "\n".join(keep).strip(), "\n".join(skip).strip()


def filter_lines_by_batch(texts, skip_set, used_lines, preserve_code, metadata=None):
    if preserve_code:
        filtered_lines = [
            filter_lines(article, skip_set, used_lines)
            if "lm_code" in eval(metadata_item)["source_dataset"]
            else (article, "")
            for article, metadata_item in zip(texts, metadata)
        ]
    else:
        filtered_lines = [
            filter_lines(article, skip_set, used_lines) for article in texts
        ]
    return tuple(zip(*filtered_lines))


# do both together and return an entry
def process_batch(batch, skip_set, used_lines, args):
    if not args.with_meta_col:
        texts, _ = filter_lines_by_batch(
            batch["text"], skip_set, used_lines, preserve_code=False
        )
        return {
            "text": texts,
        }
    else:
        texts, _ = filter_lines_by_batch(
            batch["text"],
            skip_set,
            used_lines,
            preserve_code=args.preserve_code,
            metadata=batch["meta"],
        )
        return {
            "meta": batch["meta"],
            "text": texts,
        }


# looks at up to the first 10K pages for a seed and
# records lines that appear in at least 1% of the unique pages
def get_lines_to_skip(dset, n_records, pourcentage_threshold, min_repetition_threshold):
    line_counts = defaultdict(lambda: 0)
    seen_pages = defaultdict(lambda: 0)

    seed = SeedSequence(42)
    rng = default_rng(seed)
    num_elements = min(len(dset), n_records)
    indices = rng.choice(len(dset), size=num_elements, replace=False, shuffle=False)

    dset_sample = dset.select(indices)
    for page in tqdm(dset_sample):
        article = page["text"]

        seen_pages[article] += 1
        # We count the number of times we see identical lines in different documents.
        all_lines = {line.strip() for line in article.split("\n")}
        for line in all_lines:
            line_counts[line] += 1

    # TODO understand this logic, why it's not len(line_counts)
    if pourcentage_threshold is not None:
        thres_skip = max(
            min_repetition_threshold, len(seen_pages) * pourcentage_threshold
        )
    else:
        thres_skip = min_repetition_threshold
    skip_set = {line for line, ct in line_counts.items() if ct > thres_skip}
    return skip_set, seen_pages


def clean_examples(examples, skip_lines_set, used_lines, args):
    if args.with_meta_col:
        results = {"text": [], "meta": []}
    else:
        results = {"text": []}
    # Collapses meta and cleans text
    preprocessed_batch = process_batch(examples, skip_lines_set, used_lines, args)
    assert set(results.keys()) == set(preprocessed_batch.keys())

    for idx, cleaned_article in enumerate(preprocessed_batch["text"]):
        if len(cleaned_article) <= args.min_chars:
            continue
        for key in results.keys():
            results[key].append(preprocessed_batch[key][idx])

    return results


# create a private repository and push processed seed in jsonl format
TEXT_COLUMN = "text"


def filter_and_save(dset, skip_lines_set, seen_pages, args):
    repo_name = args.save_dir
    # TODO build a caching mechanism
    repo_name_tmp = f"{repo_name}.tmp"
    if not os.path.isdir(repo_name_tmp):
        os.makedirs(repo_name_tmp)

    # process
    used_lines = set()
    dset = dset.map(
        partial(
            clean_examples,
            skip_lines_set=skip_lines_set,
            used_lines=used_lines,
            args=args,
        ),
        batched=True,
        # num_proc=args.num_proc, # single proccess for used_lines
        batch_size=args.batch_size,
        remove_columns=dset.column_names,
    )
    logger.info(f"Finished cleaning")

    # write to folder
    dset.save_to_disk(repo_name_tmp)

    logger.info(f"Ended successfully, saved at {repo_name_tmp}")

    # Saving skipped lines that are considered repetitive
    with open(os.path.join(repo_name_tmp, "skipped_lines.json"), "w") as fi:
        json.dump(list(skip_lines_set), fi, indent=2)

    # Saving num of duplicated documents
    with open(os.path.join(repo_name_tmp, "duplicate_documents.json"), "w") as fi:
        json.dump([num for num in list(seen_pages.values()) if num > 1], fi, indent=2)

    # Move so that the state becomes completed
    shutil.move(repo_name_tmp, repo_name)


def text_is_not_none(batch):
    return [text is not None for text in batch["text"]]


###
# combine everything
###
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir", required=True, type=str, help="Where to save the datasets."
    )
    parser.add_argument(
        "--dataset_dir",
        help="path to where the arrow dataset is located",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size used for mapping the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num-proc",
        help="Number of processors used for the mapping of the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--min-chars",
        help="Minimum number of chars in a line",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--n-records",
        help="Number of records used to compute the repetitions",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--pourcentage-threshold",
        help="Threshold used for filter repetitions",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--min-repetition-threshold",
        help="Minimum threshold used for filter repetitions. Used when the number of available records is not enough",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--with-meta-col",
        help="If the initial dataset has a meta column",
        action="store_true",
    )
    parser.add_argument(
        "--preserve_code",
        help="Exclude code datasets from the line dedup",
        action="store_true",
    )
    args = parser.parse_args()
    # Load dataset (data first needs to be git pulled, see above)

    dset = load_from_disk(args.dataset_dir)

    # pre-remove unecessary columns, hopefully that saves qui a bit of memory usage
    columns_to_keep = [TEXT_COLUMN] + META_COLUMNS
    dset = dset.remove_columns(list(set(dset.column_names) - set(columns_to_keep)))

    # Filter None text columns
    number_of_samples_before = len(dset)
    dset = dset.filter(text_is_not_none, batched=True, num_proc=args.num_proc)
    number_of_samples_after_filtering_none = len(dset)
    logger.info(
        f"Filtered out {number_of_samples_before - number_of_samples_after_filtering_none} / {number_of_samples_before}"
    )

    skip_lines_set, seen_pages = get_lines_to_skip(
        dset,
        n_records=args.n_records,
        pourcentage_threshold=args.pourcentage_threshold,
        min_repetition_threshold=args.min_repetition_threshold,
    )

    filter_and_save(
        dset, skip_lines_set=skip_lines_set, seen_pages=seen_pages, args=args
    )
    logger.info("Finished")


if __name__ == "__main__":
    main()
