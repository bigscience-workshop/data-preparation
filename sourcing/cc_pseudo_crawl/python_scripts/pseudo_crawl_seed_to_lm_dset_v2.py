import json
import shutil
from collections import defaultdict
import os
import argparse
import logging

import datasets
from functools import partial
import pandas as pd
from datasets import Features, load_dataset
from tqdm import tqdm
from datasets.utils.logging import set_verbosity_info
from numpy.random import SeedSequence, default_rng

"""
Cleaning text:
 - run exact deduplication
"""

set_verbosity_info()
logger = logging.getLogger(__name__)

null = None
features = {
    "HtmlPreprocessor_error": {"dtype": "int64", "id": null, "_type": "Value"},
    "HtmlPreprocessor_error_comment": {"dtype": "string", "id": null, "_type": "Value"},
    "content_languages": {"dtype": "string", "id": null, "_type": "Value"},
    "content_mime_detected": {"dtype": "string", "id": null, "_type": "Value"},
    "depth": {"dtype": "int16", "id": null, "_type": "Value"},
    "download_exception": {"dtype": "string", "id": null, "_type": "Value"},
    "external_urls": [{"dtype": "string", "id": null, "_type": "Value"}],
    "fetch_redirect": {"dtype": "string", "id": null, "_type": "Value"},
    "fetch_status": {"dtype": "int32", "id": null, "_type": "Value"},
    "fetch_time": {"dtype": "timestamp[ns]", "id": null, "_type": "Value"},
    "html_error": {"dtype": "string", "id": null, "_type": "Value"},
    "html_footer": [{"dtype": "string", "id": null, "_type": "Value"}],
    "html_head": [{"dtype": "string", "id": null, "_type": "Value"}],
    "html_str": {"dtype": "string", "id": null, "_type": "Value"},
    "html_title": [{"dtype": "string", "id": null, "_type": "Value"}],
    "metadata_html": [
        {
            "char_end_idx": {"dtype": "int64", "id": null, "_type": "Value"},
            "char_start_idx": {"dtype": "int64", "id": null, "_type": "Value"},
            "html_attrs": {
                "attrs": [{"dtype": "string", "id": null, "_type": "Value"}],
                "values": [{"dtype": "string", "id": null, "_type": "Value"}],
            },
            "key": {"dtype": "string", "id": null, "_type": "Value"},
            "relative_end_pos": {"dtype": "int64", "id": null, "_type": "Value"},
            "relative_start_pos": {"dtype": "int64", "id": null, "_type": "Value"},
            "type": {"dtype": "string", "id": null, "_type": "Value"},
            "value": {"dtype": "string", "id": null, "_type": "Value"},
        }
    ],
    "seed_id": {"dtype": "int32", "id": null, "_type": "Value"},
    "text": {"dtype": "string", "id": null, "_type": "Value"},
    "url": {"dtype": "string", "id": null, "_type": "Value"},
    "url_host_name": {"dtype": "string", "id": null, "_type": "Value"},
    "url_host_registered_domain": {"dtype": "string", "id": null, "_type": "Value"},
    "url_host_tld": {"dtype": "string", "id": null, "_type": "Value"},
    "url_surtkey": {"dtype": "string", "id": null, "_type": "Value"},
    "warc_filename": {"dtype": "string", "id": null, "_type": "Value"},
    "warc_record_length": {"dtype": "int32", "id": null, "_type": "Value"},
    "warc_record_offset": {"dtype": "int32", "id": null, "_type": "Value"},
}


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        return getattr(datasets, features["_type"])(features["dtype"])
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


final_features = convert_types(features)
final_features = Features(final_features)
final_features

###
# seed processing and upload functions
###


# extract just the metadata we wish to keep
META_COLUMNS = ["url", "content_languages", "seed_id"]


def get_meta_dict(batch):
    batch_size = len(batch[next(iter(batch))])
    meta = [{k: batch[k][idx] for k in META_COLUMNS} for idx in range(batch_size)]
    return meta


# filter text to remove certain lines (e.g. menu items, copyright notice)
def filter_lines(article, skip_set):
    # TODO discuss the strip
    lines = [line.strip() for line in article.split("\n")]
    keep = []
    skip = []
    for line in lines:
        if line in skip_set:
            skip += [line]
        else:
            keep += [line]
    return "\n".join(keep).strip(), "\n".join(skip).strip()


def filter_lines_by_batch(texts, skip_set):
    filtered_lines = [filter_lines(article, skip_set) for article in texts]
    return tuple(zip(*filtered_lines))


# do both together and return an entry
def process_batch(batch, skip_set):
    metas = get_meta_dict(batch)
    texts, _ = filter_lines_by_batch(batch["text"], skip_set)
    return {
        "meta": metas,
        "text": texts,
    }


# looks at up to the first 10K pages for a seed and
# records lines that appear in at least 1% of the unique pages
def get_lines_to_skip(dset, n_records, pourcentage_threshold, min_repetition_threshold):
    line_counts = defaultdict(lambda: 0)
    seen_pages = set()

    seed = SeedSequence(42)
    rng = default_rng(seed)
    num_elements = min(len(dset), n_records)
    indices = rng.choice(len(dset), size=num_elements, replace=False, shuffle=False)

    dset_sample = dset.select(indices)
    for page in tqdm(dset_sample):
        article = page["text"]

        # Duplicated documents are only counted once as they'll be removed in future deduplication script.
        if article in seen_pages:
            continue

        seen_pages.add(article)
        # We count the number of times we see identical lines in different documents.
        all_lines = {line.strip() for line in article.split("\n")}
        for line in all_lines:
            line_counts[line] += 1

    # TODO understand this logic, why it's not len(line_counts)
    thres_skip = max(min_repetition_threshold, len(seen_pages) * pourcentage_threshold)
    skip_set = {line for line, ct in line_counts.items() if ct > thres_skip}
    return skip_set


def clean_examples(examples, skip_lines_set, args):
    results = {"text": [], "meta": []}
    # Collapses meta and cleans text
    preprocessed_batch = process_batch(examples, skip_lines_set)
    assert set(results.keys()) == set(preprocessed_batch.keys())

    for idx, cleaned_article in enumerate(preprocessed_batch["text"]):
        if len(cleaned_article) <= args.min_chars:
            continue
        for key in results.keys():
            results[key].append(preprocessed_batch[key][idx])

    return results


def get_folder(args):
    repo_name = (
        f"lm_{args.language_code}_seed_id_{args.seed_id}_pseudocrawl_{args.name}"
    )
    if args.save_dir is not None:
        repo_name = os.path.join(args.save_dir, repo_name)
    return repo_name


# create a private repository and push processed seed in jsonl format
TEXT_COLUMN = "text"


def make_seed_jsonl(dset, skip_lines_set, args):
    repo_name = get_folder(args)
    # TODO build a caching mechanism
    repo_name_tmp = f"{repo_name}.tmp"
    if os.path.exists(repo_name):
        return
    if not os.path.isdir(repo_name_tmp):
        os.makedirs(repo_name_tmp)

    # process
    dset = dset.map(
        partial(clean_examples, skip_lines_set=skip_lines_set, args=args),
        batched=True,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        remove_columns=dset.column_names,
    )
    logger.info(f"Finished cleaning")

    # write to file
    if args.gzipped:
        file_name = os.path.join(repo_name_tmp, f"data.jsonl.gz")
        logger.info(f"the dataset will be saved at {file_name}")
        dset.to_json(
            file_name,
            num_proc=args.num_proc,
            batch_size=args.save_batch_size,
            compression="gzip",
        )
    else:
        file_name = os.path.join(repo_name_tmp, f"data.jsonl")
        logger.info(f"the dataset will be saved at {file_name}")
        dset.to_json(
            file_name,
            num_proc=args.num_proc,
            batch_size=args.save_batch_size,
        )

    logger.info(f"Ended successfully, saved at {file_name}")

    # Saving skipped lines that are considered repetitive
    with open(os.path.join(repo_name_tmp, "skipped_lines.json"), "w") as fi:
        json.dump(list(skip_lines_set), fi, indent=2)

    # Move so that the state becomes completed
    shutil.move(repo_name_tmp, repo_name)


# TODO WIP, not used currently
def get_dataset_name_and_lang_id_from_seed_id(seed_id, seed_id_info_path):
    df = pd.read_csv(seed_id_info_path)
    sub_df = df[df["id"] == seed_id]
    if len(sub_df) != 1:
        raise ValueError("You should have only one match per seed id")
    name = sub_df.name[0]
    lang_id = sub_df.lang_id[0]
    return name, lang_id


# TODO hack to change
def get_dataset_name_and_lang_id_from_seed_id_fake(seed_id, seed_id_info_path):
    return "change_name", "change_lang_id"


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
    parser = argparse.ArgumentParser(description="Load seed and upload to hub")
    parser.add_argument(
        "-sid",
        "--seed-id",
        help="seed ID",
        required=True,
        type=int,
    )
    # WIP to use with get_dataset_name_and_lang_id_from_seed_id
    # parser.add_argument(
    #     "--seed-id-info-path",
    #     help="The path to a csv containing the seed id and the corresponding lang-id and name",
    #     # required=True,
    #     type=str,
    # )
    parser.add_argument(
        "--save-dir", required=True, type=str, help="Where to save the datasets."
    )
    parser.add_argument(
        "-pc_path",
        "--pseudo_crawl_path",
        help="path to where the pseudocrawl is located",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-gz",
        "--gzipped",
        help="Write file directly in jsonl.gz compressed format",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size used for mapping the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--save-batch-size",
        help="Batch size used for saving the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num-proc",
        help="Number of processors used for the mapping and saving of the dataset",
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
        required=True,
        type=float,
    )
    parser.add_argument(
        "--min-repetition-threshold",
        help="Minimum threshold used for filter repetitions. Used when the number of available records is not enough",
        required=True,
        type=int,
    )
    args = parser.parse_args()
    # Load dataset (data first needs to be git pulled, see above)
    dset = load_dataset(
        "json",
        data_files=[
            f"{args.pseudo_crawl_path}/seed_id={args.seed_id}/text__html/*.jsonl.gz"
        ],
        features=final_features,
        split="train",
    )

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

    args.name, args.language_code = get_dataset_name_and_lang_id_from_seed_id_fake(
        args.seed_id, None
    )  # replace None with args.seed_id_info_path
    skip_lines_set = get_lines_to_skip(
        dset,
        n_records=args.n_records,
        pourcentage_threshold=args.pourcentage_threshold,
        min_repetition_threshold=args.min_repetition_threshold,
    )
    make_seed_jsonl(dset, skip_lines_set=skip_lines_set, args=args)


if __name__ == "__main__":
    main()
