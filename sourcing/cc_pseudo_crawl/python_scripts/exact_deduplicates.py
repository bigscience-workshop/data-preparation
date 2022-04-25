"""Taken from Teven and Leandro"""
import gzip
import os
import shutil
import time
import logging
import argparse
import datasets

from datasets import load_dataset, Features
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)

null = None
# features = {
#     "HtmlPreprocessor_error": {"dtype": "int64", "id": null, "_type": "Value"},
#     "HtmlPreprocessor_error_comment": {"dtype": "string", "id": null, "_type": "Value"},
#     "content_languages": {"dtype": "string", "id": null, "_type": "Value"},
#     "content_mime_detected": {"dtype": "string", "id": null, "_type": "Value"},
#     "depth": {"dtype": "int16", "id": null, "_type": "Value"},
#     "download_exception": {"dtype": "string", "id": null, "_type": "Value"},
#     "external_urls": [{"dtype": "string", "id": null, "_type": "Value"}],
#     "fetch_redirect": {"dtype": "string", "id": null, "_type": "Value"},
#     "fetch_status": {"dtype": "int32", "id": null, "_type": "Value"},
#     "fetch_time": {"dtype": "timestamp[ns]", "id": null, "_type": "Value"},
#     "html_error": {"dtype": "string", "id": null, "_type": "Value"},
#     "html_footer": [{"dtype": "string", "id": null, "_type": "Value"}],
#     "html_head": [{"dtype": "string", "id": null, "_type": "Value"}],
#     "html_str": {"dtype": "string", "id": null, "_type": "Value"},
#     "html_title": [{"dtype": "string", "id": null, "_type": "Value"}],
#     "metadata_html": [
#         {
#             "char_end_idx": {"dtype": "int64", "id": null, "_type": "Value"},
#             "char_start_idx": {"dtype": "int64", "id": null, "_type": "Value"},
#             "html_attrs": {
#                 "attrs": [{"dtype": "string", "id": null, "_type": "Value"}],
#                 "values": [{"dtype": "string", "id": null, "_type": "Value"}],
#             },
#             "key": {"dtype": "string", "id": null, "_type": "Value"},
#             "relative_end_pos": {"dtype": "int64", "id": null, "_type": "Value"},
#             "relative_start_pos": {"dtype": "int64", "id": null, "_type": "Value"},
#             "type": {"dtype": "string", "id": null, "_type": "Value"},
#             "value": {"dtype": "string", "id": null, "_type": "Value"},
#         }
#     ],
#     "seed_id": {"dtype": "int32", "id": null, "_type": "Value"},
#     "text": {"dtype": "string", "id": null, "_type": "Value"},
#     "url": {"dtype": "string", "id": null, "_type": "Value"},
#     "url_host_name": {"dtype": "string", "id": null, "_type": "Value"},
#     "url_host_registered_domain": {"dtype": "string", "id": null, "_type": "Value"},
#     "url_host_tld": {"dtype": "string", "id": null, "_type": "Value"},
#     "url_surtkey": {"dtype": "string", "id": null, "_type": "Value"},
#     "warc_filename": {"dtype": "string", "id": null, "_type": "Value"},
#     "warc_record_length": {"dtype": "int32", "id": null, "_type": "Value"},
#     "warc_record_offset": {"dtype": "int32", "id": null, "_type": "Value"},
# }
features = {
    "text": {"dtype": "string", "id": null, "_type": "Value"},
    "meta": {
        "content_languages": {"dtype": "string", "id": null, "_type": "Value"},
        "seed_id": {"dtype": "int64", "id": null, "_type": "Value"},
        "url": {"dtype": "string", "id": null, "_type": "Value"},
    },
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


def get_hash(example):
    """Get hash of content field."""
    return {"hash": hash(example["text"].replace(" ", ""))}


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


def preprocess(example):
    """Chain all preprocessing steps into one function to not fill cache."""
    results = dict()
    results.update(get_hash(example))
    return results


def filter(example, uniques, args):
    """Filter dataset with heuristics."""
    if not check_uniques(example, uniques):
        return False
    else:
        return True


def compress_file(file_path):
    """Compress a file with g-zip."""
    with open(file_path, "rb") as f_in:
        with gzip.open(file_path + ".gz", "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.unlink(file_path)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Load seed and upload to hub")
    parser.add_argument(
        "--seed-id",
        help="seed ID",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--save-dir", required=True, type=str, help="Where to save the datasets."
    )
    parser.add_argument(
        "--pseudo_crawl_path",
        help="path to where the pseudocrawl is located",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--gzipped",
        help="Write file directly in jsonl.gz compressed format",
        action="store_true",
    )
    parser.add_argument(
        "--save-batch-size",
        help="Batch size used for saving the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size used for the mapping and saving of the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num-proc",
        help="Number of processors used for the mapping and saving of the dataset",
        required=True,
        type=int,
    )
    args = parser.parse_args()

    # Load dataset
    t_start = time.time()
    ds = load_dataset(
        "json",
        # data_files=[f"{args.pseudo_crawl_path}/seed_id={args.seed_id}/text__html/*.jsonl.gz"],
        data_files=[
            f"{args.pseudo_crawl_path}/lm_change_lang_id_seed_id_{args.seed_id}_pseudocrawl_change_name/*.jsonl"
        ],
        features=final_features,
        split="train",
    )
    logger.info(f"Time to load dataset: {time.time()-t_start:.2f}")

    # Run preprocessing
    t_start = time.time()
    ds = ds.map(preprocess, num_proc=args.num_proc)
    logger.info(f"Time to preprocess dataset: {time.time()-t_start:.2f}")

    # Deduplicate hashes
    uniques = set(ds.unique("hash"))
    frac = len(uniques) / len(ds)
    logger.info(f"Fraction of duplicates: {1-frac:.2%}")

    # Deduplicate data and apply heuristics
    t_start = time.time()
    ds_filter = ds.filter(filter, fn_kwargs={"uniques": uniques, "args": args})
    logger.info(f"Time to filter dataset: {time.time()-t_start:.2f}")
    logger.info(f"Size of filtered dataset: {len(ds_filter)}")

    # Save data
    t_start = time.time()
    if args.gzipped:
        file_name = os.path.join(args.save_dir, f"data.jsonl.gz")
        logger.info(f"the dataset will be saved at {file_name}")
        ds_filter.to_json(
            file_name,
            num_proc=args.num_proc,
            batch_size=args.save_batch_size,
            compression="gzip",
        )
    else:
        file_name = os.path.join(args.save_dir, f"data.jsonl")
        logger.info(f"the dataset will be saved at {file_name}")
        ds_filter.to_json(
            file_name,
            num_proc=args.num_proc,
            batch_size=args.save_batch_size,
        )

    logger.info(f"Time to save dataset: {time.time()-t_start:.2f}")


if __name__ == "__main__":
    main()
