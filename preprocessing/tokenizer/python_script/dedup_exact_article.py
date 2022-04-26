"""Taken from Teven and Leandro"""
import gzip
import os
import shutil
import time
import logging
import argparse

from datasets import load_from_disk
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Load seed and upload to hub")
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
    return args


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
    args = get_args()

    # Load dataset
    t_start = time.time()
    ds = load_from_disk(args.dataset_dir)
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
    ds_filter.save_to_disk(args.save_dir)

    logger.info(f"Time to save dataset: {time.time()-t_start:.2f}")


if __name__ == "__main__":
    main()
