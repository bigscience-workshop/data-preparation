import os
import logging
from argparse import ArgumentParser

from datasets import load_from_disk
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset name.")

    args = parser.parse_args()
    return args


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(
        f"** The job is runned with the following arguments: **\n{args}\n **** "
    )

    for dataset_name in os.listdir(args.dataset_dir):
        dataset_path = os.path.join(args.dataset_dir, dataset_name)
        try:
            logging.info(f"Processing: {dataset_path}")
            ds = load_from_disk(dataset_path)
            new_ds = ds.filter(keep_failed_examples)
            logging.info(f"Here's the subset of failed downloads: {new_ds}")
        except Exception as e:
            logging.warning(f"Failed to process {dataset_path} with error '{str(e)}'")


def keep_failed_examples(example):
    if example["download_exception"] is None:
        return False
    return True


if __name__ == "__main__":
    main()
