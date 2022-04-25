import logging
from argparse import ArgumentParser
from pathlib import Path

import datasets
from datasets import config, load_from_disk
from datasets.utils.logging import set_verbosity_info

from .download_warc import download_warcs

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset name.")
    parser.add_argument("--num-proc", type=int, required=True, help="Dataset name.")
    parser.add_argument("--save-path", type=str, help="Where to save the datasets.")
    parser.add_argument("--use-datasets-caching", action="store_true")

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

    if not args.use_datasets_caching:
        datasets.set_caching_enabled(False)
    else:
        logger.info(
            f"the datasets results will be cached at {config.HF_DATASETS_CACHE}."
        )

    ds = load_from_disk(args.dataset_path)

    if args.save_path:
        save_path = Path(args.save_path)
    else:
        save_path = Path(args.dataset_path)

    download_warcs(ds, save_path, num_proc=args.num_proc)


if __name__ == "__main__":
    main()
