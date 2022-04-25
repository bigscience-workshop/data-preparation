import os
import logging
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import datasets
import lxml.html
from datasets import config, load_from_disk
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)

# For `soup.decode_content` that can hit the limit
sys.setrecursionlimit(10000)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="path to the parquet dataset folder",
    )
    parser.add_argument("--save-path", type=str, help="Where to save the datasets.")
    parser.add_argument("--use-datasets-caching", action="store_true")
    parser.add_argument(
        "--num-proc", type=int, default=1, help="Number of procs use for preprocessing."
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Optional argument to select a subset (used for debugging purposes). Example `10`.",
    )
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

    if os.path.isdir(args.save_path):
        logger.info(f"Seed id {args.save_path.split('/')[-1]} already processed")
        return

    if not args.use_datasets_caching:
        datasets.set_caching_enabled(False)
    else:
        logger.info(
            f"the datasets results will be cached at {config.HF_DATASETS_CACHE}."
        )

    ds = load_from_disk(args.dataset_path)
    logger.info(f"the dataset is {ds}")

    if args.num_examples:
        ds = ds.select([i for i in range(args.num_examples)])

    def detect_lang(example):
        if example["text"] is None or len(example["text"]) == 0:
            example["html_lang_attr"] = None
        else:
            root = lxml.html.fromstring(example["html_str"])
            root_lang = root.attrib.get("lang")
            example["html_lang_attr"] = root_lang
        return example

    ds = ds.map(
        detect_lang,
        batched=False,
        num_proc=args.num_proc,
    )

    if args.save_path:
        save_path = Path(args.save_path)
    else:
        save_path = Path(args.dataset_path)

    logger.info(
        f"Lang attribute detected for {len([e for e in ds['train']['html_lang_attr'] if e is not None])} rows out of {len(ds['train'])} rows."
    )

    save_path_tmp = f"{str(save_path.absolute())}.tmp"
    logger.info(f"Saving the dataset at {save_path_tmp}")
    ds.save_to_disk(save_path_tmp)
    logger.info(f"Moving the saved dataset to {str(save_path.absolute())}")
    subprocess.run(["mv", save_path_tmp, str(save_path.absolute())])


if __name__ == "__main__":
    main()
