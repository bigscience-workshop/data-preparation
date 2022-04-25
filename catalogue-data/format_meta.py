import argparse
import json
import logging
from pathlib import Path

from numpy.random import SeedSequence

from .aggregate_datasets import load_single_dataset

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset_ratios_path",
        type=Path,
        required=True,
        help="path to JSON file containing input dataset ratios. Values ares dictionary: {'dataset_path': str, 'is_catalogue': bool, 'ratio': float}",
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        required=True,
    )
    parser.add_argument("--split", type=str, default="train", help="split name, default 'train'")
    parser.add_argument(
        "--num_proc", type=int, default=1, help="number of procs to use for loading datasets, default 1"
    )
    # Parse args
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    with open(args.dataset_ratios_path, "r") as fi:
        ratios = json.load(fi)
    seed = SeedSequence(42)
    seeds = seed.spawn(len(ratios))

    ds_ratio = ratios[args.dataset_index]
    seed = seeds[args.dataset_index]
    load_single_dataset((ds_ratio, args.split, seed, args.num_proc))

if __name__ == "__main__":
    main()