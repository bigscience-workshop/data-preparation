import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, load_from_disk, concatenate_datasets
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset directory containing all shards",
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Where to save dataset."
    )
    parser.add_argument("--seed-id", type=int, required=True, help="Seed id.")
    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.save_path = Path(args.save_path)
    return args


def load_all_matching_shards(dataset_dir: Path, seed_id: int) -> Dataset:
    """We use seed id and check that the shards correspond to"""
    shard_paths = sorted(
        str((elt / f"seed_id={seed_id}").absolute())
        for elt in dataset_dir.iterdir()
        if (elt / f"seed_id={seed_id}").exists()
    )
    logger.info(f"All the following shards will be loaded: {shard_paths}")

    shards = []
    for shard_path in shard_paths:
        logger.info(f"Loading {shard_path}")
        shard = load_from_disk(shard_path)
        shards.append(shard)

    # # Parallel version seems to go OOM
    # with Pool(num_proc) as pool:
    #     async_results = pool.map_async(load_from_disk, shard_paths)
    #     shards = async_results.get()

    logger.info("Concatenating all shards together.")
    return concatenate_datasets(shards)


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

    ds = load_all_matching_shards(args.dataset_dir, args.seed_id)

    logger.info(f"the seed {args.seed_id} has {len(ds)} rows")

    if not all([seed_id == args.seed_id for seed_id in ds["seed_id"]]):
        logger.info("Not all rows correspond to the correct seed. We need to fix this.")
        exit(1)

    ds.save_to_disk(f"{args.save_path}.tmp")
    subprocess.run(["mv", f"{args.save_path}.tmp", args.save_path])


if __name__ == "__main__":
    main()
