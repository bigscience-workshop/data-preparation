"""
Deduplicating using `datasets` is much harder, we but we forgot to generate an id when building an index, so we're screwed.
"""
import logging
import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

from datasets import Dataset, load_from_disk, concatenate_datasets
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path.")
    parser.add_argument(
        "--save-prefix-path", type=str, required=True, help="Where to save the dataset."
    )
    parser.add_argument(
        "--num-proc", type=int, default=1, help="Number of procs use for preprocessing."
    )
    args = parser.parse_args()

    args.dataset_path = Path(args.dataset_path)
    args.save_prefix_path = Path(args.save_prefix_path)
    return args


def obtain_entire_dataset(dataset_dir: Path, num_proc: int) -> Dataset:
    shard_paths = sorted(str(elt.absolute()) for elt in dataset_dir.iterdir())
    logger.info(f"All the following shards will be loaded: {shard_paths}")

    shards = []
    for shard_path in shard_paths:
        logger.info(f"Loading {shard_path}")
        shard = load_from_disk(shard_path)
        shards.append(shard)

    # # Parallel version seem
    # with Pool(num_proc) as pool:
    #     async_results = pool.map_async(load_from_disk, shard_paths)
    #     shards = async_results.get()

    logger.info("Concatenating all shards together.")
    return concatenate_datasets(shards)


def select_seed_id(ds, seed_id, index_and_seed_id_per_row):
    index_to_keep = [
        index for index, seed_id_ in index_and_seed_id_per_row if seed_id_ == seed_id
    ]
    return ds.select(index_to_keep)


def shard_by_seed_id(ds: Dataset, num_proc: int) -> Dict[int, Dataset]:
    seed_ids = sorted(set(ds["seed_id"]))
    result = {}
    logger.info(f"Total number of seeds: {len(seed_ids)}")

    # Use filter
    for seed_id in seed_ids:
        logger.info(f"Done seed id: {seed_id}")
        shard = ds.filter(
            lambda seed_ids_: [seed_id == seed_id_ for seed_id_ in seed_ids_],
            input_columns="seed_id",
            batched=True,
            num_proc=num_proc,
        )
        assert all([seed_id_ == seed_id for seed_id_ in shard["seed_id"]])
        result[seed_id] = shard
    return result


def filter_func(seed_ids):
    # 508: https://zh.wikipedia.org/
    # 577: https://fr.wikipedia.org/
    # 523: https://github.com/
    # 529: https://sites.google.com/
    bad_seeds = [508, 523, 529, 577]
    return [seed_id not in bad_seeds for seed_id in seed_ids]


def run_on_shard(args):
    ds = load_from_disk(args.dataset_path)

    # Filter some generic things
    logger.info("Filtering bad seeds")
    ds = ds.filter(
        filter_func, input_columns="seed_id", batched=True, num_proc=args.num_proc
    )

    # Split dataset according to seed_id
    logger.info("Sharding by seed id")
    shards = shard_by_seed_id(ds, args.num_proc)

    # Save shard per seed
    logger.info("Saving shards")
    for seed_id, shard in shards.items():
        save_path = str((args.save_prefix_path / f"seed_id={seed_id}").absolute())
        logger.info(f"Saving seed id shard: {seed_id} at {save_path}")
        shard.save_to_disk(f"{save_path}.tmp")
        subprocess.run(["mv", f"{save_path}.tmp", save_path])


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

    run_on_shard(args)


if __name__ == "__main__":
    main()
