import os
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
import sys

from datasets import load_from_disk

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
    parser.add_argument("--save-dir", type=str, help="Where to save the datasets.")
    parser.add_argument("--num-shards", type=int, help="Total number of shards.")
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

    ds = load_from_disk(args.dataset_path)

    dataset_path = Path(args.dataset_path)

    for shard_id in range(args.num_shards):
        file_name_init = dataset_path.name
        dataset_name, shard_id_init, num_shards_init = file_name_init.split("--")

        shard_id_new = int(shard_id_init) * args.num_shards + shard_id
        total_num_shard = int(num_shards_init) * args.num_shards
        shard_name = f"{dataset_name}--{shard_id_new}--{total_num_shard}"
        save_path = Path(os.path.join(args.save_dir, shard_name))
        sub_ds = ds.shard(num_shards=args.num_shards, index=shard_id)

        save_path_tmp = f"{str(save_path.absolute())}.tmp"
        logger.info(f"Saving the dataset at {save_path_tmp}")
        sub_ds.save_to_disk(save_path_tmp)
        logger.info(f"Moving the saved dataset to {str(save_path.absolute())}")
        subprocess.run(["mv", save_path_tmp, str(save_path.absolute())])


if __name__ == "__main__":
    main()
