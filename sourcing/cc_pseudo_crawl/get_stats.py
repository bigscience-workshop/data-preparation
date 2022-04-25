import logging
from argparse import ArgumentParser
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path.")
    args = parser.parse_args()

    args.dataset_path = Path(args.dataset_path)
    return args


def load_others(dataset_path: Path):
    others_path = dataset_path / "others"
    shards = [
        load_from_disk(str(shard_path.absolute()))
        for shard_path in sorted(others_path.iterdir())
    ]
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

    others = load_others(args.dataset_path)
    features = others.features.copy()
    features.pop("compressed_warc")
    text_htmls = load_dataset(
        str((args.dataset_path / "text__html").absolute()),
        data_files="**.jsonl.gz",
        features=features,
        split="train",
    )

    logger.info(f"Text/html: {len(text_htmls)}")
    logger.info(f"Others: {len(others)}")


if __name__ == "__main__":
    main()
