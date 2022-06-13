import argparse
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(
        f"** The job is runned with the following arguments: **\n{args}\n **** "
    )

    logger.info(f" ===== Loading {args.dataset_path} =====")
    ds = load_dataset(str(args.dataset_path))
    if isinstance(ds, DatasetDict):
        if len(ds) != 1:
            raise ValueError(
                f"There is some problems in the splits of {args.dataset_path}"
            )
        ds = ds[next(iter(ds.keys()))]

    logger.info(f"ds info: {ds}")
    logger.info(f" ===== Getting len text =====")
    ds_final = ds.map(
        lambda texts: {"len": [len(text.encode()) for text in texts]},
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        input_columns=["text"],
    )

    logger.info(f"ds_final info: {ds_final}")

    logger.info(f" ===== Saving Final dataset =====")
    logger.info(f"Saving to final dataset at {args.save_path}.")
    tmp_save_path = Path(args.save_path.parent, f"tmp-{args.save_path.name}")
    if len(ds_final) == 0:
        logger.info("Dataset was empty. Not saving anything.")
    ds_final.save_to_disk(tmp_save_path)
    tmp_save_path.rename(args.save_path)
    logger.info(f" ===== Final dataset saved successfully =====")
