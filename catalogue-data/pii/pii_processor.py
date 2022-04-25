import argparse
from functools import partial
from pathlib import Path
import logging
import random

from datasets.utils.logging import set_verbosity_info
from datasets import load_dataset, load_from_disk

from bigscience_pii_detect_redact import run_pii_batch

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--load-from-disk', action="store_true")
    parser.add_argument('--save-to-json', action="store_true")
    parser.add_argument('--save-check-to-json', action="store_true")
    parser.add_argument('--check-only-modified', action="store_true")
    parser.add_argument('--dataset-path', type=Path)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--save-path', type=Path)
    parser.add_argument('--save-check-path', type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--check-sampling-size", type=int)
    parser.add_argument("--save-batch-size", type=int)
    args = parser.parse_args()
    return args

def get_check_ds(ds, args):
    if args.check_only_modified:
        ds_checks = ds.filter(
            lambda exs: exs["modified"],
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc
        )
    else:
        ds_checks = ds
    idx_samples = random.sample(range(len(ds_checks)), min(len(ds_checks), args.check_sampling_size))
    ds_checks = ds_checks.select(idx_samples)

    return ds_checks


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    logger.info(f" ===== Loading {args.dataset_path} =====")
    if args.load_from_disk:
        ds = load_from_disk(str(args.dataset_path))
    else:
        ds = load_dataset(str(args.dataset_path), data_files=[f"*{args.dataset_name}"], split="train")
    
    lang = str(args.dataset_path).split("/")[-1].replace("indic-", "").replace("lm_", "")[:2]
    logger.info(f"ds info: {ds}")
    logger.info(f" ===== Applying PII =====")
    ds = ds.map(
        partial(run_pii_batch, lang=lang),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )

    ds_final = ds.remove_columns([
        "regex_metadata",
        "old_text",
        "modified"
    ])
    logger.info(f"ds_final info: {ds_final}")
    
    logger.info(f" ===== Saving Final dataset =====")
    logger.info(f"Saving to final dataset at {args.save_path}.")
    tmp_save_path = Path(args.save_path.parent, f"tmp-{args.save_path.name}")
    if len(ds_final) == 0:
        logger.info("Dataset was empty. Not saving anything.")
    else:
        if args.save_to_json:
            ds_final.to_json(
                tmp_save_path,
                num_proc=args.num_proc,
                batch_size=args.save_batch_size
            )
        else:
            ds_final.save_to_disk(tmp_save_path)
        tmp_save_path.rename(args.save_path)
        logger.info(f" ===== Final dataset saved successfully =====")

    ds_checks = get_check_ds(ds, args)

    logger.info(f" ===== Saving check dataset =====")
    logger.info(f"Saving check dataset at {args.save_check_path}.")
    tmp_save_path = Path(args.save_check_path.parent, f"tmp-{args.save_check_path.name}")
    if len(ds_checks) == 0:
        logger.info("Dataset was empty. Not saving anything.")
    else:
        if args.save_check_to_json:
            ds_checks.to_json(
                tmp_save_path,
                num_proc=args.num_proc,
                batch_size=args.save_batch_size
            )
        else:
            ds_checks.save_to_disk(tmp_save_path)
        tmp_save_path.rename(args.save_check_path)
        logger.info(f" ===== Check dataset saved successfully =====")