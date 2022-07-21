import argparse
import json
import logging
import os
import multiprocessing
import re
from contextlib import contextmanager
from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, Union, Optional, List

import datasets
from dotenv import load_dotenv
from numpy import log10
from numpy.random import default_rng, SeedSequence

from datasets import concatenate_datasets, load_dataset, utils, Features, Value, Dataset

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset_ratios_path",
        type=str,
        required=True,
        help="path to JSON file containing input dataset ratios. Values ares dictionary: {'dataset_path': str, 'is_catalogue': bool, 'ratio': float}",
    )
    parser.add_argument("--split", type=str, default="train", help="split name, default 'train'")
    parser.add_argument(
        "--load_num_proc", type=int, default=1, help="number of procs to use for loading datasets, default 1"
    )
    # Shard
    parser.add_argument("--shard_max_size", type=int, default=10_000_000_000, help="max shard size, default 10GB")
    # Save
    parser.add_argument("--save_path", type=str, default=".", help="path to save the dataset, default '.'")
    parser.add_argument("--save_num_proc", type=int, default=1, help="number of procs to use for saving, default 1")
    parser.add_argument("--save_batch_size", type=int, help="batch size used for saving")
    # Parse args
    args = parser.parse_args()
    # Post-process args
    args.dataset_ratios_path = Path(args.dataset_ratios_path)
    args.save_path = Path(args.save_path)
    return args


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        return getattr(datasets, features["_type"])(features["dtype"])
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


def get_features():
    features = {
        "HtmlPreprocessor_error": {"dtype": "int64", "id": None, "_type": "Value"},
        "HtmlPreprocessor_error_comment": {"dtype": "string", "id": None, "_type": "Value"},
        "content_languages": {"dtype": "string", "id": None, "_type": "Value"},
        "content_mime_detected": {"dtype": "string", "id": None, "_type": "Value"},
        "depth": {"dtype": "int16", "id": None, "_type": "Value"},
        "download_exception": {"dtype": "string", "id": None, "_type": "Value"},
        "external_urls": [{"dtype": "string", "id": None, "_type": "Value"}],
        "fetch_redirect": {"dtype": "string", "id": None, "_type": "Value"},
        "fetch_status": {"dtype": "int32", "id": None, "_type": "Value"},
        "fetch_time": {"dtype": "timestamp[ns]", "id": None, "_type": "Value"},
        "html_error": {"dtype": "string", "id": None, "_type": "Value"},
        "html_footer": [{"dtype": "string", "id": None, "_type": "Value"}],
        "html_head": [{"dtype": "string", "id": None, "_type": "Value"}],
        "html_str": {"dtype": "string", "id": None, "_type": "Value"},
        "html_title": [{"dtype": "string", "id": None, "_type": "Value"}],
        "metadata_html": [
            {
                "char_end_idx": {"dtype": "int64", "id": None, "_type": "Value"},
                "char_start_idx": {"dtype": "int64", "id": None, "_type": "Value"},
                "html_attrs": {
                    "attrs": [{"dtype": "string", "id": None, "_type": "Value"}],
                    "values": [{"dtype": "string", "id": None, "_type": "Value"}],
                },
                "key": {"dtype": "string", "id": None, "_type": "Value"},
                "relative_end_pos": {"dtype": "int64", "id": None, "_type": "Value"},
                "relative_start_pos": {"dtype": "int64", "id": None, "_type": "Value"},
                "type": {"dtype": "string", "id": None, "_type": "Value"},
                "value": {"dtype": "string", "id": None, "_type": "Value"},
            }
        ],
        "seed_id": {"dtype": "int32", "id": None, "_type": "Value"},
        "text": {"dtype": "string", "id": None, "_type": "Value"},
        "url": {"dtype": "string", "id": None, "_type": "Value"},
        "url_host_name": {"dtype": "string", "id": None, "_type": "Value"},
        "url_host_registered_domain": {"dtype": "string", "id": None, "_type": "Value"},
        "url_host_tld": {"dtype": "string", "id": None, "_type": "Value"},
        "url_surtkey": {"dtype": "string", "id": None, "_type": "Value"},
        "warc_filename": {"dtype": "string", "id": None, "_type": "Value"},
        "warc_record_length": {"dtype": "int32", "id": None, "_type": "Value"},
        "warc_record_offset": {"dtype": "int32", "id": None, "_type": "Value"},
    }
    return Features(convert_types(features))


def collapse_meta_(batch):
    """{"text": str, "meta": str}"""
    # TODO: check that
    columns_not_in_meta = ["text", "html_error", "html_footer", "html_head", "html_str", "html_title", "metadata_html"]
    columns_to_collapse = [name for name in batch.keys() if name not in columns_not_in_meta]

    number_of_rows = len(batch["text"])
    metas = [
        {
            **{name: batch[name][i] for name in columns_to_collapse},
            "source_dataset": f"pseudo-crawl--{batch['seed_id'][i]}",
        }
        for i in range(number_of_rows)
    ]

    new_batch = {"text": batch["text"], "meta": [str(meta) for meta in metas]}
    return new_batch


def collapse_meta(ds: Dataset, num_proc: int):
    """{"text": str, "meta": str}"""
    columns_to_keep = ["text"]
    column_names_to_remove = [name for name in ds.column_names if name not in columns_to_keep]
    return ds.map(collapse_meta_, batched=True, num_proc=num_proc, remove_columns=column_names_to_remove)


def process_single_catalogue_meta_(meta: Optional[Union[str, Dict]], source_dataset) -> str:
    if meta is None:
        meta = {}
    elif isinstance(meta, str):
        meta = eval(meta)
    try:
        meta["source_dataset"] = source_dataset
    except:
        raise ValueError(f"Got {meta} of type {type(meta)}. Expected an dictionary. This is from {source_dataset}")
    return str(meta)


def process_catalogue_meta(batch, source_dataset=None, columns_not_in_meta_or_text=None):
    num_elts = len(batch[next(iter(batch.keys()))])
    default_meta = process_single_catalogue_meta_(None, source_dataset)

    # If other columns exist we put them into meta
    if columns_not_in_meta_or_text:
        if "meta" not in batch:
            batch["meta"] = [{} for _ in range(num_elts)]
        batch["meta"] = [
            {
                **(batch["meta"][index]),
                **{column_name: batch[column_name][index] for column_name in columns_not_in_meta_or_text}
            }
            for index in range(num_elts)
        ]

    if "meta" in batch:
        batch["meta"] = [process_single_catalogue_meta_(meta, source_dataset) for meta in batch["meta"]]
    else:
        batch["meta"] = [default_meta for _ in range(num_elts)]

    return {"text": batch["text"], "meta": batch["meta"]}

def load_single_dataset(args):
    try:
        ds_ratio, split, seed, num_proc = args
        ds_name = ds_ratio["dataset_path"]
        ratio = ds_ratio["ratio"]
        is_catalogue = ds_ratio["is_catalogue"]
        # Load
        if is_catalogue:
            ds = load_dataset(ds_name, use_auth_token=True, ignore_verifications=True)
        else:
            # We assume it comes from pseudo crawl.
            # Pseudo crawl needs to be downloaded locally beforehand.
            features = get_features()
            dataset_path = Path(ds_name)
            ds = load_dataset(
                str((dataset_path / "text__html").absolute()), data_files="**.jsonl.gz", features=features
            )
        # Split
        if split not in ds:
            logger.info(f"No split named {split} in dataset {ds_name}")
            return
        ds = ds[split]

        # Sample dataset
        if ratio < 1:
            num_samples = int(len(ds) * ratio)
            if num_samples == 0:
                return None
            rng = default_rng(seed)
            indices = rng.choice(len(ds), size=num_samples, replace=False, shuffle=False)
            ds = ds.select(indices)

        # Process meta: add source_dataset and cast dict to str
        if is_catalogue:
            columns_not_in_meta_or_text = [column_name for column_name in ds.column_names if column_name not in ["text", "meta"]]
            source_dataset = re.match(r".*bigscience-catalogue-lm-data/(lm_([^/])*)(/data)?", ds_name).group(1)
            ds = ds.map(
                partial(process_catalogue_meta, source_dataset=source_dataset, columns_not_in_meta_or_text=columns_not_in_meta_or_text),
                batched=True,
                num_proc=num_proc,
                desc=f"Processing {ds_name}",
                remove_columns=columns_not_in_meta_or_text
            )
        else:
            # collapse all meta data in "meta" column
            ds = collapse_meta(ds, num_proc=num_proc)

        return ds
    except BaseException as err:
        logger.error(f"Error while loading dataset {ds_name}")
        raise err


def compute_number_of_shards(ds, max_size=10_000_000_000):
    ds_nbytes = get_size(ds)
    logger.info(f"Estimated dataset size: {ds_nbytes} bytes")
    logger.info(f"Max shard size: {max_size} bytes")
    number_shards = ceil(ds_nbytes / max_size)
    return number_shards if number_shards < len(ds) else len(ds)

def get_shard(shard_id: int, number_shards: int, ds: Dataset) -> Dataset:
    logger.info(f"Shard {shard_id}/{number_shards}")
    shard = ds.shard(num_shards=number_shards, index=shard_id, contiguous=True)
    return shard

def shard_dataset(ds, num_proc, max_size=10_000_000_000):
    number_shards = compute_number_of_shards(ds, max_size=max_size)
    if number_shards <= 1:
        return [ds]
    logger.info(f"Shard dataset in {number_shards} shards")
    shards = []
    for shard_id in range(number_shards):
        shard = get_shard(shard_id=shard_id, number_shards=number_shards, ds=ds)
        shards.append(shard)

    # # Parallel version
    # with multiprocessing.Pool(min(number_shards, num_proc)) as pool:
    #     shards = [
    #         ds
    #         for ds in utils.tqdm(
    #             pool.imap(
    #                 partial(get_shard, ds=ds, number_shards=number_shards),
    #                 range(number_shards),
    #             ),
    #             total=number_shards,
    #             unit="ba",
    #             disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
    #             desc="Sharding dataset",
    #         )
    #         if ds is not None
    #     ]

    return shards


def save_shards(shards, path=Path("."), num_proc=1, batch_size=None):
    path.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards)
    # for i, shard in enumerate(shards):
    #     save_dataset(shard, path=path, shard_id=i, num_shards=num_shards, num_proc=num_proc, batch_size=batch_size)

    # Parallel version
    with multiprocessing.Pool(min(num_shards, num_proc)) as pool:
        pool.starmap(
            save_dataset,
            [
                (shard, path, shard_id, num_shards, 1, batch_size)
                for shard_id, shard in enumerate(shards)
            ]
        )


def save_dataset(shard: Dataset, path=Path("."), shard_id=0, num_shards=1, num_proc=1, batch_size=None):
    width = int(log10(num_shards)) + 1
    save_path = path / f"shard-{shard_id:0>{width}}-of-{num_shards:0>{width}}.jsonl.gz"
    if save_path.exists():
        logger.info(f"Shard was already saved: {save_path}")
        return
    with tmp_path(save_path) as tmp_save_path:
        shard.to_json(
            tmp_save_path,
            num_proc=num_proc,
            batch_size=batch_size,
        )


@contextmanager
def tmp_path(path):
    try:
        tmp_path = path.with_name(f"tmp-{path.name}")
        yield tmp_path
    except:
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.rename(path)

def get_size(ds: Dataset) -> int:
    if ds._indices is not None:
        return ds.data.nbytes * len(ds._indices) / len(ds.data)
    else:
        return ds.data.nbytes

def load_datasets(dset_ratios: List, num_proc: int, split: str, seed: SeedSequence) -> List[Dataset]:
    logger.info("Start load_datasets")
    # dsets = [
    #     ds
    #     for ds in utils.tqdm(
    #         [
    #             load_single_dataset((dset_ratio, split, child_seed, num_proc))
    #             for dset_ratio, child_seed in zip(dset_ratios, seed.spawn(len(dset_ratios)))
    #         ],
    #         total=len(dset_ratios),
    #         unit="ba",
    #         disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
    #         desc="Loading dataset",
    #     )
    #     if ds is not None
    # ]

    # Parallel version
    with multiprocessing.Pool(num_proc) as pool:
        dsets = [
            ds
            for ds in utils.tqdm(
                pool.imap(
                    load_single_dataset,
                    [
                        (dset_ratio, split, child_seed, 1)
                        for dset_ratio, child_seed in zip(dset_ratios, seed.spawn(len(dset_ratios)))
                    ],
                ),
                total=len(dset_ratios),
                unit="ba",
                disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
                desc="Loading dataset",
            )
            if ds is not None
        ]
    return dsets


def main():
    args = parse_args()

    # Init
    # Env variables
    if Path(".env").exists:
        load_dotenv()
    # Random generator
    seed = SeedSequence(42)
    # Read dataset ratios
    with args.dataset_ratios_path.open() as f:
        dset_ratios = json.load(f)
    # Load datasets
    dsets = load_datasets(dset_ratios, args.load_num_proc, args.split, seed)

    if not dsets:
        logger.info(f"No datasets to be aggregated")
        return
    # Concatenate datasets
    logger.info("Start concatenate_datasets")
    dset = concatenate_datasets(dsets, split=args.split)
    del dsets
    logger.info(f"Estimated size: {get_size(dset)} bytes")
    # Shuffle
    logger.info("Start shuffle dataset")
    dset = dset.shuffle(seed=seed)
    # Shard
    logger.info("Start shard_dataset")
    shards = shard_dataset(dset, num_proc=args.load_num_proc, max_size=args.shard_max_size)
    # Save
    logger.info("Start: save dataset")
    save_shards(shards, path=args.save_path, num_proc=args.save_num_proc, batch_size=args.save_batch_size)


if __name__ == "__main__":
    main()
