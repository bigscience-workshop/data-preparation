import ast
import json
import logging
from argparse import ArgumentParser
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import datasets
from datasets import load_dataset, Dataset, utils, concatenate_datasets, Features
from datasets.utils.logging import set_verbosity_info
from numpy.random import default_rng, SeedSequence

set_verbosity_info()
logger = logging.getLogger(__name__)


def sanitize(datasets_with_ratios: List[Tuple[str, str]]):
    results = []
    for dataset_with_ratio in datasets_with_ratios:
        assert len(dataset_with_ratio) == 2
        result = (Path(dataset_with_ratio[0]), float(dataset_with_ratio[1]))
        assert result[1] <= 1 and result[1] >= 0, "Ratio should be between 0 and 1"
        results.append(results)
    return results


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--datasets-ratios-path", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, required=True)
    parser.add_argument(
        "--shard_max_size",
        type=int,
        default=1_000_000_000,
        help="max shard size, default 10GB",
    )
    parser.add_argument("--save-path", type=Path, required=True)

    args = parser.parse_args()

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
        "HtmlPreprocessor_error_comment": {
            "dtype": "string",
            "id": None,
            "_type": "Value",
        },
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


def load_dataset(dataset_path: Path, ratio: int, seed) -> Optional[Dataset]:
    if ratio <= 0:
        return None

    features = get_features()
    ds = load_dataset(
        str((dataset_path / "text__html").absolute()),
        data_files="**.jsonl.gz",
        features=features,
        split="train",
    )

    # collapse all meta data in "meta" column
    ds = collapse_meta(ds, num_proc=1)

    # randomly sample ratio * len(ds)
    rng = default_rng(seed)
    indices = rng.choice(
        len(ds), size=int(len(ds) * ratio), replace=False, shuffle=False
    )

    return ds.select(indices)


def collapse_meta_(batch):
    """{"text": str, "meta": str}"""
    # TODO: check that
    columns_not_in_meta = ["text", "html_str"]
    columns_to_collapse = [
        name for name in batch.keys() if name not in columns_not_in_meta
    ]

    new_batch = {
        "text": batch["text"],
        "meta": [
            str({key: value for key, value in zip(columns_to_collapse, row)})
            for row in zip(*[batch[name] for name in columns_to_collapse])
        ],
    }
    return new_batch


def collapse_meta(ds: Dataset, num_proc):
    """{"text": str, "meta": str}"""
    columns_to_keep = ["text"]
    column_names_to_remove = [
        name for name in ds.column_names if name not in columns_to_keep
    ]
    return ds.map(
        collapse_meta_,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names_to_remove,
    )


def compute_number_of_shards(ds, max_size=10_000_000_000):
    if ds._indices is not None:
        ds_nbytes = ds.data.nbytes * len(ds._indices) / len(ds.data)
    else:
        ds_nbytes = ds.data.nbytes
    logger.info(f"Estimated dataset size: {ds_nbytes} bytes")
    logger.info(f"Max shard size: {max_size} bytes")
    return ceil(ds_nbytes / max_size)


def save_dataset(ds: Dataset, save_path: Path, shard_max_size: int):
    logger.info(f"Save dataset at {save_path}")
    num_shards = compute_number_of_shards(ds, max_size=shard_max_size)

    shards = [
        ds.shard(num_shards=num_shards, index=shard_id)
        for shard_id in range(num_shards)
    ]
    for shard_id, shard in enumerate(shards):
        logger.info(f"Saving shard: {shard_id} / {len(shards)}")
        shard_path = save_path / f"shard--{shard_id}--{len(shards)}"
        shard_path_tmp = shard_path.rename(f"{shard_path.name}.tmp")
        ds.save_to_disk(str(shard_path_tmp.absolute()))
        shard_path_tmp.rename(shard_path)

    logger.info(
        f"Now you need to manually update update states.json to concatenate all data files"
    )


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

    # Load datasets_paths with ratios
    with open(args.datasets_ratios_path, "f") as fi:
        datasets_with_ratios = sanitize(json.load(fi))

    seed = SeedSequence(42)

    # Load dataset
    with Pool(args.load_num_proc) as pool:
        datasets = pool.imap(
            load_dataset,
            [
                (ds_name, ratio, child_seed)
                for (ds_name, ratio), child_seed in zip(
                    datasets_with_ratios, seed.spawn(len(datasets_with_ratios))
                )
            ],
        )
        datasets = [
            ds
            for ds in utils.tqdm(
                datasets,
                total=len(datasets_with_ratios),
                unit="ba",
                disable=bool(utils.logging.get_verbosity() == utils.logging.NOTSET),
                desc="Loading dataset",
            )
            if ds is not None
        ]

    ds = concatenate_datasets(datasets)

    ds = ds.shuffle(seed)

    # Save dataset locally
    save_dataset(ds, args.save_path, args.shard_max_size)


if __name__ == "__main__":
    main()

# python process_for_concatenation.py --datasets-with-ratios $six_ALL_CCFRSCRATCH/pseudo_crawl/tokenization_dataset/datasets_with_ratios.json --num-proc 40 --save-path $six_ALL_CCFRSCRATCH/
