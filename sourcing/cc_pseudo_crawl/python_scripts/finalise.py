import re
from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")

    args = parser.parse_args()

    assert args.dataset not in args.datasets_to_concatenate
    matches = re.match(
        r"^bigscience-catalogue-data/pseudo_crawl_(.*)(_dedup_url)?$", args.dataset
    )
    assert matches is not None
    flavor = matches.groups()[0]
    assert re.match(r"^intermediate_depth_([0-9]+)$", flavor) is not None
    args.flavor = flavor

    return args


def get_all_datasets_to_concatenate(flavor):
    def get_rank(flavor):
        if flavor == "seed":
            return 0
        else:
            # TODO: fix for extended_depth
            depth = re.match(r"^intermediate_depth_([0-9]+)$", flavor).groups()[0]
            return int(depth)

    def get_flavor(rank):
        if rank == 0:
            return "seed"
        else:
            return f"intermediate_depth_{rank}"

    current_rank = get_rank(flavor)
    assert current_rank > 1, "seed is already finished"
    seed_datasets = [
        f"bigscience-catalogue-data/pseudo_crawl_{get_flavor(rank)}"
        for rank in range(0, current_rank + 1)
    ]
    intermediate_depth_datasets = [
        f"bigscience-catalogue-data/pseudo_crawl_{get_flavor(rank)}_partial"
        for rank in range(1, current_rank + 1)
    ]

    return seed_datasets + intermediate_depth_datasets


def compute_external_ids_(ds):
    """This is done at the end of processing and we basically convert `external_urls` in `external_ids`"""
    # For each url, find the most recent row id corresponding to that url
    # All of the duplicate of a `url` are either all in that dictionary or not in that dictionary
    # This table allows me to do a double join so I can easily compute the ids.
    # We'll then go through that csv and add the ids to final dataset.
    # No duplicates guaranteed
    url_to_id_and_timestamp = {}
    # TODO: batch this
    for data in ds:
        url = data["url"]
        id_ = data["id"]
        timestamp = data["fetch_time"]
        if url in url_to_id_and_timestamp:
            old_id, old_time_stamp = url_to_id_and_timestamp[url]
            new_timestamp, new_id = max((timestamp, id_), (old_time_stamp, old_id))
            url_to_id_and_timestamp[url] = (new_id, new_timestamp)
        else:
            url_to_id_and_timestamp[url] = (id_, timestamp)

    # TODO: batch this
    for data in ds:
        # Not all urls are part of our index. We keep `external_urls` for this sake.
        data["external_ids"] = [
            url_to_id_and_timestamp[external_url][0]
            for external_url in data["external_urls"]
            if external_url in url_to_id_and_timestamp
        ]

    return ds


def assign_id(batch, indices):
    batch["id"] = indices
    return batch


def main():
    args = get_args()

    datasets = [
        load_dataset(dataset_name, use_auth_token=True, split="train")
        for dataset_name in get_all_datasets_to_concatenate(args.flavor)
    ]

    # Concatenate all the splits together
    ds = concatenate_datasets(datasets)

    # Generate id
    ds = ds.map(assign_id, batched=True, with_indices=True)

    # Generate external_ids
    ds = compute_external_ids_(ds)

    # Add as train split
    ds.push_to_hub(args.dataset, private=True)


if __name__ == "__main__":
    main()
