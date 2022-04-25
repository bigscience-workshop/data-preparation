"""
Generate list of urls to query for next depth. We then need to use Athena to make a fancy query.
"""
import csv
import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")

    args = parser.parse_args()

    matches = re.match(
        r"^bigscience-catalogue-data/pseudo_crawl_(?:(.*)_partial|(seed))(_dedup_url)?$",
        args.dataset,
    )
    assert matches is not None
    flavors = [elt for elt in matches.groups() if elt is not None]
    assert len(flavors) == 1 or (len(flavors) == 2 and flavors[1] == "_dedup_url")
    flavor = flavors[0]
    assert (
        flavor == "seed"
        or re.match(r"^intermediate_depth_([0-9]+)$", flavor) is not None
    )
    args.cc_index_folder = Path(args.cc_index_folder) / f"cc-{flavor}"
    args.flavor = flavor

    return args


def get_depth(flavor):
    if flavor == "seed":
        return 0
    else:
        # TODO: fix for extended_depth
        empty, depth = flavor.split("intermediate_depth_")
        assert empty == ""
        return int(depth)


def get_deepest_split(dataset_dict):
    splits = sorted(dataset_dict.keys(), key=get_depth)
    return splits[-1]


def intermediate_next(url_candidates, previous_urls):
    """Query only those urls"""
    new_urls_to_query = set(url_candidates) - set(previous_urls)
    return new_urls_to_query


# def extended_next(url_candidates, previous_urls):
#     """Query new domains"""
#     def get_domain(url):
#         parsed_url = urllib.parse.urlparse(url)
#         return parsed_url.netloc
#     new_domains_to_query = set(get_domain(url) for url in url_candidates) - set(get_domain(url) for url in previous_urls)
#     return new_domains_to_query


def main():
    args = get_args()
    csv_output_dir = Path(__file__).parent / "temp"
    subprocess.run(["mkdir", "-p", str(csv_output_dir.absolute())])

    # Load previous depth dataset
    previous_ds = load_dataset(args.dataset, use_auth_token=True, split="train")

    previous_depth = max(previous_ds["depth"])
    url_candidates = {
        url for external_urls in previous_ds["external_urls"] for url in external_urls
    }
    previous_urls = set(previous_ds["url"])

    intermediate_depth_urls = intermediate_next(url_candidates, previous_urls)
    # extended_depth_domains = extended_next(url_candidates, previous_urls)

    new_depth = previous_depth + 1

    with open(csv_output_dir / f"intermediate_depth_{new_depth}.csv", "w") as fo:
        writer = csv.writer(fo)
        writer.writerow(["url"])
        for i, url in enumerate(intermediate_depth_urls):
            writer.writerow([url])

    # with open(csv_output_dir / f"extended_depth_{new_depth}.csv", "w") as fo:
    #     writer = csv.writer(fo)
    #     writer.writerow(["id", "domain", "depth"])
    #     for i, domain in enumerate(extended_depth_domains):
    #         writer.writerow([id_offset + i, domain, new_depth])


if __name__ == "__main__":
    main()
