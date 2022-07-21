import argparse
import json
import multiprocessing

from datasets import load_dataset
from tqdm import tqdm


def download_dataset_multiprocessing(name_dataset):
    try:
        dataset = load_dataset(name_dataset, use_auth_token=True,
                               ignore_verifications=True)
        samples = "".join(dataset["train"].select(range(10))["text"])
        if samples.count("/n") > samples.count("\n"):
            dataset = load_dataset(name_dataset, use_auth_token=True,
                                   ignore_verifications=True, download_mode="force_redownload")
            print(f"Reloaded for dataset {name_dataset}")
            samples = "".join(dataset["train"].select(range(10))["text"])
            if samples.count("/n") > samples.count("\n"):
                print(f"Dataset {name_dataset} still has wrong newlines")
        else:
            print(f"Done for dataset {name_dataset}")
    except Exception as e:
        print(e)
        print(f"Failed for dataset {name_dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--ratio_file', type=str, default=None)
    args = parser.parse_args()

    f = open(args.ratio_file, "r")
    ratios = json.load(f)
    list_datasets = [item["dataset_path"] for item in ratios]
    f.close()

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    results = p.imap_unordered(download_dataset_multiprocessing, list_datasets)
    for dataset in tqdm(results):
        pass
    p.close()
    p.join()
