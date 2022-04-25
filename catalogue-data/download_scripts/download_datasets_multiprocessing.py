import argparse
import json
import multiprocessing

from datasets import load_dataset


def download_dataset_multiprocessing(name_dataset):
    try:
        dataset = load_dataset(name_dataset, use_auth_token=True, ignore_verifications=True)
        print("Done for dataset:", name_dataset)
    except:
        print("Failed for dataset:", name_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--ratio_file', type=str, default=None)
    args = parser.parse_args()

    f = open(args.ratio_file, "r")
    ratios = json.load(f)
    list_datasets = [item["dataset_path"] for item in ratios]
    f.close()

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    async_result = p.map_async(download_dataset_multiprocessing, list_datasets)
    p.close()
    p.join()
