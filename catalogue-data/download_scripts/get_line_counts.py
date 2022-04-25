import argparse
import json
import multiprocessing
from collections import Counter
import os
from functools import partial

import datasets
from tqdm import tqdm

FILE_EXTENSIONS = {"text": "txt", "json": "jsonl", "csv": "csv"}


# def find_duplicate_lines_per_batch(examples):
#     all_text = "\n".join(examples["text"])
#     line_counts = Counter([line for line in all_text.split("\n") if len(line) > 0])
#     return {"line_counts": [line_counts]}
#
#
# def find_duplicate_lines_per_dataset_map(dataset_name, num_proc):
#     dataset = datasets.load_dataset(dataset_name, subset, use_auth_token=True, ignore_verifications=True, split=split)
#     batched_sets = dataset.map(find_duplicate_lines_per_batch, batched=True, remove_columns=dataset.column_names,
#                                batch_size=int(1e6), num_proc=num_proc)
#     return sum(*batched_sets["line_counts"])


def find_duplicate_lines_per_dataset_unithread(dataset_name, save_path, max_dataset_size=-1, excluded="code"):
    try:
        dataset = datasets.load_dataset(dataset_name, subset, use_auth_token=True, ignore_verifications=True,
                                        split=split, download_mode="force_redownload")
        print(f"operating on {dataset_name}, length {len(dataset)}")
        if max_dataset_size > 0 and len(dataset) > max_dataset_size:
            print(f"length {len(dataset)} larger than max size {max_dataset_size}")
            return
        if excluded in dataset_name:
            print(f"dataset {dataset_name} excluded as it is {excluded}")
            return
        else:
            print("moving forward with small enough dataset")
    except Exception as e:
        print(f"{e} found for dataset {dataset_name}")
        return
    try:
        all_text = "\n".join(dataset["text"])
    except TypeError:
        dataset = dataset.filter(lambda x: isinstance(x["text"], str))
    line_counts = Counter([line for line in all_text.split("\n") if len(line) > 0])
    line_counts = {k: v for k, v in line_counts.items() if v > 1}
    save_file = os.path.join(save_path, dataset_name.split("/")[-1] + ".json")
    json.dump(line_counts, open(save_file, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--dataset_list', type=str, default=None)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--text_feature_key', type=str, default="text")
    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--reuse_previous', action="store_true")

    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    split = args.split
    subset = args.subset
    key = args.text_feature_key
    num_proc = args.num_proc
    all_datasets = [line.strip() for line in open(args.dataset_list).readlines() if len(line.strip()) > 0]
    if args.reuse_previous:
        all_datasets = [dataset_name for dataset_name in all_datasets if
                        not os.path.exists(os.path.join(save_path, dataset_name.split("/")[-1] + ".json"))]
    if num_proc < 2:
        for dataset_name in tqdm(all_datasets):
            find_duplicate_lines_per_dataset_unithread(dataset_name, save_path=save_path)
    else:
        p = multiprocessing.Pool(num_proc)
        results = p.imap_unordered(partial(find_duplicate_lines_per_dataset_unithread, save_path=save_path),
                                   all_datasets)
        for result in tqdm(results):
            pass
