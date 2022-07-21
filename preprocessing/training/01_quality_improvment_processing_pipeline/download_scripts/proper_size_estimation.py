import argparse
import json
import multiprocessing
from functools import partial

import datasets
from datasets import load_dataset
from tqdm import tqdm

def get_size_per_example(texts):
    size_values = [len(text.encode()) for text in texts]
    examples = {"bytes_len": size_values}
    return examples

def get_size(name_dataset, args):
    try:
        if args.load_from_disk:
            dataset = load_dataset(name_dataset, use_auth_token=True, ignore_verifications=True, split="train")
        else:
            dataset = datasets.load_from_disk(name_dataset)

        dataset = dataset.map(
            partial(get_size_per_example,
            batched=True, 
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            input_columns=["text"],
            remove_columns=dataset.column_names)
        )
        len_bytes = sum(dataset["bytes_len"][:])
        print("Done for dataset:", name_dataset)
        return (name_dataset, len_bytes)
    except Exception as e:
        print(f"Failed for dataset: {name_dataset} because of {e}")
        return (name_dataset, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--dataset_list', type=str, default=None)
    parser.add_argument('--load_from_disk', action="store_true")
    parser.add_argument('--reuse_previous', action="store_true")
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    list_datasets = [dataset_path.strip() for dataset_path in open(args.dataset_list).readlines()]
    print(len(list_datasets))

    if args.reuse_previous:
        previous_sizes = json.load(open("dataset_sizes.json"))
        list_datasets = [dataset_name for dataset_name in list_datasets if previous_sizes.get(dataset_name, 0) == 0]
    else:
        previous_sizes = None

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    async_result = p.imap_unordered(get_size, list_datasets)
    result = dict(tqdm(async_result))
    if previous_sizes is not None:
        result = dict(previous_sizes, **result)
    json.dump(result, open("dataset_sizes.json", "w"), ensure_ascii=False, indent=2)
    with open("dataset_sizes.csv", "w") as g:
        for dataset_name, size in sorted(result.items(), key=lambda x: x[0]):
            g.write(f"{dataset_name},{size * 1e-9:.4f}\n")
    p.close()
    p.join()
