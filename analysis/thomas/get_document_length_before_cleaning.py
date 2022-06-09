import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-csv-path", type=str)
    parser.add_argument("--num-proc", type=int, default=40)
    parser.add_argument("--output-prefix", type=str, default="/gpfsscratch/rech/six/uty16tp/dataset_before_cleaning_pipeline")
    parser.add_argument("--output-metadata-file", type=str, default="/gpfsscratch/rech/six/uty16tp/dataset_before_cleaning_pipeline/metadata.txt")
    return parser.parse_args()

def normalise_dataset_name(dataset_name: str) -> str:
    if dataset_name.startswith("/gpfswork/rech/six/uty16tp/dataset/tokenization/"):
        if dataset_name.endswith("/data"):
            return dataset_name[48:-5]
        else:
            return dataset_name[48:]
    else:
        return dataset_name

def get_text_byte(batch: Dict[str, List[Any]]) -> Dict[str, List[int]]:
    return {
        "bytes": [len(text.encode()) for text in batch["text"]]
    }

def main():
    args=get_args()

    all_datasets = pd.read_csv(args.training_csv_path)
    all_dataset_names = all_datasets["dataset_name"]

    dataset_sizes = {}
    for dataset_name in all_dataset_names:
        normalised_dataset_name = normalise_dataset_name(dataset_name)
        ds = load_dataset(dataset_name, split="train", use_auth_token=True, ignore_verifications=True)
        bytes_per_sample = ds.map(get_text_byte, num_proc=args.num_proc, batched=True)
        save_path = Path(args.output_prefix) / normalised_dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        bytes_per_sample.save_to_disk(str(save_path.absolute()))
        assert normalised_dataset_name not in dataset_sizes
        dataset_sizes[normalised_dataset_name] = sum(bytes_per_sample["bytes"])

    with open(args.output_metadata_file, "w") as fi:
        json.dump(dataset_sizes, fi)

if __name__ == "__main__":
    main()