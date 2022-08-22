import argparse
import re
from pathlib import Path

import pandas as pd

normalise_dataset_name_regex = re.compile(
    r"^(?:/gpfswork/rech/six/uty16tp/dataset/tokenization/)?(bigscience-catalogue-lm-data/[^/]+)(?:/data)?$"
)


def get_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument("--dataset-csv-path", type=Path)
    parser.add_argument("--index", type=int)
    # Parse args
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data = pd.read_csv(args.dataset_csv_path)
    dataset = data.iloc[args.index]
    print(normalise_dataset_name_regex.match(dataset["dataset_name"]).group(1))


if __name__ == "__main__":
    main()
