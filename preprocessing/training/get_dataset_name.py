import argparse
from pathlib import Path
import pandas as pd

from clean_helpers.utils import normalise_dataset_name_regex


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
