from datasets import load_dataset
import pandas as pd

def main():
    all_datasets = pd.read_csv("/gpfswork/rech/six/uty16tp/code/big_science/data-preparation/preprocessing/training/training_v2.csv")
    all_dataset_names = all_datasets["dataset_name"]

    for dataset_name in all_dataset_names:
        load_dataset(dataset_name, split="train", use_auth_token=True, ignore_verifications=True)

if __name__ == "__main__":
    main()