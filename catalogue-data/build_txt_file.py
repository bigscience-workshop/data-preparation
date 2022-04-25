import argparse

from datasets import load_dataset, tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Dataset path holding jsonl.gz shards",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        required=True,
        help="In which file to store resulting dataset"
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    dataset = load_dataset(args.dataset_path, data_files="**.jsonl", split="train")

    with open(args.save_file, "w") as fo:
        for row in tqdm(dataset):
            fo.write(f"{row['text']}\n")

if __name__ == "__main__":
    main()
