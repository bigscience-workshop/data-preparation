import json
import shutil
from collections import defaultdict
import os
import argparse
import logging

from datasets import load_from_disk, load_dataset
from datasets.utils.logging import set_verbosity_info

"""
Cleaning text:
 - run exact deduplication
"""

set_verbosity_info()
logger = logging.getLogger(__name__)

META_COLUMNS = ["meta"]
TEXT_COLUMN = "text"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir", required=True, type=str, help="Where to save the datasets."
    )
    parser.add_argument(
        "--dataset_dir",
        help="path to where the arrow dataset is located",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size used for mapping the dataset",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num-proc",
        help="Number of processors used for the mapping of the dataset",
        required=True,
        type=int,
    )
    parser.add_argument("--load-from-disk", action="store_true")
    args = parser.parse_args()
    return args


def text_is_not_none(batch):
    return [text is not None for text in batch["text"]]


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()

    dset = (
        load_from_disk(args.dataset_dir)
        if args.load_from_disk
        else load_dataset(args.dataset_dir, data_files="**.jsonl", split="train")
    )

    # pre-remove unecessary columns, hopefully that saves qui a bit of memory usage
    columns_to_keep = [TEXT_COLUMN] + META_COLUMNS
    dset = dset.remove_columns(list(set(dset.column_names) - set(columns_to_keep)))

    # Filter None text columns
    number_of_samples_before = len(dset)
    dset = dset.filter(text_is_not_none, batched=True, num_proc=args.num_proc)
    number_of_samples_after_filtering_none = len(dset)
    logger.info(
        f"Filtered out {number_of_samples_before - number_of_samples_after_filtering_none} / {number_of_samples_before}"
    )

    seen = defaultdict(lambda: 0)

    def remove_duplicate_lines(examples):
        new_exemples = {"text": [], "meta": []}
        for text, meta in zip(examples["text"], examples["meta"]):
            if "lm_code" in eval(meta)["source_dataset"]:
                # we preserve the code examples
                new_exemples["text"].append(text)
                new_exemples["meta"].append(meta)
                continue

            new_text = []
            for line in text.split("\n"):
                line = line.strip()
                if len(line) == 0 or line in seen:
                    seen[line] += 1
                    continue
                new_text.append(line)
                seen[line] += 1
            new_exemples["text"].append("\n".join(new_text))
            new_exemples["meta"].append(meta)
        return new_exemples

    dset = dset.map(
        remove_duplicate_lines,
        batched=True,
        # num_proc=args.num_proc, # single proccess for seen dict
        batch_size=args.batch_size,
        remove_columns=dset.column_names,
    )
    logger.info(f"Finished cleaning")

    repo_name = args.save_dir
    # TODO build a caching mechanism
    repo_name_tmp = f"{repo_name}.tmp"
    if not os.path.isdir(repo_name_tmp):
        os.makedirs(repo_name_tmp)

    # write to folder
    dset.save_to_disk(repo_name_tmp)

    logger.info(f"Ended successfully, saved at {repo_name_tmp}")

    # Saving skipped lines that are considered repetitive
    with open(os.path.join(repo_name_tmp, "skipped_lines.json"), "w") as fi:
        json.dump(list(line for line, num in seen.items() if num > 1), fi, indent=2)

    # Move so that the state becomes completed
    shutil.move(repo_name_tmp, repo_name)

    logger.info("Finished")


if __name__ == "__main__":
    main()
