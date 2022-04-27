import argparse
import glob
import json
import os
from multiprocessing import Pool

from tqdm import tqdm


def remove_duplicate_lines(example):
    seen = set()
    example["text"] = "\n".join([x for x in example["text"].split("\n") if len(x) == 0 or not (x in seen or seen.add(x))])
    return example


def process_dir(dir):
    os.makedirs(os.path.join(save_dir, dir.split("/")[-1]), exist_ok=True)
    with open(os.path.join(dir, "data.jsonl")) as data_file, open(os.path.join(save_dir, dir.split("/")[-1], "data.jsonl"), "w") as out_file:
        for line in data_file:
            data = remove_duplicate_lines(json.loads(line))
            out_file.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--num_proc', type=int, default=60)

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    num_proc = args.num_proc

    dirs = glob.glob(f"{data_dir}/*/")

    p = Pool(num_proc)

    for item in tqdm(p.map(process_dir, dirs)):
        pass

    # for dir in tqdm(dirs):
    #     process_dir(dir)
