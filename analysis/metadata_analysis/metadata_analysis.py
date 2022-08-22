import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LANG_MAP = {
    "zh-HK": "zh",
    "zh-TW": "zh",
    "zh-CN": "zh",
    "zht": "zh",
    "zhs": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
}


def simplify_meta_dict(meta_dict, name):
    simple_dict = {"name": name.split("/")[1]}
    key = list(meta_dict.keys())[0]
    inner_dict = meta_dict[key]
    simple_dict["size"] = inner_dict["dataset_size"]
    simple_dict["samples"] = 0
    simple_dict["splits"] = len(inner_dict["splits"])
    for split in inner_dict["splits"]:
        simple_dict["samples"] += inner_dict["splits"][split]["num_examples"]
    return simple_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_datasets_txt", type=Path)
    parser.add_argument("--path_to_meta_jsonl", type=Path)
    parser.add_argument("--save_path_to_meta_table_csv", type=Path)
    parser.add_argument("--save_path_to_meta_table_aggregated_csv", type=Path)
    parser.add_argument("--save_path_to_meta_dataset_size_fid_pdf", type=Path)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    dataset_names, meta_dicts = [], []
    with open(args.path_to_datasets_txt, "r") as f_data:
        with open(args.path_to_meta_jsonl, "r") as f_meta:
            for i, (name_line, meta_line) in enumerate(zip(f_data, f_meta)):
                if len(meta_line.strip()) == 0:
                    continue
                else:
                    dataset_names.append(name_line.strip())
                    try:
                        meta_dicts.append(json.loads(meta_line))
                    except:
                        print(f"error in file {name_line}")

    simple_dicts = [
        simplify_meta_dict(meta_dict, name)
        for (meta_dict, name) in zip(meta_dicts, dataset_names)
    ]
    df = pd.DataFrame.from_records(simple_dicts)

    df["lang"] = df["name"].apply(lambda x: x.split("_")[1])
    df["name"] = df["name"].apply(lambda x: "_".join(x.split("_")[2:]))

    df.to_csv(args.save_path_to_meta_table_csv, index=False)

    df["lang"] = df["lang"].apply(lambda x: LANG_MAP[x] if x in LANG_MAP else x)
    df["lang"] = df["lang"].apply(lambda x: "nigercongo" if "nigercongo" in x else x)

    df_agg = (
        df.groupby("lang")[["size", "samples"]]
        .sum()
        .sort_values(by="size", ascending=False)
        .reset_index()
    )
    df_agg["bytes-per-sample"] = df_agg["size"] / df_agg["samples"]
    df_agg["size-gb"] = df_agg["size"] / 1024**3
    df_agg.to_csv(args.save_path_to_meta_table_aggregated_csv, index=False)

    plt.figure(figsize=(10, 7))
    df_agg.set_index("lang")["size"].plot(kind="bar")
    plt.yscale("log")
    labels = ["kB", "MB", "GB"]
    for i in range(3):
        plt.hlines([1024 ** (i + 1)], -0.5, len(df_agg), color="black", linestyle="--")
        plt.annotate(labels[i], (len(df_agg) - 2, 1.2 * 1024 ** (i + 1)))
    plt.xlabel("")
    plt.ylabel("Size [Bytes]")
    plt.tight_layout()
    plt.savefig(args.save_path_to_meta_dataset_size_fid_pdf)

if __name__ == "__main__":
    main()
