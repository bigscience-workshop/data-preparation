import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bigscience_corpus_doc_length_dir", type=Path, required=True)
    parser.add_argument("--statistics-pickle-file", type=Path, required=True)
    args = parser.parse_args()
    return args


def normalize_lang_codes(lang):
    # Normalise chinese languages, so that we only consider simplified and traditional chinese as the two chinese
    # languages
    if lang in ["zh", "zhs", "zh-cn"]:
        lang = "zhs"
    elif lang in ["zht", "zh-tw"]:
        lang = "zht"
    return lang


def get_datasets_per_lang(dataset_dir):
    dataset_paths = list(dataset_dir.iterdir())

    dataset_paths_per_lang = defaultdict(list)
    for dataset_path in dataset_paths:

        lang = dataset_path.name[len("cleaned_lm_") :].split("_")[0]
        lang = normalize_lang_codes(lang)

        dataset_paths_per_lang[lang].append(dataset_path)
    return dataset_paths_per_lang


def compute_stats_per_ds(dataset_path, lang):
    ds = load_from_disk(str(dataset_path))
    data_points_list = ds["len"][:]
    return (
        np.mean(data_points_list),
        np.median(data_points_list),
        data_points_list,
        dataset_path.name[len(f"cleaned_lm_{lang}_") :],
    )


def main():
    args = get_args()
    dataset_dir = args.bigscience_corpus_doc_length_dir
    dataset_paths_per_lang = get_datasets_per_lang(dataset_dir)

    all_data_point = {}
    for lang in dataset_paths_per_lang.keys():
        print(f"Processing {lang}")
        data_points = []
        for dataset_path in dataset_paths_per_lang[lang]:
            data_points.append(compute_stats_per_ds(dataset_path, lang))
        all_data_point[lang] = data_points

    with open(args.statistics_pickle_file, "wb") as handle:
        pickle.dump(all_data_point, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
