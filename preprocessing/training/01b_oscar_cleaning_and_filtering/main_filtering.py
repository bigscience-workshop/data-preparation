"""Filtering."""

from multiprocessing import cpu_count

import argparse
import glob
import os

from datasets import load_dataset, load_from_disk

from filtering import DatasetFiltering, LoadParameters


def check_num_proc(num_proc: int = -1) -> int:
    """
    Check the number of processors. Return a safe-checked value.

    Parameters
    ----------
    num_proc : int, optional
        Number of processors to use, by default -1

    Returns
    -------
    int
        Number of processors to use

    Raises
    ------
    ValueError
        If the input exceeds the number of processors available
    """
    maximum: int = cpu_count()
    if num_proc > maximum:
        raise ValueError(
            f"{num_proc} exceeds the maximum number ({maximum}) of processors"
        )

    if num_proc == -1:
        num_proc = maximum
    else:
        print(f"Using {num_proc} processors out of {maximum} can be slow")

    return num_proc


def parseArgs():
    parser = argparse.ArgumentParser(description="Filtering.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="oscar",
        help="Name of the dataset to load.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="unshuffled_deduplicated_af",
        help="Name of the dataset config to pass.",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default="/mnt/disks/looking_glass_storage/data/",
        help="'load_dataset' returns all files that match the Unix style pattern passed by 'data_files'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to consider.",
    )
    parser.add_argument(
        "--lang_dataset_id",
        type=str,
        default="af",
        help="ID of the language in which the dataset is written.",
    )
    parser.add_argument(
        "--path_fasttext_model",
        type=str,
        default="ac_dc/lid.176.bin",
        help="Path to the Fasttext model used for language identification.",
    )
    parser.add_argument(
        "--path_sentencepiece_model",
        type=str,
        default="ac_dc/af.sp.model",
        help="Path to the Sentence Piece model used to tokenize text for perplexity scores.",
    )
    parser.add_argument(
        "--path_kenlm_model",
        type=str,
        default="ac_dc/af.arpa.bin",
        help="Path to the KenLM model used to compute perplexity scores.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help="Number of processes for multiprocessing. Default at the number of processors available.",
    )
    parser.add_argument(
        "--path_dir_save_dataset",
        type=str,
        default="../dataset_filtered/",
        help="Path to the directory where the filtered version of the dataset will be saved.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()

    """
    dataset = load_dataset(
        args.dataset_name,
        args.config_name,
        data_files=args.data_files,
        split=args.split,
        use_auth_token=True,
    )
    """

    sentencepiece_model = LoadParameters.load_sentencepiece_model(
        args.lang_dataset_id, args.path_sentencepiece_model
    )
    kenlm_model = LoadParameters.load_kenlm_model(
        args.lang_dataset_id, args.path_kenlm_model
    )


    file = args.data_files + args.dataset_name
    print(file)
    print("Loading from disk on", file)
    dataset = load_from_disk(file)
    print("done loading")

    dataset_filtering = DatasetFiltering(
        dataset=dataset,
        lang_dataset_id=args.lang_dataset_id,
        path_fasttext_model=args.path_fasttext_model,
        sentencepiece_model=sentencepiece_model,
        kenlm_model=kenlm_model,
        num_proc=check_num_proc(args.num_proc),
        path_dir_save_dataset=args.path_dir_save_dataset,
        dataset_name=args.dataset_name,
    )
    dataset_filtering.modifying_documents()
    # dataset_filtering.filtering()
    dataset_filtering.save_dataset()

    """
    # ROOTS:
    ds_name = "/mnt/disks/looking_glass_storage/data/" + args.dataset_name
    for file in glob.glob(ds_name + "/*"):
        print(file)
        dataset_name = file.split("/")[-1]
        print("Loading from disk on", file)
        dataset = load_from_disk(file)
        print("done loading")

        dataset_filtering = DatasetFiltering(
            dataset=dataset,
            lang_dataset_id=args.lang_dataset_id,
            path_fasttext_model=args.path_fasttext_model,
            sentencepiece_model=sentencepiece_model,
            kenlm_model=kenlm_model,
            num_proc=check_num_proc(args.num_proc),
            path_dir_save_dataset=args.path_dir_save_dataset,
            dataset_name=dataset_name,
        )
        dataset_filtering.modifying_documents()
        # dataset_filtering.filtering()
        dataset_filtering.save_dataset()
    """


if __name__ == "__main__":
    main()
