from datasets import load_dataset
from tqdm import tqdm
import json

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ""))

from filtering import LoadParameters, ModifyingDocuments, Filtering


class GetDataForVisualization:
    def __init__(
        self,
        dataset,
        lang_dataset_id,
        path_sentencepiece_model,
        path_kenlm_model,
        path_save_stats,
    ):

        self.ds = dataset

        self.lang_dataset_id = lang_dataset_id

        self.param = LoadParameters.load_parameters(lang_dataset_id)
        self.stopwords = LoadParameters.load_stopwords(lang_dataset_id)
        self.flagged_words = LoadParameters.load_flagged_words(lang_dataset_id)
        self.sentencepiece_model = LoadParameters.load_sentencepiece_model(
            lang_dataset_id, path_sentencepiece_model
        )
        self.sentencepiece_model_tok = (
            self.sentencepiece_model if self.param["tokenization"] else None
        )
        self.kenlm_model = LoadParameters.load_kenlm_model(
            lang_dataset_id, path_kenlm_model
        )

        self.path_save_stats = path_save_stats

    def compute_stats(self):

        def _compute_stats(example):

            character_repetition_ratios = {
                n: round(
                    Filtering.compute_character_repetition_ratio(example["text"], n), 4
                )
                for n in range(2, 16)
            }
            example[
                "character_repetition_ratio"
            ] = character_repetition_ratios

            word_repetition_ratios = {
                n: round(
                    Filtering.compute_word_repetition_ratio(
                        example["text"],
                        self.sentencepiece_model_tok,
                        self.param["strip_characters"],
                        n,
                    ),
                    4,
                )
                for n in range(3, 11)
            }
            example["word_repetition_ratio"] = word_repetition_ratios

            special_characters_ratio = Filtering.compute_special_characters_ratio(
                example["text"], self.param["special_characters"]
            )
            example["special_characters_ratio"] = special_characters_ratio

            if self.stopwords:
                stopwords_ratio = Filtering.compute_stopwords_ratio(
                    example["text"],
                    self.sentencepiece_model_tok,
                    self.param["strip_characters"],
                    self.param["cond_words_augmentation"],
                    self.param["words_augmentation_group_sizes"],
                    self.param["words_augmentation_join_char"],
                    self.stopwords,
                )
                example["stopwords_ratio"] = stopwords_ratio

            if self.flagged_words:
                flagged_words_ratio = Filtering.compute_flagged_words_ratio(
                    example["text"],
                    self.sentencepiece_model_tok,
                    self.param["strip_characters"],
                    self.param["cond_words_augmentation"],
                    self.param["words_augmentation_group_sizes"],
                    self.param["words_augmentation_join_char"],
                    self.flagged_words,
                )
                example["flagged_words_ratio"] = flagged_words_ratio

            if self.kenlm_model:
                perplexity_score = Filtering.compute_perplexity_score(
                    example["text"], self.sentencepiece_model, self.kenlm_model
                )
                example["perplexity_score"] = perplexity_score

        mapped_ds = self.ds.map(_compute_stats, num_proc=os.cpu_count())
        mapped_ds.to_json(self.path_save_stats)


if __name__ == "__main__":

    lang_dataset_id = "en"

    dataset_name = "csv"  # "TurkuNLP/register_oscar"
    config_name = None
    data_files = "/home/teven_huggingface_co/helen_cc_bad.csv"  # f"{lang_dataset_id}/{lang_dataset_id}_00000.jsonl.gz"
    split = "train"

    max_size = 10000

    lang_dataset_id = lang_dataset_id
    path_sentencepiece_model = f"/home/teven_huggingface_co/data-preparation/preprocessing/training/01b_oscar_cleaning_and_filtering/models/{lang_dataset_id}.sp.model"
    path_kenlm_model = f"/home/teven_huggingface_co/data-preparation/preprocessing/training/01b_oscar_cleaning_and_filtering/models/{lang_dataset_id}.arpa.bin"
    path_save_stats = f"helen_cc_bad_examples_with_stats.json"

    dataset = load_dataset(
        dataset_name,
        config_name,
        data_files=data_files,
        split=split,
    )


    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), max_size)))

    get_data_for_visualization = GetDataForVisualization(
        dataset,
        lang_dataset_id,
        path_sentencepiece_model,
        path_kenlm_model,
        path_save_stats,
    )
    get_data_for_visualization.compute_stats()
