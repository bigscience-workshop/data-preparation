from typing import Optional, Dict, List

import pandas as pd
import re

from clean_helpers.utils import get_language, normalise_dataset_name_regex


def get_dedup_args(row: Dict) -> Optional[str]:
    ds_name = normalise_dataset_name_regex.match(row["dataset_name"]).group(1)

    # code only runs document deduplication.
    lang = get_language(ds_name)
    if lang == "code":
        return " ".join(["dedup_document", "filter_remove_empty_docs"])

    if "pseudocrawl" in ds_name:
        list_of_dedups = ["dedup_document_on_url", "dedup_document", "dedup_pseudocrawl_newspapers"]
    else:
        list_of_dedups = ["dedup_document"]

    if all(black_list_ds not in ds_name for black_list_ds in ["open_subtitles", "europarl", "uncorpus", "pseudocrawl"]):
        list_of_dedups += ["dedup_template_soft"]

    list_of_dedups += ["filter_remove_empty_docs"]
    return " ".join(list_of_dedups)

language_to_short_filter_document = {
    "ar": 300,
    "ca": 1024,
    "code": 1024,
    "en": 1024,
    "es": 1024,
    "eu": 0,
    "fr": 1024,
    "id": 300,
    "indic-as": 0,
    "indic-bn": 300,
    "indic-gu": 300,
    "indic-hi": 300,
    "indic-kn": 300,
    "indic-ml": 300,
    "indic-mr": 300,
    "indic-ne": 300,
    "indic-or": 0,
    "indic-pa": 300,
    "indic-ta": 300,
    "indic-te": 300,
    "indic-ur": 300,
    "nigercongo-sw": 0,
    "nigercongo-yo": 0,
    "nigercongo-rw": 0,
    "nigercongo-xh": 0,
    "nigercongo-ig": 0,
    "nigercongo-zu": 0,
    "nigercongo-sn": 0,
    "nigercongo-lg": 0,
    "nigercongo-wo": 0,
    "nigercongo-rn": 0,
    "nigercongo-fon": 0,
    "nigercongo-nso": 0,
    "nigercongo-ln": 0,
    "nigercongo-tn": 0,
    "nigercongo-tw": 0,
    "nigercongo-ny": 0,
    "nigercongo-st": 0,
    "nigercongo-ts": 0,
    "nigercongo-ak": 0,
    "nigercongo-bm": 0,
    "nigercongo-ki": 0,
    "nigercongo-tum": 0,
    "pt": 300,
    "vi": 300,
    "zhs": 1024,
    "zht": 1024,
}
def get_filter_on_small_documents_args(row: Dict) -> Optional[str]:
    language = get_language(row["dataset_name"])

    filter_min_length = language_to_short_filter_document[language]
    if filter_min_length > 0:
        return f"filter_small_docs_bytes_{filter_min_length}"
    else:
        return None

def main():
    data = pd.read_csv("training.csv")

    data["dedups"] = data.apply(get_dedup_args, axis=1)
    print(data[:5]["dedups"])

    data["filter_short_documents"] = data.apply(get_filter_on_small_documents_args, axis=1)
    print(data[:5]["filter_short_documents"])

    data.to_csv("training_with_dedups.csv")
    
if __name__ == "__main__":
    main()