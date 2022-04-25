import re
from typing import Dict


def parse_meta(meta) -> Dict:
    if isinstance(meta, str):
        meta = eval(meta)
    return meta


normalise_dataset_name_regex = re.compile(
    r"^(?:/gpfswork/rech/six/uty16tp/dataset/tokenization/)?(bigscience-catalogue-lm-data/[^/]+)(?:/data)?$"
)


language_regex = re.compile(
    r"^(?:/gpfswork/rech/six/uty16tp/dataset/tokenization/)?bigscience-catalogue-lm-data/lm_([^_]+)_.*(?:/data)?$"
)
def get_language(dataset_name: str):
    lang_candidate = language_regex.match(dataset_name).group(1)

    # Normalise chinese languages, so that we only consider simplified and traditional chinese as the two chinese languages
    if lang_candidate in ["zh", "zhs", "zh-cn"]:
        lang_candidate = "zhs"
    elif lang_candidate in ["zht", "zh-tw"]:
        lang_candidate = "zht"
    else:
        assert lang_candidate[:2] != "zh"

    return lang_candidate
