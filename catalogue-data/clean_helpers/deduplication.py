from collections import defaultdict
from functools import partial
from typing import List, Set, Tuple, Dict, Callable, Optional
import hashlib
import re
import string
import urllib

from datasets import Dataset


# ======== DEDUPLICATION FUNCTIONS ===================
from clean_helpers.utils import parse_meta


def build_dedup_template(min_template_line_size: int, min_template_line_occurence: int):
    def dedup_template(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
        """Computes and remove templates lines"""
        # Compute the hash of each lines
        split_into_lines_and_hashes = ds.map(
            split_text_to_lines_and_hash,
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names
        )
        lines_and_hashes = split_into_lines_and_hashes.remove_columns(
            set(split_into_lines_and_hashes.column_names) - {"lines", "hashes"}
        )

        # TODO: Batch read could help make it faster
        # Find template lines
        count_lines_occurence = defaultdict(lambda: 0)
        for row in lines_and_hashes:
            filtered_lines_and_hashes = [
                (line, hash_)
                for line, hash_ in zip(row["lines"], row["hashes"])
                if len(line) >= min_template_line_size
            ]
            for _, hash_ in filtered_lines_and_hashes:
                count_lines_occurence[hash_] += 1

        template_line_hashes = {k for k, v in count_lines_occurence.items() if v >= min_template_line_occurence}
        del count_lines_occurence

        # Clean dataset
        return split_into_lines_and_hashes.map(
            build_remove_template_lines(template_line_hashes),
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=split_into_lines_and_hashes.column_names
        )

    return dedup_template


def build_dedup_document(batch_normalizer: Callable[[Dict], List[str]]):
    def dedup_document(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
        hashed_documents = ds.map(
            lambda batch: {**batch, "hash": get_hash(batch_normalizer(batch))},
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
        )

        hashes = set()

        return hashed_documents.map(
            partial(delete_text_from_duplicates, hashes=hashes),
            num_proc=1,  # VERY IMPORTANT: hashes will be updated, and is not thread safe.
            batched=True,
            batch_size=batch_size,
            remove_columns=hashed_documents.column_names
        )

        # # TODO: This version is memory efficient and runs faster, but we lose alignment
        # return hashed_documents.filter(
        #     lambda hashes_: [is_new_hash(hash_, hashes) for hash_ in hashes_],
        #     num_proc=1,  # VERY IMPORTANT: hashes will be updated, and is not thread safe.
        #     input_columns=["hash"],
        #     batched=True,
        #     batch_size=batch_size,
        # ).remove_columns("hash")
    return dedup_document


# =========== HELPERS ===============

def get_hash(texts: List[str]) -> List[str]:
    """Get hash of content field."""
    return [hashlib.md5(text.strip().encode("utf-8")).hexdigest() for text in texts]

def split_text_in_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n")]

def split_text_to_lines_and_hash(batch: Dict[str, List]):
    lines_per_texts = [split_text_in_lines(text) for text in batch["text"]]
    return {
        **{k: v for k, v in batch.items() if k != "text"},
        "lines": lines_per_texts,
        "hashes": [get_hash(lines) for lines in lines_per_texts]
    }


def clean_text(lines_and_hashes: List[Tuple[str, int]], template_line_hashes: Set[str]):
    return "\n".join([line for line, hash_ in lines_and_hashes if hash_ not in template_line_hashes])


def build_remove_template_lines(template_line_hashes: Set[str]):
    def remove_template_lines(batch: Dict[str, List]):
        cleaned_texts = [
            clean_text(
                list(zip(lines, hashes)),
                template_line_hashes
            )
            for lines, hashes in zip(batch["lines"], batch["hashes"])
        ]
        return {
            **{
                key: value
                for key, value in batch.items()
                if key not in ["lines", "hashes"]
            },
            "text": [cleaned_text for cleaned_text in cleaned_texts]
        }

    return remove_template_lines


def is_new_hash(hash_: str, hashes: Set[str]) -> bool:
    """Check if current hash is still in set of unique hashes and remove if true."""
    if hash_ in hashes:
        return False
    else:
        hashes.add(hash_)
        return True

def delete_text_from_duplicates(batch: Dict[str, List], hashes: Set[str]) -> Dict[str, List]:
    return {
        **{k: v for k, v in batch.items() if k != "hash"},
        "text": [text if is_new_hash(hash_, hashes) else "" for text, hash_ in zip(batch["text"], batch["hash"])]
    }

def url_with_only_some_query_param(url: str, query_param_map: Optional[dict] = None) -> str:
    url_parse = urllib.parse.urlparse(url)
    query = url_parse.query

    url_query_params = urllib.parse.parse_qsl(query)

    if query_param_map is None:
        url_query_params_new = {}
    else:
        url_query_params_new = [(query_param_map[old_key], old_value) for (old_key, old_value) in url_query_params if old_key in query_param_map]
        
    url_new_query = urllib.parse.urlencode(url_query_params_new, encoding="utf-8")
    url_parse = url_parse._replace(query=url_new_query)
    new_url = urllib.parse.urlunparse(url_parse)
    return new_url

# =========== BATCH NORMALISER ===============


# this only keeps letter characters
remove_non_character_regex = re.compile(f'\s+|\d+|[{re.escape(string.punctuation)}]')
def document_batch_normalizer(batch: Dict) -> List[str]:
    return [remove_non_character_regex.sub('', text) for text in batch["text"]]


def strict_url_batch_normalizer(batch: Dict) -> List[str]:
    return [parse_meta(meta)["url"] for meta in batch["meta"]]


url_host_and_path_regex = re.compile(r"^(.[^?]*)")
def url_host_and_path_batch_normalizer(batch: Dict) -> List[str]:
    return [url_host_and_path_regex.match(parse_meta(meta)["url"]).group(1) for meta in batch["meta"]]

lm_es_pseudocrawl_filtered_341_es_cointelegraph_com_regex = re.compile(r"^((?:(?!/amp)/?(?:[^?/]*))+)(?:/amp)?")
def url_lm_es_pseudocrawl_filtered_341_es_cointelegraph_com(batch: Dict) -> List[str]:
    return [lm_es_pseudocrawl_filtered_341_es_cointelegraph_com_regex.match(parse_meta(meta)["url"]).group(1) for meta in batch["meta"]]

def url_lm_en_pseudocrawl_filtered_619_www_qut_edu_au(batch: Dict) -> List[str]:
    return [url_with_only_some_query_param(parse_meta(meta)["url"], {"id": "id", "news-id": "id"}) for meta in batch["meta"]]
