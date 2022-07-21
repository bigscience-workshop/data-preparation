from itertools import groupby

from datasets import Dataset

from clean_helpers.utils import parse_meta


def concatenate_lm_fr_ester(ds: Dataset, num_proc: int, batch_size: int) -> Dataset:
    dataset_in_memory = [
        (*parse_meta(row["meta"])["id"].split("_id_"), row["text"]) for row in ds
    ]
    dataset_in_memory.sort()
    new_texts = []
    new_metas = []
    for doc_id, segments in groupby(dataset_in_memory, key=lambda x: x[0]):
        sorted_segment = sorted(
            [elt[1:] for elt in segments],
            key=lambda x: int(x[0])
        )
        new_texts.append("\n".join([elt[1] for elt in sorted_segment]))
        new_metas.append({"id": doc_id})
    
    new_ds = Dataset.from_dict({"text": new_texts, "meta": new_metas})
    return new_ds
