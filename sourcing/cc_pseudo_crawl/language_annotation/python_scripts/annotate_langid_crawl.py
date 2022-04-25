import argparse

import datasets
from datasets import load_dataset, Features

import pathlib

import fasttext

from multiprocessing import cpu_count


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Split the crawl dataset and annotate the Fasttext language id information."
    )
    parser.add_argument(
        "seed_id",
        type=int,
        help="Id of the seed to split the dataset on with this script.",
    )
    args = parser.parse_args()
    return args


def get_features():
    null = None
    features = {
        "HtmlPreprocessor_error": {"dtype": "int64", "id": null, "_type": "Value"},
        "HtmlPreprocessor_error_comment": {
            "dtype": "string",
            "id": null,
            "_type": "Value",
        },
        "content_languages": {"dtype": "string", "id": null, "_type": "Value"},
        "content_mime_detected": {"dtype": "string", "id": null, "_type": "Value"},
        "depth": {"dtype": "int16", "id": null, "_type": "Value"},
        "download_exception": {"dtype": "string", "id": null, "_type": "Value"},
        "external_urls": [{"dtype": "string", "id": null, "_type": "Value"}],
        "fetch_redirect": {"dtype": "string", "id": null, "_type": "Value"},
        "fetch_status": {"dtype": "int32", "id": null, "_type": "Value"},
        "fetch_time": {"dtype": "timestamp[ns]", "id": null, "_type": "Value"},
        "html_error": {"dtype": "string", "id": null, "_type": "Value"},
        "html_footer": [{"dtype": "string", "id": null, "_type": "Value"}],
        "html_head": [{"dtype": "string", "id": null, "_type": "Value"}],
        "html_str": {"dtype": "string", "id": null, "_type": "Value"},
        "html_title": [{"dtype": "string", "id": null, "_type": "Value"}],
        "metadata_html": [
            {
                "char_end_idx": {"dtype": "int64", "id": null, "_type": "Value"},
                "char_start_idx": {"dtype": "int64", "id": null, "_type": "Value"},
                "html_attrs": {
                    "attrs": [{"dtype": "string", "id": null, "_type": "Value"}],
                    "values": [{"dtype": "string", "id": null, "_type": "Value"}],
                },
                "key": {"dtype": "string", "id": null, "_type": "Value"},
                "relative_end_pos": {"dtype": "int64", "id": null, "_type": "Value"},
                "relative_start_pos": {"dtype": "int64", "id": null, "_type": "Value"},
                "type": {"dtype": "string", "id": null, "_type": "Value"},
                "value": {"dtype": "string", "id": null, "_type": "Value"},
            }
        ],
        "seed_id": {"dtype": "int32", "id": null, "_type": "Value"},
        "text": {"dtype": "string", "id": null, "_type": "Value"},
        "url": {"dtype": "string", "id": null, "_type": "Value"},
        "url_host_name": {"dtype": "string", "id": null, "_type": "Value"},
        "url_host_registered_domain": {"dtype": "string", "id": null, "_type": "Value"},
        "url_host_tld": {"dtype": "string", "id": null, "_type": "Value"},
        "url_surtkey": {"dtype": "string", "id": null, "_type": "Value"},
        "warc_filename": {"dtype": "string", "id": null, "_type": "Value"},
        "warc_record_length": {"dtype": "int32", "id": null, "_type": "Value"},
        "warc_record_offset": {"dtype": "int32", "id": null, "_type": "Value"},
    }

    def convert_types(features):
        if isinstance(features, dict) and "_type" in features:
            return getattr(datasets, features["_type"])(features["dtype"])
        elif isinstance(features, dict):
            return {key: convert_types(value) for key, value in features.items()}
        elif isinstance(features, list):
            return [convert_types(value) for value in features]

    final_features = convert_types(features)
    final_features = Features(final_features)
    return final_features


def load_fasttext_model(path_fasttext_model):
    return fasttext.load_model(path_fasttext_model)


def get_fasttext_info(line, model_lang_id):
    """The line should be in lower case and without \n in it."""
    pred = model_lang_id.predict(line)
    lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
    score_pred = pred[1][0]
    return lang_pred_fasttext_id, score_pred


def get_all_fasttext_info(document, model_lang_id):
    if not document:
        document = ""
    document = document.lower()
    lang_pred_fasttext_id, score_pred = get_fasttext_info(
        document.replace("\n", " "), model_lang_id
    )
    info = {
        "lang_pred_fasttext_id": lang_pred_fasttext_id,
        "score_pred": score_pred,
        "on_lines": [
            {
                "id_line": id_line,
                "number_caracters_line": len(line),
                "lang_pred_fasttext_id_line": result_fasttext_line[0],
                "score_pred_line": result_fasttext_line[1],
            }
            for id_line, line in enumerate(document.split("\n"))
            for result_fasttext_line in [get_fasttext_info(line, model_lang_id)]
        ],
    }
    return info


class FunctionDatasetModifyingDocuments:
    def __init__(self, path_fasttext_model):
        self.path_fasttext_model = path_fasttext_model
        self.model_lang_id = load_fasttext_model(path_fasttext_model)

    def __call__(self, example):
        example["fasttext_pred"] = get_all_fasttext_info(
            example["text"], self.model_lang_id
        )
        return example

    def __reduce__(self):
        return (self.__class__, (self.path_fasttext_model,))


def main():
    args = parseArgs()

    data_files = [
        f"/gpfsscratch/rech/six/commun/pseudo_crawl/fasttext_annotation/seeds_batch_1/datasets-compressed-shards/bigscience-catalogue-data/seed_id={args.seed_id}/text__html/*.jsonl.gz"
    ]
    dataset = load_dataset("json", data_files=data_files, features=get_features())
    print("Loading dataset done")

    path_fasttext_model = "/gpfswork/rech/six/urd43gx/data/fasttext_model/lid.176.bin"
    func_dataset_modifying_documents = FunctionDatasetModifyingDocuments(
        path_fasttext_model
    )
    # Could be improved by allowing multiprocessing with map (currently doesn't work)
    dataset = dataset.map(
        func_dataset_modifying_documents, num_proc=1
    )  # num_proc=cpu_count()
    print("Fasttext done")

    path_dir_save_dataset = (
        f"/gpfsscratch/rech/six/urd43gx/crawl/shards/shard_{args.seed_id}"
    )
    pathlib.Path(path_dir_save_dataset).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(path_dir_save_dataset)
    print("Shard successfully saved")


if __name__ == "__main__":
    main()
