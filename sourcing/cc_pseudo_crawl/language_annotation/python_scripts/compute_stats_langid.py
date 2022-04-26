from datasets import load_from_disk

from multiprocessing import cpu_count

import json

# import random # A DECOMMENTER

dataset = load_from_disk(
    "/gpfsscratch/rech/six/urd43gx/crawl/annotated_langid_crawl"
)  # "/Users/hugolaurencon/Desktop/HF/Code/dataset_filtered/af/"
dataset = dataset["train"]  # A COMMENTER
print("Dataset loaded")

dataset = dataset.map(
    lambda example: {
        "pred_lang": example["fasttext_pred"]["lang_pred_fasttext_id"],
        "len_text": len(example["text"]),
    },  # random.choice(["A", "B", "C"])
    remove_columns=dataset.column_names,
    num_proc=cpu_count(),
)

stats_langid = {}
for i in range(dataset.num_rows):
    pred_lang = dataset[i]["pred_lang"]
    len_text = dataset[i]["len_text"]
    stats_langid[pred_lang] = stats_langid.get(pred_lang, 0) + len_text

f = open(
    "/gpfswork/rech/six/urd43gx/code/filtering_crawl/compute_stats_langid/stats_langid.json",
    "w",
)  # "myfile.json"
json.dump(stats_langid, f)
f.close()
