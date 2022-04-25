from clean_helpers.utils import parse_meta


def filter_wiki_user_titles(examples):
    return [not parse_meta(meta)["title"].startswith("User ") for meta in examples["meta"]]

def filter_wiki_non_text_type(examples):
    return [parse_meta(meta)["type"] == "text" for meta in examples["meta"]]

def filter_remove_empty_docs(examples):
    return [len(text.strip()) > 0 for text in examples["text"]]
