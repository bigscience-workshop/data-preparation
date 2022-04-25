def replace_newline_with_space(batch):
    return {
        **batch,
        "text": [text.replace("\n", " ") for text in batch["text"]]
    }

def remove_html_spans(batch):
    """Removes lines containing a '<' or '>' from the texts."""
    bad_strings = ["<", ">"]
    return {
        **batch,
        "text": [
            "\n".join([line for line in text.split("\n") if not any([bs in line for bs in bad_strings])])
            for text in batch["text"]
        ]
    }
