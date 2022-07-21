from typing import List

def build_line_with_substring_remover(bad_strings: List[str]):
    def remove_bad_substring(batch):
        return {
            **batch,
            "text": [
                "\n".join([line for line in text.split("\n") if not any([bs in line for bs in bad_strings])])
                for text in batch["text"]
            ]
        }
    return remove_bad_substring
