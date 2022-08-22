def build_small_docs_filter(min_word):
    def filter_small_docs(examples):
        """Discard documents with less than min_word words"""
        return [len(text.split(" ")) >= min_word for text in examples["text"]]
    return filter_small_docs

def build_small_docs_bytes_filter(min_bytes):
    # a byte of text usually turns into 0.3 tokens. a 256-token sequence would be ~850 bytes of text. I think anywhere from 500 to 1000 min bytes is reasonable.
    def filter_small_docs(examples):
        """Discard documents with less than min_word words"""
        return [len(text.encode()) >= min_bytes for text in examples["text"]]
    return filter_small_docs
