from typing import Dict

import numpy as np
import simhash

from . import INTERNAL_HASH, PUNCTUATION_REGEX


def hashing(
    record,
    column: str = "text",
    tokenization: str = "character",
    window_size: int = 4,
    ignore_punctuation: bool = True,
    lowercase: bool = True,
    output: str = INTERNAL_HASH,
) -> Dict[str, int]:
    """Hashing a document with SimHash.

    Parameters
    ----------
    record : [type]
        A dict of feature-value pairs
    column : str, optional
        The column name to use for hashing, by default "text"
    tokenization : str, optional
        Method to use for tokenization, by default "character"
    window_size : int, optional
        The size of the token window, by default 4
    ignore_punctuation : bool, optional
        To ignore punctuation or not, by default True
    lowercase : bool, optional
        To lowercase the text or not, by default True

    Returns
    -------
    Dict[str, int]
        The new hash feature column

    Raises
    ------
    Exception
        Unrecognized tokenization parameter
    """
    document = record[column]
    if lowercase:
        document = document.lower()

    if ignore_punctuation:
        document = PUNCTUATION_REGEX.sub("", document)

    if tokenization == "character":
        tokens = [
            str.encode(document[i : i + window_size])
            for i in range(len(document) - window_size)
        ]
    elif tokenization == "punctuation":
        tokens = PUNCTUATION_REGEX.split(document)
        tokens = [
            str.encode(" ".join(tokens[i : i + window_size]))
            for i in range(len(tokens) - window_size)
        ]
    elif tokenization == "space":
        tokens = document.split(" ")
        tokens = [
            str.encode(" ".join(tokens[i : i + window_size]))
            for i in range(len(tokens) - window_size)
        ]
    else:
        raise Exception(f"Unrecognized tokenization parameter {tokenization}")

    return {output: np.uint64(simhash.compute(map(simhash.unsigned_hash, tokens)))}
