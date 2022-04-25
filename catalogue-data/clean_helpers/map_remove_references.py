import functools
from clean_helpers import stopwords


def enough_stopwords(line, lang_stopwords, ratio):
    return len([word for word in line.split() if word in lang_stopwords]) > len(line.split()) * ratio


def remove_references(batch, language):
    """if there is a continuous of lines without enough stopwords at the end, removes it. doesn't currently support
    languages with different segmentation (e.g. `zh`). Designed for academic datasets."""
    lang_stopwords = stopwords.stopwords[language]
    ratio = stopwords.ratios[language]
    lines_per_example = [text.split("\n") for text in batch["text"]]
    passing_lines_per_example = [[enough_stopwords(line, lang_stopwords, ratio) for line in lines] for lines in lines_per_example]
    cutoffs = [max([0] + [i + 1 for i, passing in enumerate(passing_lines) if passing]) for passing_lines in passing_lines_per_example]
    print(cutoffs)
    filtered_lines_per_example = [lines[:cutoff] for lines, cutoff in zip(lines_per_example, cutoffs)]
    return {
        **batch,
        "text": [
            "\n".join(lines) for lines in filtered_lines_per_example
        ]
    }


def build_reference_remover(language):
    return functools.partial(remove_references, language=language)
