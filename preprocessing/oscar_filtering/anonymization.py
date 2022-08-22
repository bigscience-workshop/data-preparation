from muliwai.pii_regexes import detect_ner_with_regex_and_context
from muliwai.pii_regexes import regex_rulebase

trannum = str.maketrans("0123456789", "1111111111")


def apply_regex_anonymization(
    sentence: str,
    lang_id: str,
    context_window: int = 20,
    anonymize_condition=None,
    tag_type={"IP_ADDRESS", "KEY", "ID", "PHONE", "USER", "EMAIL", "LICENSE_PLATE"},
) -> str:
    """
    Params:
    ==================
    sentence: str, the sentence to be anonymized
    lang_id: str, the language id of the sentence
    context_window: int, the context window size
    anonymize_condition: function, the anonymization condition
    tag_type: iterable, the tag types of the anonymization. All keys in regex_rulebase is None
    """
    if tag_type == None:
        tag_type = regex_rulebase.keys()
    lang_id = lang_id.split("_")[0]
    ner = detect_ner_with_regex_and_context(
        sentence=sentence,
        src_lang=lang_id,
        context_window=context_window,
        tag_type=tag_type,
    )
    if anonymize_condition:
        for (ent, start, end, tag) in ner:
            # we need to actually walk through and replace by start, end span.
            sentence = sentence.replace(ent, f" <{tag}> ")
    return sentence, ner
