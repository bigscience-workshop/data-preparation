import stanza
from stanza_batch import batch
from indicnlp import common
from indicnlp.tokenize import sentence_tokenize
from nltk.tokenize import sent_tokenize
from underthesea import sent_tokenize as vi_sent_tokenize


def build_nltk_splitter(lang):
    lang_to_punkt = {
        "en": "english",
        "fr": "french",
        "pt": "portuguese",
        "es": "spanish"
    }
    
    def splitter(examples):
        split_texts = ["\n".join(sent_tokenize(text, language=lang_to_punkt[lang])) for text in examples["text"]]
        return {**examples, "text": split_texts }        
    return splitter


def build_vi_splitter(lang):
    def splitter(examples):
        split_texts = ["\n".join(vi_sent_tokenize(text)) for text in examples["text"]]
        return {**examples, "text": split_texts }        
    return splitter


def build_stanza_splitter(lang, batch_size=32):
    lang_to_stanza = {"zht": "zh-hant", "zhs": "zh-hans"}
    lang = lang_to_stanza.get(lang, lang)
    # TODO: @thomasw21 CUDA doesn't work well with multiprocessing
    tokenizer = stanza.Pipeline(lang, logging_level="WARNING", processors='tokenize',
                          use_gpu=False)
    
    def splitter(examples):
        split_texts = []
        for document in batch(examples["text"], tokenizer, batch_size=batch_size):
            split_texts.append("\n".join([sentence.text for sentence in document.sentences]))
        return {**examples, "text": split_texts }        
    return splitter


def build_indic_splitter(lang):
    lang_to_indic = {
        "indic-bn": "bn",
        "indic-gu": "gu",
        "indic-hi": "hi",
        "indic-kn": "kn",
        "indic-ml": "ml",
        "indic-mr": "mr",
        "indic-pa": "pa",
        "indic-ta": "ta",
        "indic-te": "te"
        }
    def splitter(examples):
        split_texts = ["\n".join(sentence_tokenize.sentence_split(text, lang=lang_to_indic[lang])) for text in examples["text"]]
        return {**examples, "text": split_texts }
    return splitter


def build_sentence_splitter(lang):
    stanza_list = {"ar", "ca", "eu", "id", "zhs", "zht"}
    nltk_list = {"en", "fr", "pt", "es"}
    indic_list = {"indic-bn", "indic-gu", "indic-hi", "indic-kn", "indic-ml", "indic-mr", "indic-pa", "indic-ta", "indic-te"}
    vi_list = {"vi"}
    
    assert len(stanza_list & nltk_list) == 0
    assert len(stanza_list & indic_list) == 0
    assert len(stanza_list & vi_list) == 0
    assert len(indic_list & nltk_list) == 0
    assert len(indic_list & vi_list) == 0
    assert len(nltk_list & vi_list) == 0

    if lang in stanza_list:
        return build_stanza_splitter(lang)
    elif lang in nltk_list:
        return build_nltk_splitter(lang)
    elif lang in indic_list:
        return build_indic_splitter(lang)
    elif lang in vi_list:
        return build_vi_splitter(lang)
    else:
        NotImplementedError(f"Lang '{lang}' has no sentence splitter implemented.")


sentence_split_langs = {"ar", "ca", "eu", "id", "vi", "zhs", "zht", "en", "fr", 
                        "pt", "es", "indic-bn", "indic-gu", "indic-hi", "indic-kn",
                        "indic-ml", "indic-mr", "indic-pa", "indic-ta", "indic-te"}