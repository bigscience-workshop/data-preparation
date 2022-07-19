import re

import numpy as np

import fasttext

import sentencepiece
import kenlm

import pathlib

from languages_id import langs_id
from parameters_filtering import parameters_filtering
from normalization import normalization
from stopwords import stopwords
from flagged_words import flagged_words


class LoadParameters:
    @staticmethod
    def load_parameters(lang_dataset_id):
        if lang_dataset_id in parameters_filtering:
            param = parameters_filtering[lang_dataset_id]
        else:
            param = parameters_filtering["default"]
        return param

    @staticmethod
    def load_stopwords(lang_dataset_id):
        stopwords_lang_id = langs_id.loc[
            langs_id["dataset_id"] == lang_dataset_id, "stopwords_id"
        ].iloc[0]
        if stopwords_lang_id:
            stopwords_lang = set(stopwords[stopwords_lang_id])
        else:
            stopwords_lang = None
        return stopwords_lang

    @staticmethod
    def load_flagged_words(lang_dataset_id):
        flagged_words_lang_id = langs_id.loc[
            langs_id["dataset_id"] == lang_dataset_id, "flagged_words_id"
        ].iloc[0]
        if flagged_words_lang_id:
            flagged_words_lang = set(flagged_words[flagged_words_lang_id])
        else:
            flagged_words_lang = None
        return flagged_words_lang

    @staticmethod
    def load_model_lang_id(lang_dataset_id, path_fasttext_model):
        fasttext_lang_id = langs_id.loc[
            langs_id["dataset_id"] == lang_dataset_id, "fasttext_id"
        ].iloc[0]
        if fasttext_lang_id:
            model_lang_id = fasttext.load_model(path_fasttext_model)
        else:
            model_lang_id = None
        return model_lang_id

    @staticmethod
    def load_sentencepiece_model(lang_dataset_id, path_sentencepiece_model):
        sentencepiece_lang_id = langs_id.loc[
            langs_id["dataset_id"] == lang_dataset_id, "sentencepiece_id"
        ].iloc[0]
        if sentencepiece_lang_id:
            sentencepiece_model = sentencepiece.SentencePieceProcessor()
            sentencepiece_model.load(path_sentencepiece_model)
        else:
            sentencepiece_model = None
        return sentencepiece_model

    @staticmethod
    def load_kenlm_model(lang_dataset_id, path_kenlm_model):
        kenlm_lang_id = langs_id.loc[
            langs_id["dataset_id"] == lang_dataset_id, "kenlm_id"
        ].iloc[0]
        if kenlm_lang_id:
            kenlm_model = kenlm.Model(path_kenlm_model)
        else:
            kenlm_model = None
        return kenlm_model


class ModifyingDocuments:
    @staticmethod
    def remove_empty_el_from_list(list_):
        return [el for el in list_ if el]

    @staticmethod
    def remove_non_printing_characters(document, non_printing_characters_re):
        return non_printing_characters_re.sub("", document)

    @staticmethod
    def uniform_whitespace(
        document,
        whitespace=[
            " ",
            " ",
            " ",
            " ",
            " ",
            "　",
            " ",
            " ",
            " ",
            " ",
            "￼",
            "",
        ],
    ):
        """There are different whitespace characters."""
        whitespace = set(whitespace)
        document = "".join(
            [char if char not in whitespace else " " for char in document]
        )
        return document

    @staticmethod
    def replace_digits_with_zeros(document, digits_re):
        return digits_re.sub("0", document)

    @staticmethod
    def replace_unicode_punctuation(document, unicode_punctuation):
        return "".join(unicode_punctuation.get(c, c) for c in document)

    @staticmethod
    def normalization(
        document,
        remove_non_printing_characters,
        strip,
        lower_case,
        uniform_whitespace,
        replace_digits_with_zeros,
        replace_unicode_punctuation,
        non_printing_characters_re=normalization["non_printing_characters_re"],
        digits_re=normalization["digits_re"],
        unicode_punctuation=normalization["unicode_punctuation"],
    ):
        if remove_non_printing_characters:
            document = ModifyingDocuments.remove_non_printing_characters(
                document, non_printing_characters_re
            )
        if strip:
            document = document.strip()
        if not document:
            return document
        if lower_case:
            document = document.lower()
        if uniform_whitespace:
            document = ModifyingDocuments.uniform_whitespace(document)
        if replace_digits_with_zeros:
            document = ModifyingDocuments.replace_digits_with_zeros(document, digits_re)
        if replace_unicode_punctuation:
            document = ModifyingDocuments.replace_unicode_punctuation(
                document, unicode_punctuation
            )
        return document

    @staticmethod
    def tokenization(document, sentencepiece_model, join_on_whitespace):
        document_tokenized = sentencepiece_model.encode_as_pieces(document)
        if join_on_whitespace:
            document_tokenized = " ".join(document_tokenized)
        return document_tokenized

    @staticmethod
    def split_on_whitespace(
        document,
        new_line=False,
        tab=False,
    ):
        """This method also removes concatenated spaces."""
        sep = [" "] + new_line * ["\n"] + tab * ["\t"]
        sep = "|".join(sep)
        split_document = re.split(sep, document)
        split_document = ModifyingDocuments.remove_empty_el_from_list(split_document)
        return split_document

    @staticmethod
    def strip(document, strip_characters):
        """Way faster than document.strip(strip_characters)
        since strip_characters is now a set instead of a str,
        and it contains a lot of elements (all the emojis)."""
        if not document:
            return document
        beg_ind = 0
        end_ind = len(document)
        for i in range(len(document)):
            if document[i] in strip_characters:
                beg_ind += 1
            else:
                break
        for i in range(1, len(document) + 1):
            if document[-i] in strip_characters:
                end_ind -= 1
            else:
                break
        document_stripped = document[beg_ind:end_ind]
        return document_stripped

    @staticmethod
    def get_words_from_document(
        document, sentencepiece_model_tok, lower_case, strip_characters
    ):
        """Get words from a document. Non reversible since the document
        is split on multiple characters, words are stripped of
        special characters and characters are converted to lower case.
        Useful to compute ratios, like the stopwords ratio."""
        if sentencepiece_model_tok:
            document_normalized = ModifyingDocuments.normalization(
                document=document,
                remove_non_printing_characters=True,
                strip=True,
                lower_case=True,
                uniform_whitespace=True,
                replace_digits_with_zeros=True,
                replace_unicode_punctuation=True,
            )
            words = ModifyingDocuments.tokenization(
                document_normalized, sentencepiece_model_tok, join_on_whitespace=False
            )
        else:
            words = ModifyingDocuments.split_on_whitespace(
                document, new_line=True, tab=True
            )
        if lower_case:
            words = [word.lower() for word in words]
        if strip_characters:
            words = [ModifyingDocuments.strip(word, strip_characters) for word in words]
            words = ModifyingDocuments.remove_empty_el_from_list(words)
        return words

    @staticmethod
    def words_augmentation(words, group_size, join_char):
        """Augment words, especially for Chinese (without a space between words)
        and Vietnamese (with a space between syllables)."""
        augmentation = [
            join_char.join(words[i : i + group_size])
            for i in range(len(words) - group_size + 1)
        ]
        return augmentation

    @staticmethod
    def split_on_newline_tab_whitespace(document):
        """First split on "\n", then on "\t", then on " "."""
        sentences = document.split("\n")
        sentences = [sentence.split("\t") for sentence in sentences]
        sentences = [
            [
                ModifyingDocuments.split_on_whitespace(subsentence)
                for subsentence in sentence
            ]
            for sentence in sentences
        ]
        return sentences

    @staticmethod
    def merge_on_whitespace_tab_newline(sentences):
        """Invert the method split_on_newline_tab_whitespace.
        Removes concatenated separators."""
        sentences = [
            [" ".join(subsentence) for subsentence in sentence if subsentence]
            for sentence in sentences
        ]
        sentences = ["\t".join(sentence) for sentence in sentences if sentence]
        if not sentences:
            return ""
        document = "\n".join(sentences)
        return document

    @staticmethod
    def should_keep_word_with_incorrect_substrings(
        word, strip_characters, incorrect_word_substrings
    ):
        word = ModifyingDocuments.strip(word, strip_characters)
        should_keep = all(
            [(i_substr not in word) for i_substr in incorrect_word_substrings]
        )
        return should_keep

    @staticmethod
    def remove_words_with_incorrect_substrings(
        document,
        strip_characters,
        incorrect_word_substrings,
    ):
        sentences = ModifyingDocuments.split_on_newline_tab_whitespace(document)
        sentences = [
            [
                [
                    word
                    for word in subsentence
                    if ModifyingDocuments.should_keep_word_with_incorrect_substrings(
                        word, strip_characters, incorrect_word_substrings
                    )
                ]
                for subsentence in sentence
            ]
            for sentence in sentences
        ]
        document = ModifyingDocuments.merge_on_whitespace_tab_newline(sentences)
        return document

    @staticmethod
    def should_keep_long_word(word, strip_characters, length_word_max_cutoff):
        """If the word is too long but it contains only one
        special character, it might be a concatenation of one word,
        a punctuation, and another word, with no space between them.
        In this case, we give the word a pass."""
        if len(word) <= length_word_max_cutoff:
            return True
        word = ModifyingDocuments.strip(word, strip_characters)
        if not word:  # The word consisted only of strip characters
            return False
        if len(word) <= length_word_max_cutoff:
            return True
        return False

    def remove_long_words(
        document,
        strip_characters,
        length_word_max_cutoff,
    ):
        sentences = ModifyingDocuments.split_on_newline_tab_whitespace(document)
        sentences = [
            [
                [
                    word
                    for word in subsentence
                    if ModifyingDocuments.should_keep_long_word(
                        word,
                        strip_characters,
                        length_word_max_cutoff,
                    )
                ]
                for subsentence in sentence
            ]
            for sentence in sentences
        ]
        document = ModifyingDocuments.merge_on_whitespace_tab_newline(sentences)
        return document

    @staticmethod
    def modifying_documents(
        document,
        cond_uniform_whitespace,
        cond_replace_unicode_punctuation,
        cond_remove_words_with_incorrect_substrings,
        strip_characters,
        incorrect_word_substrings,
        cond_remove_long_words,
        length_word_max_cutoff,
    ):
        document = ModifyingDocuments.normalization(
            document=document,
            remove_non_printing_characters=False,
            strip=True,
            lower_case=False,
            uniform_whitespace=cond_uniform_whitespace,
            replace_digits_with_zeros=False,
            replace_unicode_punctuation=cond_replace_unicode_punctuation,
        )
        if cond_remove_words_with_incorrect_substrings:
            document = ModifyingDocuments.remove_words_with_incorrect_substrings(
                document,
                strip_characters,
                incorrect_word_substrings,
            )
        if cond_remove_long_words:
            document = ModifyingDocuments.remove_long_words(
                document,
                strip_characters,
                length_word_max_cutoff,
            )
        return document


class FunctionDatasetModifyingDocuments:
    def __init__(self, lang_dataset_id):
        self.lang_dataset_id = lang_dataset_id
        self.param = LoadParameters.load_parameters(lang_dataset_id)

    def __call__(self, example):
        example["text"] = ModifyingDocuments.modifying_documents(
            document=example["text"],
            cond_uniform_whitespace=self.param["cond_uniform_whitespace"],
            cond_replace_unicode_punctuation=self.param[
                "cond_replace_unicode_punctuation"
            ],
            cond_remove_words_with_incorrect_substrings=self.param[
                "cond_remove_words_with_incorrect_substrings"
            ],
            strip_characters=self.param["strip_characters"],
            incorrect_word_substrings=self.param["incorrect_word_substrings"],
            cond_remove_long_words=self.param["cond_remove_long_words"],
            length_word_max_cutoff=self.param["length_word_max_cutoff"],
        )
        return example

    def __reduce__(self):
        return (self.__class__, (self.lang_dataset_id,))


class Filtering:
    @staticmethod
    def check_number_words(
        document,
        sentencepiece_model_tok,
        strip_characters,
        number_words_min_cutoff,
        number_words_max_cutoff,
    ):
        words = ModifyingDocuments.get_words_from_document(
            document,
            sentencepiece_model_tok,
            lower_case=False,
            strip_characters=strip_characters,
        )
        cond = (len(words) >= number_words_min_cutoff) and (
            len(words) <= number_words_max_cutoff
        )
        return cond

    @staticmethod
    def compute_character_repetition_ratio(document, character_repetition_length):
        def get_freq_character_ngrams(document, n):
            character_ngrams = [
                document[i : i + n] for i in range(len(document) - n + 1)
            ]
            freq_character_ngrams = {}
            for character_ngram in character_ngrams:
                freq_character_ngrams[character_ngram] = (
                    freq_character_ngrams.get(character_ngram, 0) + 1
                )
            return freq_character_ngrams

        freq_character_ngrams = get_freq_character_ngrams(
            document, character_repetition_length
        )
        if len(freq_character_ngrams) == 0:
            return 0
        freq_character_ngrams = list(freq_character_ngrams.values())
        freq_character_ngrams = sorted(freq_character_ngrams, reverse=True)
        val_one = len([el for el in freq_character_ngrams if el == 1])
        num_rep_character_ngrams = min(
            int(np.sqrt(len(freq_character_ngrams))),
            len(freq_character_ngrams) - val_one,
        )
        character_repetition_ratio = sum(
            freq_character_ngrams[:num_rep_character_ngrams]
        ) / sum(freq_character_ngrams)
        return character_repetition_ratio

    @staticmethod
    def check_character_repetition_removal(
        document,
        character_repetition_length,
        character_repetition_max_cutoff,
    ):
        character_repetition_ratio = Filtering.compute_character_repetition_ratio(
            document, character_repetition_length
        )
        cond = character_repetition_ratio <= character_repetition_max_cutoff
        return cond

    @staticmethod
    def compute_word_repetition_ratio(
        document, sentencepiece_model_tok, strip_characters, word_repetition_length
    ):
        def get_freq_word_ngrams(
            document, sentencepiece_model_tok, strip_characters, n
        ):
            words = ModifyingDocuments.get_words_from_document(
                document,
                sentencepiece_model_tok,
                lower_case=True,
                strip_characters=strip_characters,
            )
            word_ngrams = [
                " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
            ]
            freq_word_ngrams = {}
            for word_ngram in word_ngrams:
                freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1
            return freq_word_ngrams

        freq_word_ngrams = get_freq_word_ngrams(
            document, sentencepiece_model_tok, strip_characters, word_repetition_length
        )
        if len(freq_word_ngrams) == 0:
            return 0
        freq_word_ngrams = list(freq_word_ngrams.values())
        word_repetition_ratio = sum(
            freq for freq in freq_word_ngrams if freq > 1
        ) / sum(freq_word_ngrams)
        return word_repetition_ratio

    @staticmethod
    def check_word_repetition_removal(
        document,
        sentencepiece_model_tok,
        strip_characters,
        word_repetition_length,
        word_repetition_max_cutoff,
    ):
        word_repetition_ratio = Filtering.compute_word_repetition_ratio(
            document, sentencepiece_model_tok, strip_characters, word_repetition_length
        )
        cond = word_repetition_ratio <= word_repetition_max_cutoff
        return cond

    @staticmethod
    def compute_special_characters_ratio(document, special_characters):
        if len(document) == 0:
            return 0
        special_characters_ratio = len(
            [char for char in document if char in special_characters]
        ) / len(document)
        return special_characters_ratio

    @staticmethod
    def check_special_characters(
        document,
        special_characters,
        special_characters_max_cutoff,
    ):
        special_characters_ratio = Filtering.compute_special_characters_ratio(
            document, special_characters
        )
        cond = special_characters_ratio <= special_characters_max_cutoff
        return cond

    @staticmethod
    def compute_stopwords_ratio(
        document,
        sentencepiece_model_tok,
        strip_characters,
        cond_words_augmentation,
        words_augmentation_group_sizes,
        words_augmentation_join_char,
        stopwords,
    ):
        words = ModifyingDocuments.get_words_from_document(
            document,
            sentencepiece_model_tok,
            lower_case=True,
            strip_characters=strip_characters,
        )
        if not words:
            return 0
        augmentation = []
        if cond_words_augmentation:
            augmentation = [
                ModifyingDocuments.words_augmentation(
                    words, group_size, words_augmentation_join_char
                )
                for group_size in words_augmentation_group_sizes
            ]
            augmentation = [word for augm in augmentation for word in augm]
        stopwords_ratio = len(
            [word for word in words + augmentation if word in stopwords]
        ) / len(words)
        if stopwords_ratio > 1.0:
            stopwords_ratio = 1.0
        return stopwords_ratio

    @staticmethod
    def check_stopwords(
        document,
        sentencepiece_model_tok,
        strip_characters,
        cond_words_augmentation,
        words_augmentation_group_sizes,
        words_augmentation_join_char,
        stopwords,
        stopwords_min_cutoff,
    ):
        cond = True
        if stopwords:
            stopwords_ratio = Filtering.compute_stopwords_ratio(
                document,
                sentencepiece_model_tok,
                strip_characters,
                cond_words_augmentation,
                words_augmentation_group_sizes,
                words_augmentation_join_char,
                stopwords,
            )
            cond = stopwords_ratio >= stopwords_min_cutoff
        return cond

    @staticmethod
    def compute_flagged_words_ratio(
        document,
        sentencepiece_model_tok,
        strip_characters,
        cond_words_augmentation,
        words_augmentation_group_sizes,
        words_augmentation_join_char,
        flagged_words,
    ):
        words = ModifyingDocuments.get_words_from_document(
            document,
            sentencepiece_model_tok,
            lower_case=True,
            strip_characters=strip_characters,
        )
        if not words:
            return 0
        augmentation = []
        if cond_words_augmentation:
            augmentation = [
                ModifyingDocuments.words_augmentation(
                    words, group_size, words_augmentation_join_char
                )
                for group_size in words_augmentation_group_sizes
            ]
            augmentation = [word for augm in augmentation for word in augm]
        flagged_words_ratio = len(
            [word for word in words + augmentation if word in flagged_words]
        ) / len(words)
        if flagged_words_ratio > 1.0:
            flagged_words_ratio = 1.0
        return flagged_words_ratio

    @staticmethod
    def check_flagged_words(
        document,
        sentencepiece_model_tok,
        strip_characters,
        cond_words_augmentation,
        words_augmentation_group_sizes,
        words_augmentation_join_char,
        flagged_words,
        flagged_words_max_cutoff,
    ):
        cond = True
        if flagged_words:
            flagged_words_ratio = Filtering.compute_flagged_words_ratio(
                document,
                sentencepiece_model_tok,
                strip_characters,
                cond_words_augmentation,
                words_augmentation_group_sizes,
                words_augmentation_join_char,
                flagged_words,
            )
            cond = flagged_words_ratio <= flagged_words_max_cutoff
        return cond

    @staticmethod
    def compute_lang_id_pred_score(document, model_lang_id):
        document = document.lower().replace("\n", " ")
        pred = model_lang_id.predict(document)
        lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
        score_pred = pred[1][0]
        lang_pred_dataset_id = langs_id.loc[
            langs_id["fasttext_id"] == lang_pred_fasttext_id, "dataset_id"
        ]
        if len(lang_pred_dataset_id) > 0:
            lang_pred_dataset_id = lang_pred_dataset_id.iloc[0]
        else:
            lang_pred_dataset_id = "unknown"
        return lang_pred_dataset_id, score_pred

    @staticmethod
    def check_lang_id(
        document,
        lang_dataset_id,
        model_lang_id,
        lang_id_min_cutoff,
    ):
        cond = True
        if model_lang_id:
            lang_pred_dataset_id, score_pred = Filtering.compute_lang_id_pred_score(
                document, model_lang_id
            )
            cond = (lang_pred_dataset_id == lang_dataset_id) and (
                score_pred >= lang_id_min_cutoff
            )
        return cond

    @staticmethod
    def compute_perplexity_score(document, sentencepiece_model, kenlm_model):
        document = ModifyingDocuments.normalization(
            document=document,
            remove_non_printing_characters=True,
            strip=True,
            lower_case=False,
            uniform_whitespace=True,
            replace_digits_with_zeros=True,
            replace_unicode_punctuation=True,
        )
        document = ModifyingDocuments.tokenization(
            document, sentencepiece_model, join_on_whitespace=True
        )
        doc_log_score, doc_length = 0, 0
        for line in document.split("\n"):
            log_score = kenlm_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        pp_score = 10.0 ** (-doc_log_score / doc_length)
        pp_score = round(pp_score, 1)
        return pp_score

    @staticmethod
    def check_perplexity(
        document,
        sentencepiece_model,
        kenlm_model,
        perplexity_max_cutoff,
    ):
        cond = True
        if kenlm_model:
            score = Filtering.compute_perplexity_score(
                document, sentencepiece_model, kenlm_model
            )
            cond = score <= perplexity_max_cutoff
        return cond

    @staticmethod
    def filtering(
        document,
        cond_check_number_words,
        sentencepiece_model_tok,
        strip_characters,
        number_words_min_cutoff,
        number_words_max_cutoff,
        cond_check_character_repetition_removal,
        character_repetition_length,
        character_repetition_max_cutoff,
        cond_check_word_repetition_removal,
        word_repetition_length,
        word_repetition_max_cutoff,
        cond_check_special_characters,
        special_characters,
        special_characters_max_cutoff,
        cond_words_augmentation,
        words_augmentation_group_sizes,
        words_augmentation_join_char,
        cond_check_stopwords,
        stopwords,
        stopwords_min_cutoff,
        cond_check_flagged_words,
        flagged_words,
        flagged_words_max_cutoff,
        cond_check_lang_id,
        lang_dataset_id,
        model_lang_id,
        lang_id_min_cutoff,
        cond_check_perplexity,
        sentencepiece_model,
        kenlm_model,
        perplexity_max_cutoff,
    ):
        if cond_check_number_words:
            if not Filtering.check_number_words(
                document,
                sentencepiece_model_tok,
                strip_characters,
                number_words_min_cutoff,
                number_words_max_cutoff,
            ):
                return False
        if cond_check_character_repetition_removal:
            if not Filtering.check_character_repetition_removal(
                document,
                character_repetition_length,
                character_repetition_max_cutoff,
            ):
                return False
        if cond_check_word_repetition_removal:
            if not Filtering.check_word_repetition_removal(
                document,
                sentencepiece_model_tok,
                strip_characters,
                word_repetition_length,
                word_repetition_max_cutoff,
            ):
                return False
        if cond_check_special_characters:
            if not Filtering.check_special_characters(
                document,
                special_characters,
                special_characters_max_cutoff,
            ):
                return False
        if cond_check_stopwords:
            if not Filtering.check_stopwords(
                document,
                sentencepiece_model_tok,
                strip_characters,
                cond_words_augmentation,
                words_augmentation_group_sizes,
                words_augmentation_join_char,
                stopwords,
                stopwords_min_cutoff,
            ):
                return False
        if cond_check_flagged_words:
            if not Filtering.check_flagged_words(
                document,
                sentencepiece_model_tok,
                strip_characters,
                cond_words_augmentation,
                words_augmentation_group_sizes,
                words_augmentation_join_char,
                flagged_words,
                flagged_words_max_cutoff,
            ):
                return False
        if cond_check_lang_id:
            if not Filtering.check_lang_id(
                document,
                lang_dataset_id,
                model_lang_id,
                lang_id_min_cutoff,
            ):
                return False
        if cond_check_perplexity:
            if not Filtering.check_perplexity(
                document,
                sentencepiece_model,
                kenlm_model,
                perplexity_max_cutoff,
            ):
                return False
        return True


class FunctionDatasetFiltering:
    def __init__(
        self,
        lang_dataset_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
    ):
        self.lang_dataset_id = lang_dataset_id
        self.path_fasttext_model = path_fasttext_model
        self.path_sentencepiece_model = path_sentencepiece_model
        self.path_kenlm_model = path_kenlm_model

        self.param = LoadParameters.load_parameters(lang_dataset_id)
        self.stopwords = LoadParameters.load_stopwords(lang_dataset_id)
        self.flagged_words = LoadParameters.load_flagged_words(lang_dataset_id)
        self.model_lang_id = LoadParameters.load_model_lang_id(
            lang_dataset_id, path_fasttext_model
        )
        self.sentencepiece_model = LoadParameters.load_sentencepiece_model(
            lang_dataset_id, path_sentencepiece_model
        )
        self.sentencepiece_model_tok = (
            self.sentencepiece_model if self.param["tokenization"] else None
        )
        self.kenlm_model = LoadParameters.load_kenlm_model(
            lang_dataset_id, path_kenlm_model
        )

    def __call__(self, example):
        keep_example = Filtering.filtering(
            document=example["text"],
            cond_check_number_words=self.param["cond_check_number_words"],
            sentencepiece_model_tok=self.sentencepiece_model_tok,
            strip_characters=self.param["strip_characters"],
            number_words_min_cutoff=self.param["number_words_min_cutoff"],
            number_words_max_cutoff=self.param["number_words_max_cutoff"],
            cond_check_character_repetition_removal=self.param[
                "cond_check_character_repetition_removal"
            ],
            character_repetition_length=self.param["character_repetition_length"],
            character_repetition_max_cutoff=self.param[
                "character_repetition_max_cutoff"
            ],
            cond_check_word_repetition_removal=self.param[
                "cond_check_word_repetition_removal"
            ],
            word_repetition_length=self.param["word_repetition_length"],
            word_repetition_max_cutoff=self.param["word_repetition_max_cutoff"],
            cond_check_special_characters=self.param["cond_check_special_characters"],
            special_characters=self.param["special_characters"],
            special_characters_max_cutoff=self.param["special_characters_max_cutoff"],
            cond_words_augmentation=self.param["cond_words_augmentation"],
            words_augmentation_group_sizes=self.param["words_augmentation_group_sizes"],
            words_augmentation_join_char=self.param["words_augmentation_join_char"],
            cond_check_stopwords=self.param["cond_check_stopwords"],
            stopwords=self.stopwords,
            stopwords_min_cutoff=self.param["stopwords_min_cutoff"],
            cond_check_flagged_words=self.param["cond_check_flagged_words"],
            flagged_words=self.flagged_words,
            flagged_words_max_cutoff=self.param["flagged_words_max_cutoff"],
            cond_check_lang_id=self.param["cond_check_lang_id"],
            lang_dataset_id=self.lang_dataset_id,
            model_lang_id=self.model_lang_id,
            lang_id_min_cutoff=self.param["lang_id_min_cutoff"],
            cond_check_perplexity=self.param["cond_check_perplexity"],
            sentencepiece_model=self.sentencepiece_model,
            kenlm_model=self.kenlm_model,
            perplexity_max_cutoff=self.param["perplexity_max_cutoff"],
        )
        return keep_example

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.lang_dataset_id,
                self.path_fasttext_model,
                self.path_sentencepiece_model,
                self.path_kenlm_model,
            ),
        )


class DatasetFiltering:
    def __init__(
        self,
        dataset,
        lang_dataset_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
        num_proc,
        path_dir_save_dataset,
    ):
        self.ds = dataset
        self.lang_dataset_id = lang_dataset_id
        self.path_fasttext_model = path_fasttext_model
        self.path_sentencepiece_model = path_sentencepiece_model
        self.path_kenlm_model = path_kenlm_model
        self.num_proc = num_proc
        self.path_dir_save_dataset = path_dir_save_dataset

    def modifying_documents(self):
        func_dataset_modifying_documents = FunctionDatasetModifyingDocuments(
            self.lang_dataset_id
        )
        self.ds = self.ds.map(func_dataset_modifying_documents, num_proc=self.num_proc)

    def filtering(self):
        func_dataset_filtering = FunctionDatasetFiltering(
            self.lang_dataset_id,
            self.path_fasttext_model,
            self.path_sentencepiece_model,
            self.path_kenlm_model,
        )
        self.ds = self.ds.filter(func_dataset_filtering, num_proc=self.num_proc)

    def save_dataset(self):
        pathlib.Path(self.path_dir_save_dataset).mkdir(parents=True, exist_ok=True)
        path_dir_save_dataset = pathlib.PurePath(
            self.path_dir_save_dataset, self.lang_dataset_id
        )
        pathlib.Path(path_dir_save_dataset).mkdir(parents=True, exist_ok=True)
        self.ds.save_to_disk(path_dir_save_dataset)
