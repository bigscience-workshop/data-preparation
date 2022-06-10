# Run with: streamlit run visualization.py

import streamlit as st

import os

from io import StringIO
import base64
import json
import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np

import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], ".."))
# Append the path of the parent directory to the python path
# to find the files filtering.py and languages_id.py in the parent directory
sys.path.append(str(Path(sys.path[0]).parent.absolute().parent.absolute()))

from filtering import LoadParameters, ModifyingDocuments, Filtering
from languages_id import langs_id


class Visualization_for_lang:
    def __init__(
        self,
        path_data,
        lang,
        num_docs,
        num_docs_for_words,
        max_len_text_display,
        lang_dataset_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
    ):
        self.path_data = path_data
        self.lang = lang
        self.num_docs = num_docs
        self.num_docs_for_words = num_docs_for_words
        self.max_len_text_display = max_len_text_display

        self.lang_dataset_id = lang_dataset_id
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

    def set_title(self):
        st.title(f"Filtering visualization for {self.lang}")

    def open_data(self):
        with open(self.path_data) as json_file:
            data = json.load(json_file)

        self.num_docs = min(self.num_docs, len(data))
        self.num_docs_for_words = min(self.num_docs_for_words, len(data))

        if "words" in data[0]:
            words = [doc["words"] for doc in data[: self.num_docs_for_words]]
            words = [word for doc in words for word in doc]
            self.words = pd.DataFrame(words)
        else:
            self.words = None

        docs = data[: self.num_docs]
        for doc in docs:
            if not (self.words is None):
                del doc["words"]
            if len(doc["text"]) > self.max_len_text_display:
                doc["text"] = (
                    doc["text"][: self.max_len_text_display]
                    + " [...] [THIS LONG TEXT HAS BEEN TRUNCATED FOR DISPLAY REASONS]"
                )
        self.docs_checkpoint = pd.DataFrame(docs)
        self.docs = self.docs_checkpoint

    @staticmethod
    def print_discarded_by_cond(cond):
        st.caption(
            f"{(len(cond) - np.sum(1*cond)) / len(cond) * 100:.2f}% of the total is discarded with this filter."
        )

    @staticmethod
    def plot_hist(dataframe, key, num_bins=50):
        checkbox = st.checkbox(
            "Diplay distribution", value=True, key=f"display_distribution_{key[0]}"
        )
        if checkbox:
            fig, ax = plt.subplots()
            val = dataframe[key[0]].values
            if np.median(val) != 0:
                val = val[
                    abs(val - np.median(val))
                    < 9 * np.median(np.absolute(val - np.median(val)))
                ]
            ax.hist(val, bins=num_bins, density=True)
            ax.set_title(" ".join(key[0].split("_")))
            ax.axvline(x=key[1], color="r", linestyle="dashed")
            st.pyplot(fig)

    @staticmethod
    def display_dataset(dataframe, cond, description, type_of_examples):
        displayed_examples = dataframe.loc[cond]
        st.subheader(
            f"{description}: {len(displayed_examples)} {type_of_examples} ({len(displayed_examples) / len(dataframe.index) * 100:.2f}%)"
        )
        st.markdown(
            "Click on a column to sort by it, place the cursor on the text to display it."
        )
        st.dataframe(displayed_examples)

    def filtering_of_docs(self):
        def set_sliders():
            columns = list(self.docs)
            keys = []
            conds = {}

            def get_cond(key, cutoff, max_cutoff):
                if max_cutoff:
                    return self.docs[key] <= cutoff
                return self.docs[key] >= cutoff

            if "number_words" in columns:
                with st.sidebar.expander("Number of words"):
                    cutoff_def = "If the number of words of a document is lower than this number, the document is removed."
                    max_nb_words = int(np.max(self.docs["number_words"])) + 1
                    cutoff_min_number_words = st.slider(
                        cutoff_def, 0, min(max_nb_words, 500), 0
                    )
                    new_key = ("number_words", cutoff_min_number_words, False)
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond_1 = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond_1)

                    cutoff_def = "If the number of words of a document is higher than this number, the document is removed."
                    cutoff_max_number_words = st.slider(
                        cutoff_def, 0, max_nb_words, max_nb_words
                    )
                    new_key = ("number_words", cutoff_max_number_words, True)
                    keys.append(new_key)
                    cond_2 = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond_2)

                    conds["number_words"] = [cond_1, cond_2]

            if "character_repetition_ratio" in columns:
                with st.sidebar.expander("Character repetition ratio"):
                    val_repetitions_lengths = list(
                        self.docs["character_repetition_ratio"].iloc[0].keys()
                    )
                    default_index = (
                        val_repetitions_lengths.index("10")
                        if "10" in val_repetitions_lengths
                        else 0
                    )
                    label_selectbox = "Length of repetitions in characters (that will influence the character repetition ratio)."
                    repetitions_length = st.selectbox(
                        label=label_selectbox,
                        options=val_repetitions_lengths,
                        index=default_index,
                    )
                    st.caption(
                        "Choosing a higher or lower number does not mean that the filtering "
                        "is stronger or weaker. Be careful, choosing a low number (below 5 for languages like English) "
                        "tends to associate a high character repetition ratio to very long documents (like book chapters), but with "
                        "few or no repetitions, simply because their length gives them more diversity, and we do "
                        "not want to discard such documents. It is generally better to increase this number, so that false "
                        "positives are very short documents (which we want to delete anyway) rather than long ones. However, "
                        "a low number can be useful for Chinese, where a character can designate a whole word."
                    )
                    self.docs["character_repetition_ratio"] = self.docs_checkpoint[
                        "character_repetition_ratio"
                    ]
                    for i in range(len(self.docs["character_repetition_ratio"])):
                        self.docs["character_repetition_ratio"].iloc[i] = self.docs[
                            "character_repetition_ratio"
                        ].iloc[i][repetitions_length]

                    cutoff_def = "If the character repetition ratio of a document is higher than this number, the document is removed."
                    cutoff_character_repetition_ratio = st.slider(
                        cutoff_def, 0.0, 1.0, 1.0, step=0.01
                    )
                    new_key = (
                        "character_repetition_ratio",
                        cutoff_character_repetition_ratio,
                        True,
                        repetitions_length,
                    )
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["character_repetition_ratio"] = [cond]

            if "word_repetition_ratio" in columns:
                with st.sidebar.expander("Word repetition ratio"):
                    val_repetitions_lengths = list(
                        self.docs["word_repetition_ratio"].iloc[0].keys()
                    )
                    default_index = (
                        val_repetitions_lengths.index("5")
                        if "5" in val_repetitions_lengths
                        else 0
                    )
                    label_selectbox = "Length of repetitions in words (that will influence the word repetition ratio)."
                    repetitions_length = st.selectbox(
                        label=label_selectbox,
                        options=val_repetitions_lengths,
                        index=default_index,
                    )
                    st.caption(
                        "Choosing a higher or lower number does not mean that the filtering "
                        "is stronger or weaker. Be careful, choosing a low number (like 3) could "
                        "tend to associate a high word repetition ratio to very long documents (like book chapters), but with "
                        "few or no repetitions, simply because their length gives them more diversity, and we do "
                        "not want to discard such documents. It is generally better to increase a bit this number, so that false "
                        "positives are very short documents (which we want to delete anyway) rather than long ones."
                    )
                    self.docs["word_repetition_ratio"] = self.docs_checkpoint[
                        "word_repetition_ratio"
                    ]
                    for i in range(len(self.docs["word_repetition_ratio"])):
                        self.docs["word_repetition_ratio"].iloc[i] = self.docs[
                            "word_repetition_ratio"
                        ].iloc[i][repetitions_length]

                    cutoff_def = "If the word repetition ratio of a document is higher than this number, the document is removed."
                    cutoff_word_repetition_ratio = st.slider(
                        cutoff_def, 0.0, 1.0, 1.0, step=0.01
                    )
                    new_key = (
                        "word_repetition_ratio",
                        cutoff_word_repetition_ratio,
                        True,
                        repetitions_length,
                    )
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["word_repetition_ratio"] = [cond]

            if "special_characters_ratio" in columns:
                with st.sidebar.expander("Special characters ratio"):
                    cutoff_def = "If the special characters ratio of a document is higher than this number, the document is removed."
                    cutoff_special_characters_ratio = st.slider(
                        cutoff_def, 0.0, 1.0, 1.0, step=0.01
                    )
                    new_key = (
                        "special_characters_ratio",
                        cutoff_special_characters_ratio,
                        True,
                    )
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["special_characters_ratio"] = [cond]

            if "stopwords_ratio" in columns:
                with st.sidebar.expander("Stop words ratio"):
                    stopwords_file = st.file_uploader(
                        "Upload your own list of stop words (one per line). If there is none, the default one is used."
                    )
                    if stopwords_file:
                        new_stopwords = StringIO(
                            stopwords_file.getvalue().decode("utf-8")
                        ).read()
                        new_stopwords = set(new_stopwords.split("\n"))
                        self.docs["stopwords_ratio"] = self.docs_checkpoint[
                            "stopwords_ratio"
                        ]
                        for i in range(len(self.docs["stopwords_ratio"])):
                            self.docs["stopwords_ratio"].iloc[
                                i
                            ] = Filtering.compute_stopwords_ratio(
                                self.docs["text"].iloc[i],
                                self.sentencepiece_model_tok,
                                self.param["strip_characters"],
                                self.param["cond_words_augmentation"],
                                self.param["words_augmentation_group_sizes"],
                                self.param["words_augmentation_join_char"],
                                new_stopwords,
                            )
                    cutoff_def = "If the stop words ratio of a document is lower than this number, the document is removed."
                    cutoff_stopwords_ratio = st.slider(
                        cutoff_def, 0.0, 1.0, 0.0, step=0.01
                    )
                    new_key = ("stopwords_ratio", cutoff_stopwords_ratio, False)
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["stopwords_ratio"] = [cond]

            if "flagged_words_ratio" in columns:
                with st.sidebar.expander("Flagged words ratio"):
                    flagged_words_file = st.file_uploader(
                        "Upload your own list of flagged words (one per line). If there is none, the default one is used."
                    )
                    if flagged_words_file:
                        new_flagged_words = StringIO(
                            flagged_words_file.getvalue().decode("utf-8")
                        ).read()
                        new_flagged_words = set(new_flagged_words.split("\n"))
                        self.docs["flagged_words_ratio"] = self.docs_checkpoint[
                            "flagged_words_ratio"
                        ]
                        for i in range(len(self.docs["flagged_words_ratio"])):
                            self.docs["flagged_words_ratio"].iloc[
                                i
                            ] = Filtering.compute_flagged_words_ratio(
                                self.docs["text"].iloc[i],
                                self.sentencepiece_model_tok,
                                self.param["strip_characters"],
                                self.param["cond_words_augmentation"],
                                self.param["words_augmentation_group_sizes"],
                                self.param["words_augmentation_join_char"],
                                new_flagged_words,
                            )
                    cutoff_def = "If the flagged words ratio of a document is higher than this number, the document is removed."
                    max_fwr = np.max(self.docs["flagged_words_ratio"])
                    max_fwr = np.ceil(max_fwr * 1000) / 1000
                    max_fwr = float(max_fwr)
                    cutoff_flagged_words_ratio = st.slider(
                        cutoff_def,
                        0.000,
                        max_fwr,
                        max_fwr,
                        step=0.001,
                        format="%f",
                    )
                    new_key = ("flagged_words_ratio", cutoff_flagged_words_ratio, True)
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["flagged_words_ratio"] = [cond]

            if "lang_id_score" in columns:
                with st.sidebar.expander("Language ID confidence score"):
                    cutoff_def = "If the confidence score for the language identification prediction of a document is lower than this number, the document is removed."
                    cutoff_lang_id_score = st.slider(
                        cutoff_def, 0.0, 1.0, 0.0, step=0.01
                    )
                    new_key = ("lang_id_score", cutoff_lang_id_score, False)
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["lang_id_score"] = [cond]

            if "perplexity_score" in columns:
                with st.sidebar.expander("Perplexity score"):
                    cutoff_def = "If the perplexity score of a document is higher than this number, the document is removed."
                    max_pp = int(np.max(self.docs["perplexity_score"])) + 1
                    cutoff_perplexity_score = st.slider(cutoff_def, 0, max_pp, max_pp)
                    new_key = ("perplexity_score", cutoff_perplexity_score, True)
                    keys.append(new_key)
                    Visualization_for_lang.plot_hist(self.docs, new_key)
                    cond = get_cond(new_key[0], new_key[1], new_key[2])
                    Visualization_for_lang.print_discarded_by_cond(cond)
                    conds["perplexity_score"] = [cond]

            return keys, conds

        with st.expander(
            f"Filtering on documents, for {self.num_docs} {self.lang} documents"
        ):
            st.header(
                f"Filtering on documents, for {self.num_docs} {self.lang} documents"
            )

            if "labels" in list(self.docs):
                chosen_label = st.selectbox(
                    label="Consider only documents that include the following label",
                    options=[
                        "All",
                        "NA: Narrative",
                        "IN: Informational Description",
                        "OP: Opinion",
                        "ID: Interactive Discussion",
                        "HI: How-to/Instruction",
                        "IP: Informational Persuasion",
                        "LY: Lyrical",
                        "SP: Spoken",
                    ],
                )
                chosen_label = chosen_label.split(":")[0]
                if chosen_label != "All":
                    cond_label = list(
                        self.docs["labels"].apply(
                            lambda x: True if chosen_label in x else False
                        )
                    )
                    self.docs = self.docs[cond_label]

            if self.docs.empty:
                st.markdown(
                    "No document to display, please try to select a different label."
                )
                self.keys = []
                self.parameters = []

            else:
                st.sidebar.subheader("Parameters of the filtering on documents")
                self.keys, conds = set_sliders()
                self.parameters = self.keys * 1

                all_conds = [
                    subcond for cond in list(conds.values()) for subcond in cond
                ]
                all_conds = np.all(all_conds, axis=0)

                Visualization_for_lang.display_dataset(
                    self.docs, np.invert(all_conds), "Discarded documents", "docs"
                )

                # st.subheader("Display discarded documents by filter")
                display_discarded_documents_by_filter = st.checkbox(
                    "Display discarded documents by filter"
                )

                if display_discarded_documents_by_filter:
                    columns = list(self.docs)

                    if "number_words" in columns:
                        cond_filter = np.invert(np.all(conds["number_words"], axis=0))
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the number of words",
                            "docs",
                        )

                    if "character_repetition_ratio" in columns:
                        cond_filter = np.invert(
                            np.all(conds["character_repetition_ratio"], axis=0)
                        )
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the character repetition ratio",
                            "docs",
                        )

                    if "word_repetition_ratio" in columns:
                        cond_filter = np.invert(
                            np.all(conds["word_repetition_ratio"], axis=0)
                        )
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the word repetition ratio",
                            "docs",
                        )

                    if "special_characters_ratio" in columns:
                        cond_filter = np.invert(
                            np.all(conds["special_characters_ratio"], axis=0)
                        )
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the special characters ratio",
                            "docs",
                        )

                    if "stopwords_ratio" in columns:
                        cond_filter = np.invert(
                            np.all(conds["stopwords_ratio"], axis=0)
                        )
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the stop words ratio",
                            "docs",
                        )

                    if "flagged_words_ratio" in columns:
                        cond_filter = np.invert(
                            np.all(conds["flagged_words_ratio"], axis=0)
                        )
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the flagged words ratio",
                            "docs",
                        )

                    if "lang_id_score" in columns:
                        cond_filter = np.invert(np.all(conds["lang_id_score"], axis=0))
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the language identification confidence score",
                            "docs",
                        )

                    if "perplexity_score" in columns:
                        cond_filter = np.invert(
                            np.all(conds["perplexity_score"], axis=0)
                        )
                        Visualization_for_lang.display_dataset(
                            self.docs,
                            cond_filter,
                            "Discarded documents for the filter on the perplexity score",
                            "docs",
                        )

                Visualization_for_lang.display_dataset(
                    self.docs, all_conds, "Retained documents", "docs"
                )

            st.header("Download data")

            with open(self.path_data) as json_file:
                btn = st.download_button(
                    label="Download data as json",
                    data=json_file,
                    file_name="data.json",
                )

    def filtering_of_words(self):
        if not (self.words is None):
            columns = list(self.words)

            st.sidebar.subheader("Parameter of the filtering on words")

            conds_words = {}

            if "len_word" in columns:
                with st.sidebar.expander("Length of words"):
                    cutoff_def = "If the length of a word is higher than this number, the word is removed."
                    max_len_word = min(int(np.max(self.words["len_word"])) + 1, 200)
                    cutoff_word = st.slider(cutoff_def, 0, max_len_word, max_len_word)
                    new_key = ("len_word", cutoff_word, True)
                    self.parameters.append(new_key)
                    Visualization_for_lang.plot_hist(self.words, new_key)
                    cond_len_words = self.words["len_word"] <= cutoff_word
                    Visualization_for_lang.print_discarded_by_cond(cond_len_words)
                    conds_words["len_word"] = cond_len_words

            if "incorrect_substrings" in columns:
                with st.sidebar.expander("Words with incorrect substrings"):
                    incorrect_substrings = st.checkbox(
                        "Remove words with incorrect substrings."
                    )
                    self.parameters.append(
                        ("incorrect_substrings", incorrect_substrings)
                    )

                    checkbox = st.checkbox(
                        "Diplay distribution",
                        value=True,
                        key="display_distribution_incorrect_substrings",
                    )
                    if checkbox:
                        incor_sub = np.array(self.words["incorrect_substrings"]) * 1
                        with_incor_sub = np.sum(incor_sub)
                        without_incor_sub = len(incor_sub) - with_incor_sub
                        st.markdown(
                            f"Number of words with incorrect substrings: {with_incor_sub}"
                        )
                        st.markdown(
                            f"Number of words without incorrect substrings: {without_incor_sub}"
                        )

                    if incorrect_substrings:
                        cond_incorrect_substrings = np.invert(
                            self.words["incorrect_substrings"]
                        )
                    else:
                        cond_incorrect_substrings = np.array(
                            [
                                True
                                for i in range(len(self.words["incorrect_substrings"]))
                            ]
                        )
                    Visualization_for_lang.print_discarded_by_cond(
                        cond_incorrect_substrings
                    )
                    conds_words["incorrect_substrings"] = cond_incorrect_substrings

            all_conds_words = np.all(list(conds_words.values()), axis=0)

            with st.expander(
                f"Filtering on words, for {self.num_docs_for_words} {self.lang} documents"
            ):
                st.header(
                    f"Filtering on words, for {self.num_docs_for_words} {self.lang} documents"
                )

                st.markdown(
                    f"Since the number of words is way larger than the number of documents, "
                    f"we consider in this section words for only {self.num_docs_for_words} documents."
                )

                Visualization_for_lang.display_dataset(
                    self.words, np.invert(all_conds_words), "Discarded words", "words"
                )

                # st.subheader("Display discarded words by filter")
                display_discarded_words_by_filter = st.checkbox(
                    "Display discarded words by filter"
                )

                if display_discarded_words_by_filter:

                    if "len_word" in columns:
                        cond_filter = np.invert(conds_words["len_word"])
                        Visualization_for_lang.display_dataset(
                            self.words,
                            cond_filter,
                            "Discarded words for the filter on length",
                            "words",
                        )

                    if "incorrect_substrings" in columns:
                        cond_filter = np.invert(conds_words["incorrect_substrings"])
                        Visualization_for_lang.display_dataset(
                            self.words,
                            cond_filter,
                            "Discarded words for the filter on incorrect substrings",
                            "words",
                        )

                Visualization_for_lang.display_dataset(
                    self.words, all_conds_words, "Retained words", "words"
                )

    def download_parameters(self):
        st.sidebar.subheader("Download parameters")
        btn = st.sidebar.download_button(
            label="Download current parameters as json",
            data=json.dumps(self.parameters),
            file_name=f"parameters_{self.lang_dataset_id}.json",
        )

    """
    def plot_zipf_law(self):
        if not (self.words is None):
            st.header("Zipf's Law")

            display_zipf_law = st.checkbox("Display Zipf's Law")

            if display_zipf_law:

                freq_words = {}
                for _, row in self.words.iterrows():
                    freq_words[row["word"]] = freq_words.get(row["word"], 0) + 1
                freq_words = np.array(list(freq_words.values()))
                freq_words = -np.sort(-freq_words)

                fig, ax = plt.subplots()
                ax.loglog(freq_words)
                ax.set_title("Zipf's Law")
                ax.set_xlabel("$i$-th most frequent word")
                ax.set_ylabel("frequency in the documents")
                st.pyplot(fig)
    """

    def analyse_personal_doc(self):
        with st.expander("Analyse your own document"):
            st.header("Analyse your own document")

            personal_doc = st.text_area(
                label="Paste here the document you want to analyse",
                value="",
                max_chars=10000,
            )

            is_discarded = False

            def is_doc_discarded(key, score):
                if key[2]:  # max cutoff
                    return score > key[1]
                else:
                    return score < key[1]

            if personal_doc:

                st.markdown("Statistics of the document:")

                for key in self.keys:
                    if key[0] == "number_words":
                        words = ModifyingDocuments.get_words_from_document(
                            personal_doc,
                            self.sentencepiece_model_tok,
                            lower_case=False,
                            strip_characters=self.param["strip_characters"],
                        )
                        if key[2]:
                            st.markdown(f"Number of words: {len(words)}")
                        if is_doc_discarded(key, len(words)):
                            is_discarded = True

                    elif key[0] == "character_repetition_ratio":
                        character_repetition_ratio = (
                            Filtering.compute_character_repetition_ratio(
                                personal_doc, int(key[3])
                            )
                        )
                        character_repetition_ratio = round(
                            character_repetition_ratio, 3
                        )
                        st.markdown(
                            f"Character repetition ratio: {character_repetition_ratio}"
                        )
                        if is_doc_discarded(key, character_repetition_ratio):
                            is_discarded = True

                    elif key[0] == "word_repetition_ratio":
                        word_repetition_ratio = Filtering.compute_word_repetition_ratio(
                            personal_doc,
                            self.sentencepiece_model_tok,
                            self.param["strip_characters"],
                            int(key[3]),
                        )
                        word_repetition_ratio = round(word_repetition_ratio, 3)
                        st.markdown(f"Word repetition ratio: {word_repetition_ratio}")
                        if is_doc_discarded(key, word_repetition_ratio):
                            is_discarded = True

                    elif key[0] == "special_characters_ratio":
                        special_characters_ratio = (
                            Filtering.compute_special_characters_ratio(
                                personal_doc, self.param["special_characters"]
                            )
                        )
                        special_characters_ratio = round(special_characters_ratio, 3)
                        st.markdown(
                            f"Special characters ratio: {special_characters_ratio}"
                        )
                        if is_doc_discarded(key, special_characters_ratio):
                            is_discarded = True

                    elif key[0] == "stopwords_ratio":
                        stopwords_ratio = Filtering.compute_stopwords_ratio(
                            personal_doc,
                            self.sentencepiece_model_tok,
                            self.param["strip_characters"],
                            self.param["cond_words_augmentation"],
                            self.param["words_augmentation_group_sizes"],
                            self.param["words_augmentation_join_char"],
                            self.stopwords,
                        )
                        stopwords_ratio = round(stopwords_ratio, 3)
                        st.markdown(f"Stop words ratio: {stopwords_ratio}")
                        if is_doc_discarded(key, stopwords_ratio):
                            is_discarded = True

                    elif key[0] == "flagged_words_ratio":
                        flagged_words_ratio = Filtering.compute_flagged_words_ratio(
                            personal_doc,
                            self.sentencepiece_model_tok,
                            self.param["strip_characters"],
                            self.param["cond_words_augmentation"],
                            self.param["words_augmentation_group_sizes"],
                            self.param["words_augmentation_join_char"],
                            self.flagged_words,
                        )
                        flagged_words_ratio = round(flagged_words_ratio, 3)
                        st.markdown(f"Flagged words ratio: {flagged_words_ratio}")
                        if is_doc_discarded(key, flagged_words_ratio):
                            is_discarded = True

                    elif key[0] == "lang_id_score":
                        (
                            lang_pred_dataset_id,
                            lang_id_score,
                        ) = Filtering.compute_lang_id_pred_score(
                            personal_doc, self.model_lang_id
                        )
                        lang_id_score = round(lang_id_score, 3)
                        st.markdown(
                            f"Language identification confidence score: {lang_id_score}"
                        )
                        if is_doc_discarded(key, lang_id_score) or (
                            self.lang_dataset_id != lang_pred_dataset_id
                        ):
                            is_discarded = True

                    elif key[0] == "perplexity_score":
                        perplexity_score = Filtering.compute_perplexity_score(
                            personal_doc,
                            self.sentencepiece_model,
                            self.kenlm_model,
                        )
                        perplexity_score = round(perplexity_score, 3)
                        st.markdown(f"Perplexity score: {perplexity_score}")
                        if is_doc_discarded(key, perplexity_score):
                            is_discarded = True

                is_discarded = "" if is_discarded else "not "
                st.markdown(
                    f"With the current filtering parameters, this document **is {is_discarded}discarded**."
                )

    def visualization_for_lang(self):
        self.set_title()
        self.open_data()
        self.filtering_of_docs()
        self.filtering_of_words()
        self.download_parameters()
        self.analyse_personal_doc()


class Visualization:
    def __init__(self, path_instructions, param_visu_langs):
        self.path_instructions = path_instructions
        self.param_visu_langs = param_visu_langs

    def preamble(self):
        def get_binary_file_downloader_html(bin_file, file_label="File"):
            with open(bin_file, "rb") as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
            return href

        st.markdown(
            "Before diving into this demo, you might want to take a look at how the filtering pipeline looks like in more detail in this "
            + get_binary_file_downloader_html(
                self.path_instructions,
                "pdf",
            )
            + ".",
            unsafe_allow_html=True,
        )

    def warning_preamble(self):
        st.markdown(
            "This demo can be a little slow, and only allows you to process up to 5000 documents "
            "for a decent speed. If you want to display up to three times more documents and have "
            "a faster visualization, we invite you to run this "
            "[code](https://github.com/bigscience-workshop/data_tooling/tree/master/ac_dc/visualization) "
            "on your computer."
        )

    def choose_lang(self):
        options = [
            self.param_visu_langs[lang_dataset_id]["lang"]
            for lang_dataset_id in self.param_visu_langs
        ]
        index = options.index("English") if ("English" in options) else 0
        lang_chosen = st.selectbox(
            label="Select the language for visualization",
            options=options,
            index=index,
        )
        if lang_chosen != "None":
            lang_chosen_dataset_id = langs_id.loc[
                langs_id["lang"] == lang_chosen, "dataset_id"
            ].iloc[0]
            visualization_for_lang = Visualization_for_lang(
                path_data=self.param_visu_langs[lang_chosen_dataset_id]["path_data"],
                lang=self.param_visu_langs[lang_chosen_dataset_id]["lang"],
                num_docs=self.param_visu_langs[lang_chosen_dataset_id]["num_docs"],
                num_docs_for_words=self.param_visu_langs[lang_chosen_dataset_id][
                    "num_docs_for_words"
                ],
                max_len_text_display=self.param_visu_langs[lang_chosen_dataset_id][
                    "max_len_text_display"
                ],
                lang_dataset_id=self.param_visu_langs[lang_chosen_dataset_id][
                    "lang_dataset_id"
                ],
                path_fasttext_model=self.param_visu_langs[lang_chosen_dataset_id][
                    "path_fasttext_model"
                ],
                path_sentencepiece_model=self.param_visu_langs[lang_chosen_dataset_id][
                    "path_sentencepiece_model"
                ],
                path_kenlm_model=self.param_visu_langs[lang_chosen_dataset_id][
                    "path_kenlm_model"
                ],
            )
            visualization_for_lang.visualization_for_lang()

    def visualization(self):
        self.preamble()
        # self.warning_preamble()
        self.choose_lang()


path_instructions = "./ac_dc/explanation_filtering_pipeline.pdf"

param_visu_langs = {
    lang_dataset_id: {
        "path_data": f"./ac_dc/visualization/{lang_dataset_id}_examples_with_stats.json",
        "lang": langs_id.loc[langs_id["dataset_id"] == lang_dataset_id, "lang"].iloc[0],
        "num_docs": 15000,
        "num_docs_for_words": 1500,
        "max_len_text_display": 10000,
        "lang_dataset_id": lang_dataset_id,
        "path_fasttext_model": "./ac_dc/lid.176.bin",
        "path_sentencepiece_model": f"./ac_dc/{lang_dataset_id}.sp.model",
        "path_kenlm_model": f"./ac_dc/{lang_dataset_id}.arpa.bin",
    }
    for lang_dataset_id in ["en", "zh"]
}

visualization = Visualization(path_instructions, param_visu_langs)
visualization.visualization()
