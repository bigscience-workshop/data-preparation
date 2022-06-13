from datasets import load_from_disk
from parameters_filtering import parameters_filtering
import matplotlib.pyplot as plt
import seaborn as sns


GLOBAL_PATH_DIR_DATASET = "/Users/hugolaurencon/Desktop/filter_values_CC_unfiltered"

SAVE_PATH = "/Users/hugolaurencon/Desktop/plots_discarded_by_filters"

LANGUAGES = {
    "Arabic": "ar",
    "Basque": "eu",
    "Bengali": "bn",
    "Catalan": "ca",
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "Hindi": "hi",
    "Indonesian": "id",
    "Portuguese": "pt",
    "Spanish": "es",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

FILTERS = {
        "number_of_words": ("min", "number_words_min_cutoff", "cond_check_number_words"),
        "character_repetition_ratio": ("max", "character_repetition_max_cutoff", "cond_check_character_repetition_removal"),
        "word_repetition_ratio": ("max", "word_repetition_max_cutoff", "cond_check_word_repetition_removal"),
        "special_character_ratio": ("max", "special_characters_max_cutoff", "cond_check_special_characters"),
        "closed_class_word_ratio": ("min", "stopwords_min_cutoff", "cond_check_stopwords"),
        "flagged_word_ratio": ("max", "flagged_words_max_cutoff", "cond_check_flagged_words"),
        "perplexity_score": ("max", "perplexity_max_cutoff", "cond_check_perplexity"),
}


def should_keep_val(val, threshold, type_threshold):
    if type_threshold == "max":
        if val > threshold:
            return False
        return True
    elif type_threshold == "min":
        if val < threshold:
            return False
        return True
    else:
        print("WARNING: wrong type_threshold")

for lang, iso_code in LANGUAGES.items():

    dataset = load_from_disk(f"{GLOBAL_PATH_DIR_DATASET}/{iso_code}")

    removed_by_filters = {key: 0 for key in FILTERS}

    for filter, (type_threshold, key_threshold, key_cond_filter) in FILTERS.items():

        if not parameters_filtering[iso_code][key_cond_filter]:
            removed_by_filters[filter] = 0.0

        else:
            filter_values = dataset[filter]
            threshold = parameters_filtering[iso_code][key_threshold]
            removed_by_filters[filter] = len(
                [val for val in filter_values if not should_keep_val(val, threshold, type_threshold)]
            ) / len(filter_values) * 100

    print(removed_by_filters)

    x = [filter.replace("_", "\n") for filter in list(FILTERS.keys())]
    y = [removed_by_filters[filter] for filter in FILTERS]
    sns.barplot(x, y)
    plt.ylim(bottom=0)
    plt.show()
