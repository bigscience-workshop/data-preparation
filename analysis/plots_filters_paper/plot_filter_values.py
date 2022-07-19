from datasets import load_from_disk
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


GLOBAL_PATH_DIR_DATASET = "/Users/hugolaurencon/Desktop/filter_values_small"

SAVE_PATH = "/Users/hugolaurencon/Desktop/plots"

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

for lang, iso_code in LANGUAGES.items():

    PATH_DIR_DATASET = {
        "Catalogue": f"{GLOBAL_PATH_DIR_DATASET}/{iso_code}_catalogue",
        "Pseudo Crawl": f"{GLOBAL_PATH_DIR_DATASET}/{iso_code}_PC",
        "Common Crawl": f"{GLOBAL_PATH_DIR_DATASET}/{iso_code}_CC",
    }

    DATASETS = {}
    for key, val in PATH_DIR_DATASET.items():
        try:
            DATASETS[key] = load_from_disk(val)
        except:
            pass

    MAX_NUMBER_FILTER_VALUES = 1_000_000
    MAX_PERCENTILE = 99

    FILTERS = {
        "number_of_words": "max",
        "character_repetition_ratio": "max",
        "word_repetition_ratio": "max",
        "special_character_ratio": "max",
        "closed_class_word_ratio": "max",
        "flagged_word_ratio": "none",
        "perplexity_score": "max",
    }

    for filter, type_max_percentile in FILTERS.items(): 

        type_data = []
        filter_values = []

        for t_data, dataset in DATASETS.items():
            new_filter_values = dataset[filter][:MAX_NUMBER_FILTER_VALUES]
            if type_max_percentile == "max":
                max_quantile = np.percentile(new_filter_values, MAX_PERCENTILE)
                new_filter_values = [val for val in new_filter_values if val <= max_quantile]
            random.shuffle(new_filter_values)
            new_type_data = [t_data] * len(new_filter_values)
            type_data.append(new_type_data)
            filter_values.append(new_filter_values)

        # Oversampling to match the size of the largest list
        max_len = max([len(list_filter_values) for list_filter_values in filter_values])
        for i in range(len(filter_values)):
            current_len = len(filter_values[i])
            if current_len != max_len:
                filter_values[i] = filter_values[i] * (max_len // current_len) + filter_values[i][:max_len % current_len]
                type_data[i] = type_data[i] * (max_len // current_len) + type_data[i][:max_len % current_len]

        filter_values = [val for list_filter_values in filter_values for val in list_filter_values]
        type_data = [t_data for list_typ_data in type_data for t_data in list_typ_data]

        if filter == "flagged_word_ratio":
            idx_del = []
            for ind, val in enumerate(filter_values):
                if val == 0:
                    idx_del.append(ind)
            idx_del = set(idx_del)
            filter_values = [val for ind, val in enumerate(filter_values) if ind not in idx_del]
            type_data = [val for ind, val in enumerate(type_data) if ind not in idx_del]

        dataframe = {
            "Data": type_data,
            "Filter value": filter_values, 
        }
        dataframe = pd.DataFrame(dataframe)

        displot = sns.displot(
            dataframe,
            x="Filter value",
            hue="Data",
            kind="kde",
            fill=True,
            palette={
                "Catalogue": "#003d9e",
                "Pseudo Crawl": "#3498db",
                "Common Crawl": "#95a5a6",
            },
        )
        sns.move_legend(displot, "upper right", bbox_to_anchor=(.7, .8), title="Data")
        plt.subplots_adjust(top=0.9)
        filter_title = filter.replace("_", " ")
        plt.suptitle(f"Distribution of the filter values for the filter \non the {filter_title} for {lang}", fontsize = 10)
        #plt.xlim(0, 0.015) # Used only for flagged_word_ratio
        #plt.show()
        plt.savefig(f"{SAVE_PATH}/{iso_code}_{filter}.png", dpi=300)

        print(f"Filter {filter} for language {lang} done!")

    print(f"Successfully done language {lang}!")
    print("\n\n\n")

