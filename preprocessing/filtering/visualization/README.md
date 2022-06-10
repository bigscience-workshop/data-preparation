# Visualization tool

Use this visualization tool online at https://huggingface.co/spaces/huggingface/text-data-filtering.

However, by running the code on your computer, it is faster, it can handle in practice up to three times more documents, and it works for every language.

1) Use get_data_for_visualization.py to get the json gathering examples with their computed statistics for the language you chose.
It uses the streaming mode of the Datasets library, so no need to download the dataset, but you have to download the fasttext model (for the language identification) and the sentencepiece / kenlm models (for the tokenization and the perplexity).

2) Specify the path to this json and the fasttext / sentencepiece / kenlm models in visualization.py and run the command "streamlit run visualization/visualization.py".
