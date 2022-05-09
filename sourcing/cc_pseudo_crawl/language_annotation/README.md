# Language annotation
The file contains the different algorithmic methods we tried to determine the text language of the pseudo-crawl dataset examples. 

Knowing the language of each document was important for us to 1) have an idea of the size of the datasets per language and 2) have the possibility to apply filtering methods adapted to a particular language. 

We tested 2 methods:
- the prediction of a language identification model produced by FastText
- retrieving the language tag from the HTML code of the documents

## FastText method
The main objective of `annotate_langid_crawl.py` was to apply fastText to pseudo-crawl documents.

As the pseudo-crawl data is shared between several seeds, to gain speed, the code was applied to hundreds of jobs on Jean Zay (one job per seed).
Some seeds did not work, others with silent errors.
The `check_wrong_files.py` code was used for debugging, and helped to identify errors and correct them.

The `compute_stats_langid.py` code takes the pseudo-crawl data annotated with fastText information as a parameter, and calculates the cumulative length of all documents classified as belonging to that language.
This gave a first estimate of the distribution of languages in the pseudo-crawl data.

## HTML tag method
For this method we used the `02_detect_html_lang_attrib.slurm` - which uses `detect_html_lang_attrib.py`- script to annotate the whole pseudo-crawl dataset.