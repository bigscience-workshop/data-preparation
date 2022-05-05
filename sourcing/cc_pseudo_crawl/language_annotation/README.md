# Language annotation

The main goal was to apply fastText on pseudo-crawl data documents, to get an intuition of the size of the datasets per language.

In addition, this annotation allowed to apply language dependent filtering methods later on.

This is essentially the purpose of the `annotate_langid_crawl.py` code.

The data of the pseudo-crawl being shared in several seeds, to gain in speed, the code was applied on hundreds of jobs on Jean Zay (one job per seed).
Some seeds did not work, some with silent errors.
The `check_wrong_files.py` code was used for debugging, and allowed to identify errors to correct them.

The `compute_stats_langid.py` code takes the pseudo-crawl data annotated with fastText information as a parameter, and computes the summed length of all documents classified as belonging to that language.
This allowed to have a first estimation of the distribution of languages in the pseudo-crawl data.