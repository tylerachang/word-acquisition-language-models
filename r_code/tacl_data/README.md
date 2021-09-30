# Data

This directory contains data used for the final paper.<br/>
Description of files:
* concreteness_data.tsv: concreteness norms from Brysbaert et al. (2014).

child_data:
* child_american_english_proportions.csv: proportions of children producing words at different ages, from Wordbank (Frank et al., 2017).
This file is only used to generate acquisition plots for individual words for children.
Manually acquired from the Wordbank site.
* child_aoa.tsv: cleaned child AoA data from get_child_aoa.R and clean_wordbank_words.py.
Contains cleaned tokens, smoothed ages of acquisition, and lexical classes.
Based on Wordbank data and code. 
* childes_eng-na.tsv: CHILDES data (word counts, mean sentence lengths) for individual words, from get_childes_data.py. Based on the CHILDES dataset (MacWhinney, 2000).

lm_data:
* lm_data_stats.txt: frequencies and mean sequence lengths for individual tokens in the language modeling training dataset.
* Raw data for evaluation loss during training (model_log.txt), surprisals for different tokens at each checkpoint (model_surprisals.txt), and cross-entropies and KL divergences with different distributions at different checkpoints (model_xent.txt).
* Processed data with fitted sigmoids for each word (model_sigmoids.txt).
Includes the sigmoid parameters for each token (upper/lower bounds, midpoint, scale).
Tokens have the new word character U+2581 removed.
* Processed data with age of acquisition for each word, based on the fitted sigmoids above.
Uses the method described in the paper (in model_aoa.txt) or using the midpoint of the fitted sigmoid (in model_aoa_midpoint.txt).
Tokens have the new word character U+2581 removed.
