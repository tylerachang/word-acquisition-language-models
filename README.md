# word-acquisition-language-models
Code and data for the paper [Word Acquisition in Neural Language Models](https://arxiv.org/abs/2110.02406) (TACL 2021).
Includes code for pre-training and evaluating surprisals in Transformer (BERT, GPT) and RNN (unidirectional and bidirectional LSTM) language models.
Also contains code to analyze words' ages of acquisition in children and language models during pre-training.
Last tested on Python 3.9.13, Pytorch 1.10.2, and Hugging Face Transformers 4.14.1 (see requirements.txt).
Data is in r_code/tacl_data.

## Training language models.
This section contains instructions to train a language model from scratch.
First, place your training and evaluation raw text files in the sample_data directory, with file names train_text.txt and eval_text.txt.
Then, train a tokenizer on your training data:
<pre>
python3 scripts/train_spm_tokenizer.py \
--input_file="./sample_data/train_text.txt" \
--output="./sample_data/spm" \
--vocab_size=30000
</pre>
Then, tokenize the training dataset.
In this example, we concatenate each pair of lines.
<pre>
python3 scripts/tokenize_dataset.py \
--tokenizer="./sample_data/spm.model" \
--input_file="./sample_data/train_text.txt" \
--output_file="./sample_data/train_tokenized.txt" \
--max_segments=2 --max_seq_len=-1
</pre>
Repeat for the evaluation dataset.
<pre>
python3 scripts/tokenize_dataset.py \
--tokenizer="./sample_data/spm.model" \
--input_file="./sample_data/eval_text.txt" \
--output_file="./sample_data/eval_tokenized.txt" \
--max_segments=2 --max_seq_len=-1
</pre>
Then, train a model on the tokenized training dataset.
For Transformer language models (BERT and GPT-2), the config should specify the model vocab size, which should be the tokenizer vocab size + 4 (for CLS, SEP, PAD, and MASK tokens).
For RNN language models, the vocab size is inferred from the tokenizer.
Sample configs are provided in the lm_configs directory.
The code below should be run with GPU(s) unless the config specifies a fairly small language model.
To train BERT:
<pre>
python3 lm_code/run_transformer_language_modeling.py \
--tokenizer_name="./sample_data/spm.model" \
--config_name="./lm_configs/bert_base_config.json" \
--do_train --do_eval \
--train_data_file="./sample_data/train_tokenized.txt" \
--train_iterable --eval_iterable \
--eval_data_file="./sample_data/eval_tokenized.txt" \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--evaluation_strategy="steps" \
--eval_steps=1000 \
--save_steps=1000 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--output_dir="./models/bert"
</pre>
To train GPT-2, replace the following options:
<pre>
--config_name="./lm_configs/gpt2_config.json" \
--output_dir="./models/gpt2"
</pre>
To train a forward RNN (LSTM):
<pre>
python3 lm_code/run_rnn_language_modeling.py \
--tokenizer_name="./sample_data/spm.model" \
--config_name="./lm_configs/rnn_unidirectional_config.json" \
--do_train --do_eval \
--train_data_file="./sample_data/train_tokenized.txt" \
--train_iterable --eval_iterable \
--eval_data_file="./sample_data/eval_tokenized.txt" \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--evaluation_strategy="steps" \
--eval_steps=1000 \
--save_steps=1000 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--output_dir="./models/rnn-unidirectional"
</pre>
To train a bidirectional RNN (LSTM), replace the following options:
<pre>
--config_name="./lm_configs/rnn_bidirectional_config.json" \
--output_dir="./models/rnn-bidirectional"
</pre>

## Word/token evaluation.
To collect surprisals for individual tokens at each checkpoint:
<pre>
python3 lm_code/word_evaluation.py \
--tokenizer="./sample_data/spm.model" \
--wordbank_file="./r_code/tacl_data/child_data/child_aoa.tsv" \
--examples_file="./sample_data/eval_tokenized.txt" \
--max_samples=512 \
--batch_size=128 \
--output_file="./sample_data/bert_surprisals.txt" \
--model_dir="./models/bert" --model_type="bert" \
--save_samples="./sample_data/bidirectional_samples.pickle"
</pre>
To use different models, replace the model_dir and model_type (bert, gpt2, lstm, or bilstm), and save the samples as unidirectional or bidirectional (corresponding to model type).
The saved samples store the occurrences of target tokens in the eval dataset.

To run specific examples, the run_model() function in word_evaluation.py can run tokenized sequences through any of the four model types, outputting masked token probabilities.
Each model can be loaded with the load_single_model() function in word_evaluation.py.

## Cross-entropy and KL divergence.
To compute cross-entropies and KL divergences with the uniform, unigram, and bigram distributions:
<pre>
python3 lm_code/xent_evaluation.py \
--tokenizer="./sample_data/spm.model" \
--examples_file="./sample_data/eval_tokenized.txt" \
--frequencies_file="./sample_data/lm_data_stats.txt" \
--bigrams_file="./sample_data/bigram_counts.pt" \
--batch_size=128 \
--backward_bigrams=yes \
--output_file="./sample_data/bert_xent.txt" \
--model_dir="./models/bert" --model_type="bert" \
--save_samples="./sample_data/masked_eval_samples.txt"
</pre>
As in word/token evaluation above, this can use any of the four different model types, and it requires that the sequences are already tokenized in the evaluation file.
The frequencies_file (containing token frequencies) can be generated by scripts/count_tokens.py (which also outputs mean sequence lengths).
The bigrams file can be generated by scripts/get_bigram_matrix.py.

## Child data.
The pipeline for obtaining the child age of acquisition data is as follows.
Child age of acquisition data is pulled from Wordbank (Frank et al., 2017).
Child-directed speech data is pulled from CHILDES (MacWhinney, 2000).
1. Get age of acquisition data (r_code/get_child_aoa.R).
Note that this pulls multilingual AoA data.
2. Clean the AoA token data from the previous step (scripts/clean_wordbank_words.py).
This is beause the UTF-8 bytes can sometimes get offset.
This adds a CleanedWord and CleanedSingle column to the dataset, where CleanedSingle represents each word's single canonical form.
The output should be saved as the child AoA file (e.g. child_aoa.tsv).
3. Download the American English trajectory data (proportions learned for each month) as a csv from the Wordbank site.
This should be the by-word summary data.
This should be saved as the child proportions file (e.g. child_american_english_proportions.csv).
4. Get the CHILDES data (scripts/get_childes_data.py).
This outputs a txt and tsv file for each language.
The txt file contains the raw sentences, and the tsv file contains statistics (word counts and mean sentence lengths) for each word.
The North American English tsv should be saved as the CHILDES data file (e.g. childes_eng-na.tsv).
5. Get the concreteness norms directly from the supplementary materials in Brysbaert et al. (2014), saved as a tsv (e.g. concreteness_data.tsv).

## Analyses.
After running the code above, all analyses can be reproduced using r_code/tacl_analyses.rmd.
This includes fitting sigmoids to the LM token surprisal data, computing ages of acquisition for the LMs, running regressions for both the child and LM AoA data, and generating plots.
Original data is in r_code/tacl_data.

## Citation.
<pre>
@article{chang-bergen-2021-word,
  title={Word Acquisition in Neural Language Models},
  author={Tyler Chang and Benjamin Bergen},
  journal={Transactions of the Association for Computational Linguistics},
  year={2021},
}
</pre>
