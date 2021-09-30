"""
Evaluate language model at each step.
Computes cross-entropy with a uniform distribution, token-frequency distribution,
and a bigram distribution (based on adjacent token(s) only).
Also computes accuracy and loss.
See readme for sample usage.
"""

import logging
import math
import os
import json
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import codecs
from collections import Counter
import random
import statistics
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
    AlbertTokenizer,
)

from rnn_models import (
    RNNLanguageModel,
    BidirectionalRNNLanguageModel,
)
from run_rnn_language_modeling import format_inputs
from word_evaluation import run_model, load_single_model

def create_parser():
    parser = argparse.ArgumentParser()
    # Note: config.json should be in the model directory.
    parser.add_argument('--model_dir', default="")
    # gpt2, bert, lstm, or bilstm.
    parser.add_argument('--model_type', default="")
    parser.add_argument('--tokenizer', default="")
    parser.add_argument('--output_file', default="xent.txt")
    # If empty, uses all checkpoints in the directory.
    parser.add_argument('--checkpoints', type=int, nargs='*')
    parser.add_argument('--batch_size', type=int, default=32)
    # Tokenized examples.
    parser.add_argument('--examples_file', default="")
    # The number of examples to use from the examples file.
    # For best results, this should be a multiple of batch_size.
    parser.add_argument('--max_samples', type=int, default=16384)
    # Load token data (sample sentences for each token) from file.
    # If file does not exist, saves the examples to this file.
    parser.add_argument('--save_samples', default="masked_eval_samples.txt")
    parser.add_argument('--bigrams_file', default="bigram_counts.pt")
    parser.add_argument('--frequencies_file', default="lm_data_stats.txt")
    # Set nonempty to evaluate backward bigrams (takes more memory).
    parser.add_argument('--backward_bigrams', default="")
    return parser


# Create an iterable_pairs file of masked examples from the input file.
# Each line will be a space-separated list of token ids.
# Masked token ids will have _m appended after them.
def create_masked_samples(tokenizer, tokenized_examples_file, save_file, max_samples, max_seq_len):
    # Load sentences.
    infile = codecs.open(tokenized_examples_file, 'rb', encoding='utf-8')
    outfile = codecs.open(save_file, 'wb', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count % 100000 == 0:
            print("Finished line {}.".format(line_count))
        if line_count >= max_samples:
            break
        example = line.strip().split()
        if len(example) > max_seq_len:
            example = example[:max_seq_len]
        # Do not include endpoints, to allow bidirectional predictions.
        # These are usually [CLS] and [SEP] tokens anyways.
        # Exactly one mask per sequence.
        mask_idx = random.randint(1, len(example)-2) # Inclusive.
        example[mask_idx] = example[mask_idx] + "_m"
        new_example_string = " ".join(example)
        outfile.write("{}\n".format(new_example_string))
    infile.close()
    outfile.close()
    # Logging.
    print("Saved masked eval examples.")
    return


# Get a vector of token frequencies.
def get_unigram_probs(frequencies_file, tokenizer):
    frequencies = torch.zeros(len(tokenizer), dtype=torch.float32)
    infile = codecs.open(frequencies_file, 'rb', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count == 0: # Ignore header.
            continue
        split_line =  line.strip().split('\t')
        if len(split_line) != 4: # Contains token, frequency, MLU, unidirectional MLU.
            continue
        token = split_line[0] # Note: already fully processed token (lowercase and with space character).
        token_id = tokenizer._convert_token_to_id_with_added_voc(token)
        if token_id != tokenizer.unk_token_id:
            freq = float(split_line[1]) # Counts are frequencies per 1000 tokens.
            frequencies[token_id] = freq
    infile.close()
    # Normalize.
    frequencies = frequencies / torch.sum(frequencies)
    return frequencies


def process_xent_batch(logits, y_true_list, previous_ids_list, next_ids_list,
                       bigram_probs, p_unigram, metrics_dict, bidirectional):
    # Updates the metrics_dict by adding the metric value for each sequence in the batch.
    p_pred = torch.nn.Softmax(dim=-1)(logits) # Shape: batch_size x vocab_size.
    p_pred += 0.000000001 # Smooth with (1e-9).
    logp_pred = torch.log2(p_pred)
    y_true = torch.tensor(y_true_list, dtype=torch.int64)
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]
    # Get accuracy:.
    predictions = torch.argmax(p_pred, dim=-1) # Shape: batch_size.
    num_correct = torch.sum(predictions == y_true).item()
    metrics_dict["Accuracy"] += num_correct
    # Get loss.
    pred_scores = torch.gather(p_pred, -1, y_true.reshape(batch_size, 1))
    loss = -1.0 * torch.sum(torch.log2(pred_scores)).item()
    metrics_dict["Loss"] += loss

    # Get uniform metrics.
    p_uniform = torch.ones(vocab_size, dtype=torch.float32) / vocab_size
    xent = -1.0 * torch.sum(p_uniform.reshape(1, vocab_size) * logp_pred).item() # Sum over all batches and token ids.
    metrics_dict["UniformXent"] += xent
    ent = -1.0 * torch.sum(p_uniform * torch.log2(p_uniform)).item() # Entropy H(p_uniform) for each example.
    metrics_dict["UniformKL"] += xent - ent*batch_size
    # Get unigram metrics.
    xent = -1.0 * torch.sum(p_unigram.reshape(1, vocab_size) * logp_pred).item()
    metrics_dict["UnigramXent"] += xent
    ent = -1.0 * torch.sum(p_unigram * torch.log2(p_unigram)).item()
    metrics_dict["UnigramKL"] += xent - ent*batch_size

    # Get bigram metrics.
    # Forward bigrams.
    p_next_bigrams = [] # P(w_i | w_{i-1}).
    for id in previous_ids_list:
        p_next_bigrams.append(bigram_probs[0][id, :])
    p_next_bigrams = torch.stack(p_next_bigrams, dim=0) # Shape: batch_size x vocab_size.
    p_next_bigrams += 0.000000001
    xent = -1.0 * torch.sum(p_next_bigrams * logp_pred).item()
    metrics_dict["ForwardBigramXent"] += xent
    ent = -1.0 * torch.sum(p_next_bigrams * torch.log2(p_next_bigrams)).item() # Computed over all sequences.
    metrics_dict["ForwardBigramKL"] += xent - ent
    # KL divergence between forward bigrams and unigrams:
    # xent = -1.0 * torch.sum(p_next_bigrams * torch.log2(p_unigram)).item()
    # if "ForwardBigramUnigram" not in metrics_dict:
    #     metrics_dict["ForwardBigramUnigram"] = 0.0
    # metrics_dict["ForwardBigramUnigram"] += xent-ent
    p_next_bigrams -= 0.000000001 # Undo smoothing.

    # Backward bigrams.
    if bigram_probs[1] is not None:
        p_prev_bigrams = [] # P(w_i | w_{i+1}).
        for id in next_ids_list:
            p_prev_bigrams.append(bigram_probs[1][:, id])
        p_prev_bigrams = torch.stack(p_prev_bigrams, dim=0) # Shape: batch_size x vocab_size.
        p_prev_bigrams += 0.000000001
        xent = -1.0 * torch.sum(p_prev_bigrams * logp_pred).item()
        metrics_dict["BackwardBigramXent"] += xent
        ent = -1.0 * torch.sum(p_prev_bigrams * torch.log2(p_prev_bigrams)).item() # Computed over all sequences.
        metrics_dict["BackwardBigramKL"] += xent - ent

    # Bidirectional bigrams.
    # Compute using P(w_i | w_{i-1}) * P(w_{i+1} | w_i).
    # Alternative P(w_i | w_{i-1}) * P(w_i | w_{i+1}) / P(w_i) causes issues
    # because of division for probabilities near zero.
    p_next2_bigrams = [] # P(w_{i+1} | w_i).
    for id in next_ids_list:
        p_next2_bigrams.append(bigram_probs[0][:, id])
    p_next2_bigrams = torch.stack(p_next2_bigrams, dim=0) # Shape: batch_size x vocab_size.
    # The bigram probability for each w_i is proportional to the product.
    p_bigrams = p_next_bigrams * p_next2_bigrams
    p_bigrams = p_bigrams / torch.sum(p_bigrams, dim=-1, keepdim=True)
    p_bigrams[torch.isnan(p_bigrams)] = 0.0
    p_bigrams += 0.000000001 # Smooth.
    xent = -1.0 * torch.sum(p_bigrams * logp_pred).item()
    metrics_dict["BidirBigramXent"] += xent
    ent = -1.0 * torch.sum(p_bigrams * torch.log2(p_bigrams)).item() # Computed over all sequences.
    metrics_dict["BidirBigramKL"] += xent - ent
    # KL divergence between bidirectional bigrams and unigrams:
    # xent = -1.0 * torch.sum(p_bigrams * torch.log2(p_unigram)).item()
    # if "BidirBigramUnigram" not in metrics_dict:
    #     metrics_dict["BidirBigramUnigram"] = 0.0
    # metrics_dict["BidirBigramUnigram"] += xent-ent

    # Return updated dict.
    return metrics_dict


# Run evaluations for a single model.
def evaluate_xent(model, model_type, tokenizer, outfile,
                  masked_examples_file, bigram_probs, unigram_probs,
                  curr_steps, batch_size):
    """
    Notes on the selected metrics:
    Xent is computed as H(y_true, y_pred). This is the expected log loss if the
    true distribution is y_true. It can also be interpreted as the expected number
    of bits required to encode a message from y_true given a code optimized for y_pred.
    All logarithms use log base 2. The KL divergence is also computed as
    KL(y_true, y_pred) = H(y_true, y_pred) - H(y_true).

    The loss is computed as usual: H(one_hot, y_pred) = -log(y_pred[correct_class]).
    The KL divergence from the loss is equal to the loss, because H(one_hot) = 0.
    All metrics are averaged over sequences.
    """
    bidirectional = (model_type ==  "bert" or model_type == "bilstm")
    # This will contain the total for each metric, which will later be divided
    # by num_examples.
    metrics_dict = {"Accuracy": 0.0, "Loss": 0.0,
                    "UniformXent": 0.0, "UniformKL": 0.0,
                    "UnigramXent": 0.0, "UnigramKL": 0.0,
                    "BidirBigramXent": 0.0, "BidirBigramKL": 0.0,
                    "ForwardBigramXent": 0.0, "ForwardBigramKL": 0.0,
                    "BackwardBigramXent": 0.0, "BackwardBigramKL": 0.0}

    # Read examples.
    infile = codecs.open(masked_examples_file, 'rb', encoding='utf-8')
    example_count = 0
    example_batch = []
    y_true = [] # The correct predictions for the masks.
    previous_ids = [] # The preceding tokens before masks (used for bigrams).
    next_ids = [] # The next tokens after masks (used for bidirectional bigrams).
    for line in infile:
        if len(line.strip()) == 0:
            continue
        # Add the example to the batch.
        example_raw = line.strip().split()
        example = []
        for idx, string_id in enumerate(example_raw):
            if "_m" in string_id:
                # This token is masked. Save the true prediction and the previous/next tokens.
                token_id = tokenizer.mask_token_id
                true_id = int(string_id[:-2])
                y_true.append(true_id) # Add the true label.
                if idx > 0: # Add the previous token.
                    previous_ids.append(example[idx-1])
                else:
                    previous_ids.append(-1)
                if idx < len(example_raw)-1: # Add the next token.
                    next_ids.append(int(example_raw[idx+1]))
                else:
                    next_ids.append(-1)
            else:
                token_id = int(string_id)
            example.append(token_id)
        example_batch.append(example)
        example_count += 1
        # Process a batch.
        if example_count % batch_size == 0:
            logits = run_model(model, model_type, example_batch, batch_size, tokenizer)
            metrics_dict = process_xent_batch(logits, y_true, previous_ids, next_ids,
                                              bigram_probs, unigram_probs, metrics_dict, bidirectional)
            example_batch = []
            y_true = []
            previous_ids = []
            next_ids = []
    infile.close()
    # Process last batch.
    if len(example_batch) != 0:
        logits = run_model(model, model_type, example_batch, batch_size, tokenizer)
        metrics_dict = process_xent_batch(logits, y_true, previous_ids, next_ids,
                                          bigram_probs, unigram_probs, metrics_dict, bidirectional)
    # Logging:
    final_metrics_dict = dict()
    for key, value in metrics_dict.items():
        final_metrics_dict[key] = value / example_count
    del metrics_dict
    print("Metrics:")
    print(final_metrics_dict)
    outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(
        curr_steps, final_metrics_dict["Accuracy"], final_metrics_dict["Loss"],
        final_metrics_dict["UniformXent"], final_metrics_dict["UniformKL"],
        final_metrics_dict["UnigramXent"], final_metrics_dict["UnigramKL"],
        final_metrics_dict["BidirBigramXent"], final_metrics_dict["BidirBigramKL"],
        final_metrics_dict["ForwardBigramXent"], final_metrics_dict["ForwardBigramKL"],
        final_metrics_dict["BackwardBigramXent"], final_metrics_dict["BackwardBigramKL"]))
    return


def main(args):
    # Load config.
    config_path = os.path.join(args.model_dir, "config.json")
    args.model_type = args.model_type.lower()
    if args.model_type == "gpt2" or args.model_type == "bert":
        config = AutoConfig.from_pretrained(config_path)
    else: # LSTMs.
        with codecs.open(config_path, "rb", encoding="utf-8") as reader:
            text = reader.read()
        config = json.loads(text)
        max_seq_len = config["max_seq_len"]

    # Load tokenizer.
    print("Attempting to use local sentencepiece model file as tokenizer.")
    tokenizer = AlbertTokenizer.from_pretrained(args.tokenizer)
    # Overwrite special token ids in the configs.
    if args.model_type == "bert":
        config.pad_token_id = tokenizer.pad_token_id
        max_seq_len = config.max_position_embeddings
    elif args.model_type == "gpt2":
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id
        max_seq_len = config.n_positions

    # Create the masked eval samples file if necessary.
    if not os.path.isfile(args.save_samples):
        create_masked_samples(tokenizer, args.examples_file, args.save_samples,
                              args.max_samples, max_seq_len)

    # Prepare for evaluation.
    outfile = codecs.open(args.output_file, 'w', encoding='utf-8')
    # File header.
    # Note: loss/entropy/surprisal logarithms in base 2.
    outfile.write("Steps\tAccuracy\tLoss\tUniformXent\tUniformKL\tUnigramXent\tUnigramKL\tBidirBigramXent\tBidirBigramKL\t")
    outfile.write("ForwardBigramXent\tForwardBigramKL\tBackwardBigramXent\tBackwardBigramKL\n")
    # Get checkpoints.
    if args.checkpoints is None or len(args.checkpoints) == 0:
        checkpoints = set() # Set of ints.
        for root, dirs, files in os.walk(args.model_dir):
            for dir in dirs:
                if "checkpoint-" in dir:
                    checkpoint = int(dir.strip().split("-")[-1])
                    checkpoints.add(checkpoint)
    else:
        checkpoints = args.checkpoints
    checkpoints = list(checkpoints)
    checkpoints.sort()

    # Shape: vocab_size.
    unigram_probs = get_unigram_probs(args.frequencies_file, tokenizer)
    unigram_probs += 0.000000001 # Smooth.

    # Shape: vocab_size x vocab_size.
    bigram_counts = torch.load(args.bigrams_file) # Starts as counts.
    # P(w_i | w_{i-1}). Rows w_{i-1}, columns w_i.
    bigram_forward_probs = bigram_counts / torch.sum(bigram_counts, dim=-1, keepdim=True)
    # A few tokens never appear in the training set, so their row sum is zero,
    # leading to NaN for the entire row.
    bigram_forward_probs[torch.isnan(bigram_forward_probs)] = 0.0
    bigram_backward_probs = None
    if args.backward_bigrams:
        # P(w_i | w_{i+1}). Rows w_i, columns w_{i+1}.
        bigram_backward_probs = bigram_counts / torch.sum(bigram_counts, dim=0, keepdim=True)
        bigram_backward_probs[torch.isnan(bigram_backward_probs)] = 0.0
    del bigram_counts
    # Warning: this can be a lot of memory.
    bigram_probs = (bigram_forward_probs, bigram_backward_probs)

    # Run evaluation.
    for checkpoint in checkpoints:
        print("CHECKPOINT STEPS: {}".format(checkpoint))
        single_model_dir = os.path.join(args.model_dir, "checkpoint-{}".format(checkpoint))
        model = load_single_model(single_model_dir, args.model_type, config, tokenizer)
        evaluate_xent(model, args.model_type, tokenizer, outfile, args.save_samples,
                      bigram_probs, unigram_probs, checkpoint, args.batch_size)
    outfile.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.backward_bigrams = args.backward_bigrams != ""
    main(args)
