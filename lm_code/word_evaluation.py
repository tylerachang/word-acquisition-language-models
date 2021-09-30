"""
Evaluate language models (surprisal, accuracy, rank of correct prediction) for
certain tokens. See readme for sample usage.
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

def create_parser():
    parser = argparse.ArgumentParser()
    # Note: config.json should be in the model directory.
    parser.add_argument('--model_dir', default="")
    # gpt2, bert, lstm, or bilstm.
    parser.add_argument('--model_type', default="")
    # The sentencepiece tokenizer for the model.
    parser.add_argument('--tokenizer', default="")
    parser.add_argument('--output_file', default="surprisals.txt")
    # If empty, uses all checkpoints in the directory.
    parser.add_argument('--checkpoints', type=int, nargs='*')
    parser.add_argument('--batch_size', type=int, default=32)
    # Child AoA file, used to identify the desired wordbank words.
    parser.add_argument('--wordbank_file', default="child_aoa.tsv")
    parser.add_argument('--wordbank_lang', default="English (American)")
    # Examples should already be tokenized. Each line should be a
    # space-separated list of integer token ids.
    parser.add_argument('--examples_file', default="")
    # The minimum number of sample sentences to evaluate a token.
    parser.add_argument('--min_samples', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=512)
    # The minimum sequence length to evaluate a token in a sentence.
    # For unidirectional models, this only counts context before the target token.
    parser.add_argument('--min_seq_len', type=int, default=8)
    # Load token data (sample sentences for each token) from file.
    # If file does not exist, saves the token data to this file.
    parser.add_argument('--save_samples', default="")
    # Whether to include token inflections (all, only, or none).
    parser.add_argument('--inflections', default="none")
    return parser


def get_inflected_tokens(wordbank_tokens, inflections):
    # Get inflected tokens if desired (all, only, or none).
    # Note: inflections are not tagged with their inflection (e.g. NN vs. NNS),
    # so inflection data will need to be marked later.
    if inflections != "all" and inflections != "only":
        return wordbank_tokens # No inflections.
    # Get inflections.
    import spacy # Import here because unneeded otherwise.
    import lemminflect
    all_inflections = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
                   "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
                   "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO",
                   "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT",
                   "WP", "WP$"]
    spacy_nlp = spacy.load('en')
    inflected_tokens = set()
    for token in wordbank_tokens:
        base_token = spacy_nlp(token)[0]._
        for inflection in all_inflections:
            inflected_tokens.add(base_token.inflect(inflection))
    del spacy_nlp
    print("Obtained inflected tokens.")
    # Return the desired inflections.
    if inflections == "all": # Inflections and base forms.
        return inflected_tokens.union(wordbank_tokens)
    elif inflections == "only": # Only use inflections.
        return inflected_tokens.difference(wordbank_tokens)
    return set() # Should never reach this line.


def get_sample_sentences(tokenizer, wordbank_file, wordbank_lang, tokenized_examples_file,
                         max_seq_len, min_seq_len, max_samples, bidirectional=True,
                         inflections="none"):
    # Each entry of token data is a tuple of token, token_id, masked_sample_sentences.
    token_data = []
    # Load words.
    # Assume that wordbank words are cleaned in the final column.
    # Words are not inflected, because the tokenizer treats inflections as separate tokens.
    infile = codecs.open(wordbank_file, 'rb', encoding='utf-8')
    wordbank_tokens = set()
    for line in infile:
        split_line = line.strip().split('\t')
        if split_line[5] == wordbank_lang:
            wordbank_tokens.add(split_line[-1])
    infile.close()
    # Get inflections if desired.
    wordbank_tokens = get_inflected_tokens(wordbank_tokens, inflections)
    # Get token ids.
    for token in wordbank_tokens:
        token = "\u2581" + token # Add space before (for spm tokenizer).
        token_id = tokenizer._convert_token_to_id_with_added_voc(token)
        if token_id != tokenizer.unk_token_id:
            token_data.append(tuple([token, token_id, []]))
    # Load sentences.
    infile = codecs.open(tokenized_examples_file, 'rb', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count % 100000 == 0:
            print("Finished line {}.".format(line_count))
        example_string = line.strip()
        example = [int(token_id) for token_id in example_string.split()]
        # Use the pair of sentences (instead of individual sentences), to have
        # longer sequences. Also more similar to training.
        if len(example) < min_seq_len:
            continue
        if len(example) > max_seq_len:
            example = example[:max_seq_len]
        for token, token_id, sample_sents in token_data:
            if len(sample_sents) >= max_samples:
                # This token already has enough sentences.
                continue
            token_indices = [index for index, curr_id in enumerate(example) if curr_id == token_id]
            # Warning: in bidirectional contexts, the mask can be in the first or last position,
            # which can cause no mask prediction to be made for the biLSTM.
            if not bidirectional:
                # The token must have enough unidirectional context.
                # The sequence length (including the target token) must be at least min_seq_len.
                token_indices = [index for index in token_indices if index >= min_seq_len-1]
            if len(token_indices) > 0:
                new_example = example.copy()
                mask_idx = random.choice(token_indices)
                new_example[mask_idx] = tokenizer.mask_token_id
                sample_sents.append(new_example)
    infile.close()
    # Logging.
    for token, token_id, sample_sents in token_data:
        print("{0} ({1}): {2} sentences.".format(token, token_id, len(sample_sents)))
    return token_data


# Convert a list of integer token_id lists into input_ids, attention_mask, and labels tensors.
# Inputs should already include CLS and SEP tokens.
# All sequences will be padded to the length of the longest example, so this
# should be called per batch.
# Note that the mask token will remain masked in the labels as well.
def prepare_tokenized_examples(tokenized_examples, tokenizer, model_type):
    # Convert into a tensor.
    tensor_examples = [torch.tensor(e, dtype=torch.long) for e in tokenized_examples]
    input_ids = pad_sequence(tensor_examples, batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    labels = input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100
    attention_mask = input_ids != tokenizer.pad_token_id
    # Format for LSTMs as in run_rnn_language_modeling.py.
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if model_type == "bilstm" or model_type == "lstm":
        bidirectional = model_type == "bilstm"
        # Note: this also puts input_ids and labels on cuda.
        inputs = format_inputs(inputs, input_ids.shape[1], bidirectional=bidirectional)
    elif torch.cuda.is_available():
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
        inputs["labels"] = inputs["labels"].cuda()
    return inputs


# Output the logits given examples (lists of token_ids).
# Assumes one mask per example. Outputs tensor of shape n_examples x vocab_size.
# Only considers the masked token.
# Handles batching and example tensorizing.
# model_type is bert, gpt2, lstm, or bilstm.
def run_model(model, model_type, examples, batch_size, tokenizer):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    num_dummies = 0
    if len(examples) % batch_size != 0:
        batches.append(examples[i:])
        # Fill the remainder of the last batch with dummy examples.
        # Huggingface Transformers already handles batch sizes that are not
        # divisible by n_gpus.
        if model_type == "lstm" or model_type == "bilstm":
            dummy_example = [0, 0, tokenizer.mask_token_id, 0, 0]
            while len(batches[-1]) % torch.cuda.device_count() != 0:
                batches[-1].append(dummy_example)
                num_dummies += 1
    # Run evaluation.
    model.eval()
    with torch.no_grad():
        eval_logits = []
        for batch_i in range(len(batches)):
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer, model_type)
            # Run model.
            if model_type == "bert" or model_type == "gpt2": # Transformers
                outputs = model(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                labels=inputs["labels"],
                                output_hidden_states=False, return_dict=True)
                logits = outputs['logits'].detach()
            else: # LSTMS.
                outputs = model(inputs["input_ids"], inputs["labels"],
                                hidden=None, loss_only=False)
                logits = outputs[1].detach()
            # Get logit indices whose corresponding label is masked.
            if model_type == "gpt2":
                logits = logits[:, :-1, :] # All batches, remove last prediction, all vocab.
                inputs["labels"] = inputs["labels"][:, 1:]
            # Now, logits correspond to labels.
            target_indices = inputs["labels"] == tokenizer.mask_token_id
            # Get logits at the target indices.
            # Initial logits had shape: batch_size x seq_len x vocab_size.
            # Output shape: n_masks x vocab_size. n_masks should equal batch_size.
            mask_logits = logits[target_indices, :]
            eval_logits.append(mask_logits.detach().cpu()) # Send to CPU so not all need to be held on GPU.
    # Remove dummy examples from the last batch.
    if num_dummies > 0:
        eval_logits[-1] = eval_logits[-1][:-num_dummies, :]
    # Logits output shape: num_examples x vocab_size.
    all_eval_logits = torch.cat(eval_logits, dim=0)
    if all_eval_logits.shape[0] != len(examples):
        # This can happen if there is not exactly one mask per example.
        # For example, if the last token in the sequence is masked, then the bidirectional LSTM
        # does not make a prediction for the masked token.
        print("WARNING: length of logits {0} does not equal number of examples {1}!!".format(
            all_eval_logits.shape[0], len(examples)
        ))
    return all_eval_logits


# Run token evaluations for a single model.
def evaluate_tokens(model, model_type, token_data, tokenizer, outfile,
                    curr_steps, batch_size, min_samples):
    token_count = 0
    for token, token_id, sample_sents in token_data:
        print("\nEvaluation token: {}".format(token))
        token_count += 1
        print("{0} / {1} tokens".format(token_count, len(token_data)))
        print("CHECKPOINT STEP: {}".format(curr_steps))
        num_examples = len(sample_sents)
        print("Num examples: {}".format(num_examples))
        if num_examples < min_samples:
            print("Not enough examples; skipped.")
            continue
        # Get logits with shape: num_examples x vocab_size.
        logits = run_model(model, model_type, sample_sents, batch_size, tokenizer)
        print("Finished inference.")
        probs = torch.nn.Softmax(dim=-1)(logits)
        # Get median rank of correct token.
        rankings = torch.argsort(probs, axis=-1, descending=True)
        ranks = torch.nonzero(rankings == token_id) # Each output row is an index (sentence_i, token_rank).
        ranks = ranks[:, 1] # For each example, only interested in the rank (not the sentence index).
        median_rank = torch.median(ranks).item()
        # Get accuracy.
        predictions = rankings[:, 0] # The highest rank token_ids.
        accuracy = torch.mean((predictions == token_id).float()).item()
        # Get mean/stdev surprisal.
        token_probs = probs[:, token_id]
        token_probs += 0.000000001 # Smooth with (1e-9).
        surprisals = -1.0*torch.log2(token_probs)
        mean_surprisal = torch.mean(surprisals).item()
        std_surprisal = torch.std(surprisals).item()
        # Logging.
        print("Median rank: {}".format(median_rank))
        print("Mean surprisal: {}".format(mean_surprisal))
        print("Stdev surprisal: {}".format(std_surprisal))
        print("Accuracy: {}".format(accuracy))
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(
            curr_steps, token, median_rank, mean_surprisal, std_surprisal,
            accuracy, num_examples))
    return


# Load one model.
def load_single_model(single_model_dir, model_type, config, tokenizer):
    print("Loading from directory: {}".format(single_model_dir))
    if model_type == "gpt2": # GPT2LMHeadModel.
        model = AutoModelForCausalLM.from_pretrained(
            single_model_dir,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
    elif model_type == "bert": # BertForMaskedLM.
        model = AutoModelForMaskedLM.from_pretrained(
            single_model_dir,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
    elif model_type == "lstm":
        model = RNNLanguageModel.from_config(len(tokenizer), config)
    elif model_type == "bilstm":
        model = BidirectionalRNNLanguageModel.from_config(len(tokenizer), config)
    # LSTM from checkpoint.
    if model_type == "lstm" or model_type == "bilstm":
        checkpoint = torch.load(os.path.join(single_model_dir, "checkpoint.pt"))
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            # The DataParallel module might have been saved.
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.module # Remove data parallel wrapper.
    # Load onto GPU.
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


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
    bidirectional = (args.model_type ==  "bert" or args.model_type == "bilstm")

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


    # Get the tokens to consider, and the corresponding sample sentences.
    print("Getting sample sentences for tokens.")
    if args.save_samples != "" and os.path.isfile(args.save_samples):
        print("Loading sample sentences from file.")
        token_data = pickle.load(open(args.save_samples, "rb"))
    else: # save_samples is empty or file does not exist.
        token_data = get_sample_sentences(
            tokenizer, args.wordbank_file, args.wordbank_lang, args.examples_file,
            max_seq_len, args.min_seq_len, args.max_samples, bidirectional=bidirectional,
            inflections=args.inflections)
        if args.save_samples != "":
            pickle.dump(token_data, open(args.save_samples, "wb"))

    # Prepare for evaluation.
    outfile = codecs.open(args.output_file, 'w', encoding='utf-8')
    # File header.
    outfile.write("Steps\tToken\tMedianRank\tMeanSurprisal\tStdevSurprisal\tAccuracy\tNumExamples\n")
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

    # Run evaluation.
    for checkpoint in checkpoints:
        print("CHECKPOINT STEPS: {}".format(checkpoint))
        single_model_dir = os.path.join(args.model_dir, "checkpoint-{}".format(checkpoint))
        model = load_single_model(single_model_dir, args.model_type, config, tokenizer)
        evaluate_tokens(model, args.model_type, token_data, tokenizer, outfile,
                        checkpoint, args.batch_size, args.min_samples)
    outfile.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
