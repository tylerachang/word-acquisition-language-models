"""
Tokenize a text file given a tokenizer.
Creates a text file where each line is a space-separated list of token ids.
Adds [CLS] and [SEP] tokens, but no padding.
Sample usage:

python3 scripts/tokenize_dataset.py \
--tokenizer="./sample_data/spm.model" \
--input_file="./sample_data/train_text.txt" \
--output_file="./sample_data/train_tokenized.txt" \
--max_examples=10000000 \
--max_segments=-1 --max_seq_len=128 \
--prepend_cls=True --include_sep=True

"""

import argparse
import os
import codecs
import math
from transformers import (
    AutoTokenizer,
    AlbertTokenizer,
)

# To process lines in batches.
MAX_STORED_LINE_COUNT = 10000


def create_parser():
    parser = argparse.ArgumentParser()
    # The tokenizer name or path, corresponding to either a SentencePiece
    # tokenizer or model_name_or_path from Huggingface.
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    # Maximum number of examples.
    parser.add_argument("--max_examples", type=int, default=-1)
    # Maximum number of segments (input lines) per example.
    # I.e. how many lines to concatenate in each example.
    parser.add_argument("--max_segments", type=int, default=-1)
    # Maximum number of tokens per example.
    # E.g. BERT has maximum sequence length 512.
    # Models will automatically truncate long examples, so it is better to
    # be slightly too long.
    # Examples will be unpadded.
    parser.add_argument("--max_seq_len", type=int, default=512)
    # Special tokens. Using str type because bool sometimes parses
    # unintuitively from the args.
    parser.add_argument("--prepend_cls", type=str, default="True")
    parser.add_argument("--include_sep", type=str, default="True")
    return parser


def tokenize_file(input_path, output_path, tokenizer, max_seq_len,
                  max_examples=math.inf, max_segments=math.inf,
                  prepend_cls=True, include_sep=True):
    print("Tokenizing file: {}".format(input_path))
    # Prepare special tokens.
    cls_token_id = None
    sep_token_id = None
    if prepend_cls:
        cls_token_id = tokenizer.cls_token_id
        if cls_token_id is None:
            print("Warning: [CLS] token does not exist.")
    if include_sep:
        sep_token_id = tokenizer.sep_token_id
        if sep_token_id is None:
            print("Warning: [SEP] token does not exist.")
    # Prepare files.
    if os.path.isfile(output_path):
        print("File already exists: {}".format(output_path))
        return
    infile = codecs.open(input_path, "rb", encoding="utf-8")
    outfile = codecs.open(output_path, "wb", encoding="utf-8")
    example_count = 0
    line_count = 0
    stored_lines = []
    # Sub-function to tokenize a batch of inputs.
    def tokenize_batch():
        nonlocal stored_lines, example_count, max_examples, max_seq_len, sep_token_id, cls_token_id, outfile
        curr_example = [] if cls_token_id is None else [cls_token_id]
        curr_n_segments = 0
        batch_encoding = tokenizer(stored_lines, add_special_tokens=False,
                truncation=True, max_length=max_seq_len)
        for tokenized_line in batch_encoding["input_ids"]:
            curr_example = curr_example + tokenized_line
            if sep_token_id is not None: curr_example.append(sep_token_id)
            curr_n_segments += 1
            if len(curr_example) >= max_seq_len or curr_n_segments >= max_segments:
                # Process an example.
                curr_example = curr_example[:max_seq_len]
                # Note that these examples are unpadded.
                outfile.write(" ".join([str(token_id) for token_id in curr_example]))
                outfile.write("\n")
                curr_example = [] if cls_token_id is None else [cls_token_id]
                curr_n_segments = 0
                example_count += 1
                if example_count >= max_examples:
                    print("Finished tokenization.")
                    return True  # Return True if completely finished.
        stored_lines = []
        return False
    # Tokenize lines.
    for line in infile:
        line_count += 1
        stripped_line = line.strip()
        if stripped_line != "":
            stored_lines.append(stripped_line)
        # Process the currently stored lines.
        if line_count % MAX_STORED_LINE_COUNT == 0:
            completed = tokenize_batch()  # This also resets the stored_lines.
            print("Processed up to line {0} ({1} examples)".format(line_count, example_count))
            if completed: break
    # Process the remaining set of lines.
    if len(stored_lines) > 0:
        tokenize_batch()
    outfile.close()
    infile.close()
    print("Finished tokenization: {} examples.".format(example_count))
    return


def main(args):
    tokenizer_cache_dir = "tokenizer_cache"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=tokenizer_cache_dir)
    except:
        # If passing in a raw tokenizer model file, assume ALBERT sentencepiece model.
        print("Attempting to use local sentencepiece model file as tokenizer.")
        tokenizer = AlbertTokenizer.from_pretrained(args.tokenizer)

    max_examples = math.inf if args.max_examples == -1 else args.max_examples
    max_segments = math.inf if args.max_segments == -1 else args.max_segments
    # Cannot input math.inf to the tokenizer, so just use a large number.
    max_seq_len = 999999999 if args.max_seq_len == -1 else args.max_seq_len

    prepend_cls = args.prepend_cls.lower() == "true"
    include_sep = args.include_sep.lower() == "true"
    print(f"Prepending CLS: {prepend_cls}, using SEP: {include_sep}")

    tokenize_file(args.input_file, args.output_file, tokenizer, max_seq_len,
                  max_examples=max_examples, max_segments=max_segments,
                  prepend_cls=prepend_cls, include_sep=include_sep)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
