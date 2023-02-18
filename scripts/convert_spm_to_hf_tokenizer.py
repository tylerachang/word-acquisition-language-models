"""
Convert the tokenizer from train_spm_tokenizer.py to a Hugging Face tokenizer.
Sets parameters such as lowercasing (no lowercasing for SentencePiece).
Otherwise, Hugging Face will assume lowercasing by default, and many of the
SPM tokens will be unused. Also adds special tokens such that the tokenizer
vocab size is a multiple of multiple_of (ideally a power of 2).
Sample usage:

python3 scripts/convert_spm_to_hf_tokenizer.py \
--input="./sample_data/spm.model" \
--output_dir="./sample_data/hf_tokenizer" \
--multiple_of=2048

"""

import argparse
from transformers import AlbertTokenizer

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--do_lower_case', type=bool, default=False)
    parser.add_argument('--keep_accents', type=bool, default=False)
    parser.add_argument('--multiple_of', type=int, default=256)
    return parser

def main(args):
    print("Using local sentencepiece model file as tokenizer.")
    tokenizer = AlbertTokenizer.from_pretrained(args.input,
            do_lower_case=args.do_lower_case, keep_accents=args.keep_accents)
    # Add tokens until a multiple of 512.
    special_tokens_list = []
    if len(tokenizer) % args.multiple_of != 0:
        n_to_add = args.multiple_of - (len(tokenizer) % args.multiple_of)
    for add_i in range(n_to_add):
        special_tokens_list.append("[XXXXX{}]".format(add_i))
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("{} tokens added.".format(n_to_add))
    print("Final vocab size: {} (this should be set in the language model configs).".format(len(tokenizer)))
    # Save tokenizer.
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
