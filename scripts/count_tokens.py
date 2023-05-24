"""
Counts tokens in a text file of tokenized examples.
Each line should be a space-separated list of integer token ids.
Also computes token frequencies and mean sequence lengths.
Sample usage:

python3 count_tokens.py \
--tokenizer="./sample_data/spm.model" \
--input_file="./sample_data/train_tokenized.txt" \
--output_file="./sample_data/lm_data_stats.txt" \
--max_seq_len=512

"""

from __future__ import unicode_literals

import codecs
import argparse
import sentencepiece as spm
from transformers import AutoTokenizer
from collections import Counter

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="")
    parser.add_argument('--tokenizer', default="")
    parser.add_argument('--output_file', default="output.txt")
    parser.add_argument('--max_seq_len', type=int, default=512)
    return parser

def main(input_file, tokenizer, output_file, max_seq_len):
    # Load words.
    if tokenizer.endswith('.model'):
        # SentencePiece model.
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer)
        tokens = dict()
        for token_id in range(len(sp)):
            token = sp.id_to_piece(token_id)
            tokens[token_id] = token
        vocab_size = len(sp)
    else:
        # Hugging Face tokenizer.
        tok = AutoTokenizer.from_pretrained(tokenizer, cache_dir='hf_cache')
        tokens = dict()
        for token_id in range(len(tok)):
            token = tok.decode(token_id)
            tokens[token_id] = token
        vocab_size = len(tok) # Including special tokens.

    # Count tokens.
    infile = codecs.open(input_file, 'rb', encoding='utf-8')
    token_counts = Counter() # Count tokens.
    seq_len_total = Counter() # Total seq_len for each token.
    uni_seq_len_total = Counter() # Only counting previous tokens.
    total_tokens = 0
    for line_count, line in enumerate(infile):
        if line_count % 100000 == 0:
            print("Finished line {}".format(line_count))
        example_string = line.strip()
        example_pair = [int(token_id) for token_id in example_string.split()]
        if len(example_pair) > max_seq_len:
            example_pair = example_pair[:max_seq_len]
        for idx, token_id in enumerate(example_pair):
            if token_id >= vocab_size:
                continue
            token_counts[tokens[token_id]] += 1
            seq_len_total[tokens[token_id]] += len(example_pair)
            uni_seq_len_total[tokens[token_id]] += idx+1 # Unidirectional length.
            total_tokens += 1
    infile.close()

    outfile = codecs.open(output_file, 'w', encoding='utf-8')
    outfile.write('Token\tCountPerThousand\tMeanSeqLen\tUnidirectionalMeanSeqLen\n')
    for token, count in token_counts.items():
        per_thousand = count*1000.0/total_tokens
        mean_seq_len = seq_len_total[token]*1.0/count
        uni_mean_seq_len = uni_seq_len_total[token]*1.0/count
        outfile.write('{0}\t{1}\t{2}\t{3}\n'.format(token, per_thousand, mean_seq_len, uni_mean_seq_len))
    outfile.close()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input_file, args.tokenizer, args.output_file, args.max_seq_len)
