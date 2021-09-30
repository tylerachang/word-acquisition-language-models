"""
Creates a vocab_size x vocab_size bigram matrix from the text file of tokenized
examples. Each line should be a space-separated list of integer token ids.
"""

import codecs
import torch
import argparse
import sentencepiece as spm

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="")
    parser.add_argument('--tokenizer', default="")
    parser.add_argument('--output_file', default="output.txt")
    parser.add_argument('--max_seq_len', type=int, default=128)
    return parser

def main(input_file, tokenizer, output_file, max_seq_len):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer)
    vocab_size = len(sp) + 4 # UNK, CLS, SEP, MASK.
    bigrams = torch.zeros(vocab_size, vocab_size).int()
    example_count = 0
    infile = codecs.open(input_file, 'rb', encoding='utf-8')
    for line in infile:
        tokens = [int(token) for token in line.strip().split()]
        for i in range(1, len(tokens)):
            if i >= max_seq_len:
                break
            prefix_token = tokens[i-1]
            if prefix_token >= 0 and prefix_token < vocab_size:
                bigrams[prefix_token, tokens[i]] += 1
            else:
                print("Token out of range!")
        example_count += 1
        if example_count % 100000 == 0:
            print("Read {0} examples".format(example_count))
    print('TOTAL EXAMPLES: {}'.format(example_count))
    infile.close()
    torch.save(bigrams, output_file)
    print("Saved.")

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input_file, args.tokenizer, args.output_file, args.max_seq_len)
