"""
Train a SentencePiece tokenizer given a raw text file.
To train faster, randomly samples 10,000,000 lines of the training data.
Outputs the tokenizer files [output].model and [output].vocab. Other scripts
should refer to the tokenizer path [output].model.
Sample usage:

python3 scripts/train_spm_tokenizer.py \
--input_file="./sample_data/train_text.txt" \
--output="./sample_data/spm" \
--vocab_size=30000

"""

import argparse
import sentencepiece as spm

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--sample_size', type=int, default=10000000)
    parser.add_argument('--shuffle_input', default="true")
    return parser

def main(args):
    shuffle_input = args.shuffle_input == "true"
    spm.SentencePieceTrainer.train(input=args.input_file,
        model_prefix=args.output, vocab_size=args.vocab_size,
        input_sentence_size=args.sample_size,
        shuffle_input_sentence=shuffle_input)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
