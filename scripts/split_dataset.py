"""
Splits each file in a directory into train, eval, and test.
Outputs files to the directory: [dataset_dir]_split
Sample usage:

python3 split_datasets.py --dataset_dir="tokenized" \
--train_proportion=0.80 --eval_proportion=0.20 --test_proportion=0.0

"""

import argparse
import os
import codecs
import numpy as np
from tqdm import tqdm
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="The directory containing the input files.")
    parser.add_argument("--train_proportion", type=float, required=True)
    parser.add_argument("--eval_proportion", type=float, required=True)
    parser.add_argument("--test_proportion", type=float, required=True)
    return parser.parse_args()


def split_file(dataset_dir, filename, train_proportion, eval_proportion, test_proportion):
    inpath = os.path.join(dataset_dir, filename)
    filename_prefix = filename.replace(".txt", "")
    os.makedirs(dataset_dir + "_split", exist_ok=True)
    train_outpath = os.path.join(dataset_dir + "_split", filename_prefix + "_train.txt")
    eval_outpath = os.path.join(dataset_dir + "_split", filename_prefix + "_eval.txt")
    test_outpath = os.path.join(dataset_dir + "_split", filename_prefix + "_test.txt")

    print("Counting lines in infile.")
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    line_count = 0
    for line in tqdm(infile):
        line_count += 1
    infile.close()

    # Choose splits.
    n_test = int(test_proportion * line_count)
    n_eval = int(eval_proportion * line_count)
    n_train = int(train_proportion * line_count)
    # Due to rounding.
    if n_train+n_eval+n_test>line_count or math.isclose(train_proportion+eval_proportion+test_proportion, 1.0):
        n_train = line_count - n_eval - n_test
    # Choose indices.
    shuffled_indices = np.arange(line_count, dtype=int)
    np.random.shuffle(shuffled_indices)
    train_indices = iter(np.sort(shuffled_indices[:n_train]))
    eval_indices = iter(np.sort(shuffled_indices[n_train:n_train+n_eval]))
    test_indices = iter(np.sort(shuffled_indices[n_train+n_eval:n_train+n_eval+n_test]))

    train_next = next(train_indices, -1)
    eval_next = next(eval_indices, -1)
    test_next = next(test_indices, -1)

    print("Writing outputs.")
    train_out = codecs.open(train_outpath, 'wb', encoding='utf-8')
    eval_out = codecs.open(eval_outpath, 'wb', encoding='utf-8')
    test_out = codecs.open(test_outpath, 'wb', encoding='utf-8')
    # Write outputs.
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    for index, line in tqdm(enumerate(infile)):
        if index == train_next:
            train_out.write(line.strip() + "\n")
            train_next = next(train_indices, -1)
        elif index == eval_next:
            eval_out.write(line.strip() + "\n")
            eval_next = next(eval_indices, -1)
        elif index == test_next:
            test_out.write(line.strip() + "\n")
            test_next = next(test_indices, -1)
    infile.close()
    train_out.close()
    eval_out.close()
    test_out.close()
    return True


def main(args):
    assert args.train_proportion + args.eval_proportion + args.test_proportion <= 1.0, \
        "train/eval/test proportions should sum <= 1.0."

    for filename in os.listdir(args.dataset_dir):
        if not filename.endswith(".txt"):
            continue
        print("Splitting file: {}".format(filename))
        split_file(args.dataset_dir, filename, args.train_proportion,
                   args.eval_proportion, args.test_proportion)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
