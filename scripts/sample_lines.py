"""
Collects a random subset of lines from text files in a directory.
Outputs a text file of the sequences. Note that the lines are randomly selected,
but not shuffled.
Sample usage:

python3 sample_lines.py --input_dir="sample_data" \
--output_path="sampled_lines.txt" \
--output_line_counts="sample_output_line_counts.tsv" \
--input_line_counts=""

"""

import argparse
import os
import codecs
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="The directory containing the input files.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_line_counts", required=True,
                        help="A tsv of the desired output line count for each file.")
    parser.add_argument("--input_line_counts", default="",
                        help="An optional tsv of the input line count for each file. "
                             "Avoids having to count the lines in each input file.")
    return parser.parse_args()


"""
If inpath is a tsv, then reads line counts from the tsv, assuming each line is
a filename and line count separated by a tab character.
If inpath is a directory, then counts the lines for each file in the directory.
If nonempty, only the files in filenames_to_include will be considered.
Outputs a dictionary from filenames to line counts.
"""
def get_line_counts(inpath, filenames_to_include=[]):
    line_counts = dict()
    if os.path.isfile(inpath):
        # Assume inpath is a tsv.
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        for line in infile:
            if line.strip() == "":
                continue # Skip.
            # Assume each line contains the filename and line count.
            filename, line_count = tuple(line.strip().split("\t"))
            if len(filenames_to_include) == 0 or filename in filenames_to_include:
                line_counts[filename] = int(line_count)
        infile.close()
    elif os.path.isdir(inpath):
        # Read all files in the inpath directory.
        for filename in os.listdir(inpath):
            if ".txt" not in filename:
                continue # Skip.
            if len(filenames_to_include) > 0 and filename not in filenames_to_include:
                continue # Skip.
            # Count lines in the file.
            print("Counting lines in file: {}".format(os.path.join(inpath, filename)))
            filepath = os.path.join(inpath, filename)
            infile = codecs.open(filepath, 'rb', encoding='utf-8')
            line_count = 0
            for line in infile:
                line_count += 1
            infile.close()
            line_counts[filename] = line_count
    return line_counts


"""
Appends a subset of the lines in a file to the outpath.
"""
def append_lines_subset(inpath, infile_length, outpath, n_lines):
    if n_lines > infile_length:
        print("WARNING: attempting to sample {0} lines from file {1} with {2} "
              "lines; using all lines.".format(n_lines, inpath, infile_length))
        n_lines = infile_length
    indices = np.random.choice(infile_length, n_lines, replace=False)
    indices = iter(np.sort(indices))
    # Read input and write to output.
    outfile = codecs.open(outpath, 'ab', encoding='utf-8')
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    target_index = next(indices) # The first index to search for.
    for line_i, line in tqdm(enumerate(infile), total=infile_length):
        if line_i > infile_length:
            break # Do not read past the expected infile_length.
        if line_i == target_index:
            outfile.write(line.strip() + "\n")
            target_index = next(indices, -1)
    infile.close()
    outfile.close()
    return True


def main(args):
    if os.path.isfile(args.output_path):
        print("Output file already exists. Stopping.")
        return False
    print("Getting output line counts.")
    output_line_counts = get_line_counts(args.output_line_counts)
    output_filenames = list(output_line_counts.keys())
    if len(output_filenames) == 0:
        print("No output lines to write.")
        return False

    print("Getting input line counts.")
    if os.path.isfile(args.input_line_counts):
        # Get input line counts from the input tsv.
        input_line_counts = get_line_counts(args.input_line_counts, filenames_to_include=output_filenames)
    else:
        # Count lines for all files in the input directory.
        input_line_counts = get_line_counts(args.input_dir, filenames_to_include=output_filenames)

    print("Writing output.")
    for filename, output_line_count in output_line_counts.items():
        infile_length = input_line_counts[filename]
        inpath = os.path.join(args.input_dir, filename)
        print("Writing for file: {}".format(inpath))
        append_lines_subset(inpath, infile_length, args.output_path, output_line_count)
    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
