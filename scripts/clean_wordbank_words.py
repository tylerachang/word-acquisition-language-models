"""
Clean words (convert to UTF-8) from the WordBank dataset.
Run on the smooth AoAs computed in R (get_child_aoa.R).
Cleans the raw UTF-8 bytes for each token, which can get offset.
Adds two columns: CleanedWord and CleanedSingle (corresponding to the cleaned
token, and the cleaned token in a single canonical form).
"""

from __future__ import unicode_literals

import codecs
import argparse
import re

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="")
    parser.add_argument('--output_file', default="output.txt")
    return parser

def main(input_file, output_file):
    infile = codecs.open(input_file, 'rb', encoding='utf-8')
    outfile = codecs.open(output_file, 'w', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count == 0: # Header.
            outfile.write("Index\t{}\tCleanedWord\tCleanedSingle\n".format(line.strip()))
            continue
        split_line = line.strip().split('\t')
        word_bytes = split_line[13]
        word_bytes = word_bytes.replace(" ", "").lower()
        num_replacements = 0
        while True:
            try:
                cleaned_field = bytes.fromhex(word_bytes).decode('utf-8')
                if num_replacements > 0:
                    print("Num replacements: {}".format(num_replacements))
                # Clean the word to get a single form.
                cleaned_single = cleaned_field.strip().split()[0].lower()
                if cleaned_single[0] == '(': # Remove possible prefix.
                    cleaned_single = cleaned_single[cleaned_single.find(")")+1:]
                # Only consider first form.
                cleaned_single = re.split('[.,(/*#\uFF08]', cleaned_single)[0]
                break
            except UnicodeDecodeError as e:
                error_char = word_bytes[2*e.start]
                # For some reason, the first char in the byte is often 2 values too large.
                adjusted = int(error_char, base=16)-2
                if adjusted < 0:
                    # Remove the character and the following character (removes
                    # the entire byte).
                    if len(word_bytes) > 2*e.start+2:
                        word_bytes = word_bytes[:2*e.start] + word_bytes[2*e.start+2:]
                    else:
                        word_bytes = word_bytes[:2*e.start]
                else:
                    new_char = format(adjusted, 'x')
                    word_bytes = word_bytes[:2*e.start] + new_char + word_bytes[2*e.start+1:]
                num_replacements += 1
        outfile.write('{0}\t{1}\t{2}\n'.format("\t".join(split_line), cleaned_field, cleaned_single))
    infile.close()
    outfile.close()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input_file, args.output_file)
