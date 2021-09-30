"""
Parse a training log outputted by Transformer language modeling (e.g. trainer_state.json),
or a training log outputted by RNN language modeling (e.g. log.txt). Outputs
the eval loss at each eval step.
"""

from __future__ import unicode_literals

import codecs
import argparse
import json

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output', default='output.txt')
    # transformer or lstm.
    parser.add_argument('--model_type', default='transformer')
    return parser

def main(input, output, model_type):
    with codecs.open(input, "rb", encoding="utf-8") as reader:
        text = reader.read()
    # Map steps to eval loss.
    eval_results = dict()

    if model_type == "transformer" or model_type == "bert" or model_type == "gpt2":
        parsed_json = json.loads(text)
        history = parsed_json["log_history"]
        for log in history:
            if "eval_loss" in log and "step" in log:
                eval_results[log["step"]] = log["eval_loss"]
    else: # LSTM.
        text = text.split('\n')
        for line in text:
            if not "Eval loss" in line:
                continue
            split_line = line.split('\t')
            step = int(split_line[0].split()[-1])
            eval_loss = float(split_line[1].split()[-1])
            eval_results[step] = eval_loss

    # Write the output.
    outfile = codecs.open(output, 'w', encoding='utf-8')
    outfile.write("Step\tEvalLoss\n")
    for step, eval_loss in eval_results.items():
        outfile.write("{0}\t{1}\n".format(step, eval_loss))
    outfile.close()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input, args.output, args.model_type)
