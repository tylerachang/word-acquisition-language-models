# Utils for Transformer and RNN language modeling.

import numpy as np

from dataclasses import dataclass, field
from typing import Optional
from transformers import PreTrainedTokenizer
from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback

from dataset_classes import (
    LineByLineTextDataset,
    IterableTextDataset,
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Config name or path if not the same as model_name; should be a bert or gpt2 config"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    override_vocabsize: int = field(
        default=-1,
        metadata={"help": "Can override the vocab size in the model config. Otherwise, sets to the tokenizer length."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_iterable: bool = field(
        default=False,
        metadata={"help": "Whether to use an iterable training dataset as input."
                          "If so, train_data_file should be a text file where each line is a string of space-separated token indices."
                          "Data should be pre-shuffled."
                          "If false, a line_by_line text dataset is assumed."},
    )
    eval_iterable: bool = field(default=False)
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    overwrite_save_strategy: str = field(
        default="",
        metadata={"help": "Can set to exponential save strategy (see run_transformer_language_modeling.py)."},
    )
    override_n_examples: int = field(
        default=-1,
        metadata={"help": "Can set this to avoid having to count through the training dataset at the beginning."},
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    is_iterable: bool = False,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    n_examples = -1 if evaluate else args.override_n_examples
    if is_iterable:
        return IterableTextDataset(file_path=file_path, block_size=args.block_size,
                                   pad_token_id=tokenizer.pad_token_id, sep_token_id=tokenizer.sep_token_id,
                                   n_examples=n_examples)
    else:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)


"""
Callback to overwrite the save steps.
E.g. to save every exp_base^(scalar*n)+constant steps.
This should be passed through the trainer callbacks parameter.
One of the following must be called after initialization:
set_checkpoint_rates()
"""
class SaveStepsCallback(TrainerCallback):
    def __init__(self):
        # This should be set with one of the functions below.
        self.checkpoint_steps = None

    def get_steps_from_rates(self, s0, s1, t1, max_steps):
        def get_step(n):
            term1 = s0 * t1 / (s1 - s0)
            exponent = n * (s1 - s0) / t1
            term2 = np.exp(exponent) - 1.0
            return term1 * term2
        save_steps = []
        save_step = 0
        n = 0
        while save_step <= max_steps:
            save_steps.append(save_step)
            # Increment.
            n += 1
            save_step = int(np.round(get_step(n), 0))
        return save_steps

    # Set checkpoint steps such that there are s0 steps per save at step 0,
    # and s1 steps per save at step t1. This results in an exponential function
    # for steps vs. saves.
    def set_checkpoint_rates(self, s0, s1, t1, max_steps):
        self.checkpoint_steps = self.get_steps_from_rates(s0, s1, t1, max_steps)

    # Function override.
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.checkpoint_steps is None:
            print("Warning: checkpoint steps not initialized for SaveStepsCallback.")
            return control
        is_save_step = self.checkpoint_steps and state.global_step in self.checkpoint_steps
        # Evaluate on these steps.
        if is_save_step and args.do_eval:
            control.should_evaluate = True
        # Save only on these steps.
        if is_save_step:
            control.should_save = True
        else:
            control.should_save = False
        return control
