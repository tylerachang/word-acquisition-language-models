# Utils for Transformer and RNN language modeling.

from dataclasses import dataclass, field
from typing import Optional
from transformers import PreTrainedTokenizer

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


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    is_iterable: bool = False,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if is_iterable:
        return IterableTextDataset(file_path=file_path, block_size=args.block_size,
                                   pad_token_id=tokenizer.pad_token_id, sep_token_id=tokenizer.sep_token_id)
    else:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
