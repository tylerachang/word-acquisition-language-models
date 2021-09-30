# Modified from the huggingface/transformers/examples/language_modeling directory.
"""
Train a transformer language model.
Use for BERT or GPT2 pre-training.
See readme for sample usage.
"""

import logging
import math
import os
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AlbertTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction

from dataset_classes import (
    DataCollatorForLanguageModeling,
)
from lm_utils import (
    ModelArguments,
    DataTrainingArguments,
    get_dataset,
)

logger = logging.getLogger(__name__)


# Compute evaluation raw accuracy.
# This can be passed into trainer.
# Unfortunately, this takes too much memory (because all logits are
# stored in eval_prediction).
def compute_metrics(eval_prediction: EvalPrediction):
    logits = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    if type(logits) == tuple:
        # First item should be token logits.
        logits = logits[0]
    # Compute accuracy.
    predictions = torch.argmax(logits, dim=2)
    del logits
    total_predicted = 0
    total_correct = 0
    for example in range(labels.shape[0]):
        for index in range(labels.shape[1]):
            if labels[example][index].item() == -100: # Not predicted.
                continue
            total_predicted += 1
            if labels[example][index] == predictions[example][index]:
                total_correct += 1
    accuracy = float(total_correct) / float(total_predicted)
    return {"accuracy": accuracy}


def main():
    # See all possible training arguments in src/transformers/training_args.py in the
    # Huggingface Transformers repository, or by passing the --help flag to this script.
    # Model args and data args are defined in lm_utils.py.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        logger.warning("Must provide model config file.")

    if model_args.tokenizer_name:
        tokenizer_path = model_args.tokenizer_name
    elif model_args.model_name_or_path:
        tokenizer_path = model_args.model_name_or_path
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=model_args.cache_dir)
    except:
        # If passing in a raw tokenizer model file, assume ALBERT sentencepiece model.
        print("Attempting to use local sentencepiece model file as tokenizer.")
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)

    # Overwrite special token ids in the configs.
    if config.model_type == "bert":
        config.pad_token_id = tokenizer.pad_token_id
    elif config.model_type == "gpt2":
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id

    # Load models.
    if model_args.model_name_or_path:
        if config.model_type == "gpt2": # GPT2LMHeadModel.
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else: # BertForMaskedLM.
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
    else:
        logger.info("Training new model from scratch")
        if config.model_type == "gpt2":
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Print total model parameters.
    # Note that usually output word embeddings are tied to input word embeddings (to save parameters).
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUM TRAINABLE MODEL PARAMETERS: {}".format(num_params))

    if config.model_type == "gpt2":
        data_args.block_size = config.n_positions
    else:
        data_args.block_size = config.max_position_embeddings

    # Get datasets.
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, is_iterable=data_args.train_iterable)
        if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=config.model_type=="bert",
        mlm_probability=data_args.mlm_probability,
    )

    # Initialize our Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None
    )

    # Training.
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation.
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
