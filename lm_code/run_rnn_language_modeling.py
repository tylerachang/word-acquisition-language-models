"""
Train an RNN language model (unidirectional or bidirectional).
See readme for sample usage.
"""

import torch
from torch import nn
import os
import json
from codecs import open
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import (
    AutoTokenizer,
    AlbertTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AdamW,
)
from transformers.optimization import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint

from rnn_models import (
    RNNLanguageModel,
    BidirectionalRNNLanguageModel,
)
from lm_utils import (
    ModelArguments,
    DataTrainingArguments,
    get_dataset,
)
from dataset_classes import DataCollatorForLanguageModeling

def format_inputs(inputs, max_seq_len, bidirectional=False):
    # Format the inputs from the DataCollatorForLanguageModeling.
    # Note: inputs will have shape batch_size x seq_len.
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    if input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        labels = labels[:, :max_seq_len]
    if bidirectional:
        # Remove the first and last label (no corresponding predictions from
        # either forward or backward).
        labels = labels[:,1:-1]
        # Forward and backward inputs.
        # Remove last two forward inputs: last one has no corresponding label, and
        # second to last one corresponds to a label with no backwards prediction.
        input_ids = (input_ids[:,:-2], input_ids[:,2:])
    else:
        # Remove the first label (not predicted by forward) and the last
        # input (no corresponding label).
        labels = labels[:,1:] # The first token is not a label.
        input_ids = input_ids[:,:-1] # The last token is not inputted.
    if torch.cuda.is_available():
        if bidirectional:
            input_ids = tuple([input.cuda() for input in input_ids])
        else:
            input_ids = input_ids.cuda()
        labels = labels.cuda()
    return { "input_ids": input_ids, "labels": labels }

def evaluate(model, eval_dataloader, vocab_size, max_seq_len, training_step, log_file, bidirectional):
    # Evaluate the model on the development set.
    # Metrics output to the log file.
    model.eval()
    total_loss = 0.0
    steps = 0
    for inputs in eval_dataloader:
        with torch.no_grad():
            inputs = format_inputs(inputs, max_seq_len, bidirectional=bidirectional)
            loss = model(inputs["input_ids"], inputs["labels"],
                         hidden=None, loss_only=True)[0]
            total_loss += loss.mean().detach()
            steps += 1
    loss = float(total_loss) / steps
    log_file.write("Step: {0}\tEval loss: {1}\n".format(training_step, loss))
    return

def main():
    # See all possible training arguments in src/transformers/training_args.py in the
    # Huggingface Transformers repository, or by passing the --help flag to this script.
    # Supported training args for RNNs are:
    # do_eval, do_train, output_dir, overwrite_output_dir, no_cuda, seed,
    # learning_rate, adam_beta1, adam_beta2, adam_epsilon, warmup_steps, max_steps,
    # logging_steps, n_gpu, train_batch_size, dataloader_drop_last, dataloader_num_workers,
    # max_grad_norm, eval_steps, eval_batch_size, save_steps
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
                f"Output directory ({training_args.output_dir}) already exists and contains no checkpoints. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print("Checkpoint detected, resuming training at {}.".format(last_checkpoint))
    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Load config.
    config_save_path = os.path.join(training_args.output_dir, "config.json")
    if model_args.config_name:
        config_path = model_args.config_name
    else:
        # Try to find config in the output directory.
        config_path = config_save_path
    # Open config.
    with open(config_path, "rb", encoding="utf-8") as reader:
        text = reader.read()
    # Write a copy of the config to the output_dir.
    if config_path != config_save_path:
        with open(config_save_path, "wb", encoding="utf-8") as writer:
            writer.write(text)
    # Note: config should have fields:
    # rnn_cell, embedding_size, hidden_sizes, dropout, emb_tied, rnn_mode, max_seq_len
    config = json.loads(text)
    rnn_cell = config["rnn_cell"] # LSTM or GRU
    embedding_size = config["embedding_size"] # int
    dropout = config["dropout"] # float
    emb_tied = config["emb_tied"] # standard or empty
    rnn_mode = config["rnn_mode"] # bidirectional or forward
    hidden_sizes = config["hidden_sizes"] # list of ints
    max_seq_len = config["max_seq_len"] # int

    # Load tokenizer.
    if model_args.tokenizer_name:
        tokenizer_path = model_args.tokenizer_name
    else:
        print("tokenizer_name (path to tokenizer) must be provided.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=model_args.cache_dir)
    except:
        # If passing in a raw tokenizer model file, assume ALBERT sentencepiece model.
        print("Attempting to use local sentencepiece model file as tokenizer.")
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer) # Note that this includes special tokens.

    # Load models.
    if model_args.model_name_or_path:
        print("Loading model from directory.")
        model_path = model_args.model_name_or_path
        if "pytorch_model.pt" not in model_path:
            model_path = os.path.join(model_path, "pytorch_model.pt")
        model = torch.load(model_path)
    else:
        print("Training model from scratch.")
        bidirectional = rnn_mode == "bidirectional"
        if bidirectional:
            model = BidirectionalRNNLanguageModel(vocab_size, embedding_size, hidden_sizes, dropout,
                                               rnn_cell, fixed_embs=False, tied=emb_tied)
        else: # Simple forward RNN.
            model = RNNLanguageModel(vocab_size, embedding_size, hidden_sizes, dropout,
                                     rnn_cell, fixed_embs=False, tied=emb_tied)
    # Move model to GPU before instantiating optimizers. Then, optimizers will
    # load parameters onto GPU.
    if torch.cuda.is_available():
        if training_args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(training_args.seed)
            model.cuda()
    # Print total model parameters.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUM TRAINABLE MODEL PARAMETERS: {}".format(num_params))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    # Load from checkpoint if possible.
    training_step = 0
    if last_checkpoint is not None:
        print("Loading model from checkpoint.")
        checkpoint = torch.load(os.path.join(last_checkpoint, "checkpoint.pt"))
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            # Backwards compatibility: the DataParallel module might have been saved.
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        training_step = checkpoint['training_step']
        if training_step % training_args.logging_steps != 0:
            print("Initial training loss logging may be strange when starting from this "
                  "checkpoint, because logging_steps does not divide the checkpoint steps.")
    if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        # Note: batch is dimension 0 (seq_len is dimension 1).
        model = torch.nn.DataParallel(model)

    # Get datasets.
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, is_iterable=data_args.train_iterable)
        if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, is_iterable=data_args.eval_iterable)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.train_batch_size, # Automatically computed from n_gpus and per_device batch size.
        sampler=None if data_args.train_iterable else SequentialSampler(train_dataset),
        collate_fn=data_collator,
        drop_last=training_args.dataloader_drop_last,
        num_workers=training_args.dataloader_num_workers,
    )

    # Handle epochs and steps.
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = training_args.max_steps // num_update_steps_per_epoch + int(
        training_args.max_steps % num_update_steps_per_epoch > 0
    )
    epochs_trained = training_step // num_update_steps_per_epoch
    steps_trained_in_current_epoch = training_step % num_update_steps_per_epoch

    # Train.
    print("Note: {} steps per epoch.".format(num_update_steps_per_epoch))
    optimizer.zero_grad()
    log_filepath = os.path.join(training_args.output_dir, "log.txt")
    log_file = open(log_filepath, 'ab', encoding='utf-8')
    total_loss = 0.0
    for epoch in range(epochs_trained, num_train_epochs):
        print("Starting epoch {}.".format(epoch))
        for step, inputs in enumerate(train_dataloader):
            # Skip previously trained steps.
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            if training_step >= training_args.max_steps:
                break

            # Step.
            model.train()
            optimizer.zero_grad()
            inputs = format_inputs(inputs, max_seq_len, bidirectional=bidirectional)
            # Pass through model.
            loss = model(inputs["input_ids"], inputs["labels"],
                         hidden=None, loss_only=True)[0]
            if training_args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            total_loss += loss.detach().item()
            # Gradient clipping.
            if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_args.max_grad_norm,
                )
            optimizer.step()
            lr_scheduler.step()
            training_step += 1

            # Log training loss, run and log eval, save checkpoint.
            if training_step % training_args.logging_steps == 0 and training_step != 0:
                # Log training loss.
                print("Logging step {}".format(training_step))
                loss = float(total_loss) / training_args.logging_steps
                total_loss = 0.0
                log_file.write("Step: {0}\tTrain loss: {1}\n".format(training_step, loss))
            if training_step % training_args.eval_steps == 0 and training_step != 0:
                print("Evaluating step {}".format(training_step))
                eval_dataloader = DataLoader(
                    eval_dataset,
                    sampler=None if data_args.eval_iterable else SequentialSampler(eval_dataset),
                    batch_size=training_args.eval_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                )
                evaluate(model, eval_dataloader, vocab_size, max_seq_len,
                         training_step, log_file, bidirectional=bidirectional)
            if training_step % training_args.save_steps == 0 and training_step != 0:
                # Save a model checkpoint.
                outdir = os.path.join(training_args.output_dir,
                    "checkpoint-{}".format(training_step))
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                outpath = os.path.join(outdir, "checkpoint.pt")
                if isinstance(model, torch.nn.DataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                torch.save({
                    'training_step': training_step,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    }, outpath)
                print("Saved step {}".format(training_step))

    # Do this at end:
    log_file.close()
    model_outpath = os.path.join(training_args.output_dir, "pytorch_model.pt")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model, model_outpath)


if __name__ == "__main__":
    main()
