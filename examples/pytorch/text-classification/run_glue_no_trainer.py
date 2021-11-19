# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import torch
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from datetime import datetime
from pruner import Prune
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from torch.nn import CrossEntropyLoss, MSELoss


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    parser.add_argument('--jiant_prepruned_weight', type=str, default="")
    parser.add_argument("--eval_step", default=200, type=int, help="eval step.")
    
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--kd', action='store_true')
    parser.add_argument('--sift', action='store_true')
    parser.add_argument("--sift_version", default=0, type=int, help="1,2,3")
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--extract_mask', action='store_true')
    # parser.add_argument('--early_stop', action='store_true')
    # parser.add_argument('--early_stop_metric', type=str, default='acc', help='Early stop metric')
    parser.add_argument('--fixed_mask', default=None, type=str, help="Fixed mask path.")
    parser.add_argument('--mask', default=None, type=str, help="mask path")
    # parser.add_argument('--sparse_mnli_init', default=None, type=str, help="MNLI weight initialization.")
    parser.add_argument('--pruning_sparsity',type=float, default=0.875, help='sparsity')
    parser.add_argument("--current_step", default=0, type=int, help="current step.")
    parser.add_argument("--start_epoch", default=0, type=int, help="current step.")
    parser.add_argument('--deploy_device',type=str, default='none', help='also known as balance. options none, fix=asic, fpga')
    parser.add_argument('--group_size',type=int, default=64, help='also known as bank_size')
    parser.add_argument('--pruning_frequency',type=int, default=800, help='also known as bank_size')
    parser.add_argument('--pruning_epochs',type=int, default=0, help='pruning epochs')
    parser.add_argument('--local_rank',type=int, default=0, help='rank')

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        logfilename = 'log_sift_{}_version_{}_dd{}_gs{}_bs{}_lr{}_sp{}_{}.txt'.format(args.sift, args.sift_version, args.deploy_device, args.group_size, args.per_device_train_batch_size, args.learning_rate, args.pruning_sparsity, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        logfilename = os.path.join(args.output_dir, logfilename)
        handler = logging.FileHandler(logfilename)
        logger.addHandler(handler)
        logger.info('------------> log file =={}'.format(logfilename))
        logger.info(args)

    logger.info(accelerator.state)
    accelerator.wait_for_everyone()

    if args.sift:
        if args.sift_version == 1:
            from sift import AdversarialLearner, hook_sift_layer
        elif args.sift_version == 2:
            from sift_v2 import AdversarialLearner, hook_sift_layer
        elif args.sift_version == 3:
            from sift_v3 import AdversarialLearner, hook_sift_layer

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    teacher = None
    if args.kd:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        teacher.eval()

    if args.jiant_prepruned_weight:
        weights_dict = torch.load(args.jiant_prepruned_weight)
        keys = list(weights_dict.keys())

        if "encoder" in keys[0]:
            from collections import OrderedDict
            weights = []
            for k, v in weights_dict.items():
                # if k == "classifier.weight" or k == "classifier.bias":
                #     weights.append((f"taskmodels_dict.mnli.head.{k}", v))
                # else:
                if not k.startswith("taskmodels_dict") and "pooler" not in k:
                    weights.append((f"bert.{'.'.join(k.split('.')[1:])}", v))
                elif "qa_outputs" in k:
                    weights.append(('.'.join(k.split('.')[-2:]), v))
            weights_dict = OrderedDict(weights)

        output = model.load_state_dict(weights_dict, strict=False)
        logger.info(f"Load weight result: {output}")

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    eval_train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, teacher, optimizer, train_dataloader, eval_dataloader, eval_train_dataloader = accelerator.prepare(
        model, teacher, optimizer, train_dataloader, eval_dataloader, eval_train_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    prune_dict = {}
    for k, v in model.named_parameters():
        # FF nn
        if ('intermediate.dense.weight' in k or 'output.dense.weight' in k) and ('attention.output.dense.weight' not in k):
            prune_dict[k] = args.pruning_sparsity
        # Att nn
        if 'attention.self.query.weight' in k or 'attention.self.key.weight' in k or 'attention.self.value.weight' in k or 'attention.output.dense.weight' in k:
            prune_dict[k] = args.pruning_sparsity

    
    pruner = None
    if args.prune:
        pruner = Prune(
            model=model, 
            pretrain_step=0,
            sparse_step=num_update_steps_per_epoch * args.pruning_epochs,
            current_step=args.current_step,
            frequency=args.pruning_frequency,
            prune_dict=prune_dict,
            restore_sparsity=False,
            fix_sparsity=False,
            prune_device='default',
            deploy_device=args.deploy_device,
            group_size=args.group_size,
            fixed_mask=args.fixed_mask,
            mask=args.mask
        )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    tr_att_loss = 0
    tr_rep_loss = 0
    tr_cls_loss = 0
    tr_loss = 0

    loss_mse = MSELoss()

    if args.sift:
        if args.sift_version == 1:
            adv_modules = hook_sift_layer(model, hidden_size=768)
            adv = AdversarialLearner(model, adv_modules)
            def logits_fn(model, *wargs, **kwargs):
                outputs = model(*wargs, **kwargs)
                return outputs.logits
        else:
            adv_modules = hook_sift_layer(model, teacher, hidden_size=768)
            adv = AdversarialLearner(model, teacher, adv_modules)
            def logits_fn(model, *wargs, **kwargs):
                outputs = model(*wargs, **kwargs)
                return outputs.logits

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, output_hidden_states=True, output_attentions=True)
            student_loss = outputs.loss
            student_logits = outputs.logits
            student_hidden_states = outputs.hidden_states
            student_attentions = outputs.attentions

            if teacher:
                with torch.no_grad():
                    teacher_outputs = teacher(**batch, output_hidden_states=True, output_attentions=True)
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states
                    teacher_attentions = teacher_outputs.attentions

            att_loss = torch.zeros(1).cuda()
            rep_loss = torch.zeros(1).cuda()
            cls_loss = torch.zeros(1).cuda()
            loss = torch.zeros(1).cuda()
            
            if args.kd:
                for student_att, teacher_att in zip(student_attentions[10:], teacher_attentions[10:]):
                    # student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att), student_att, )
                    # teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att), teacher_att)
                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss

                for student_rep, teacher_rep in zip(student_hidden_states[10:], teacher_hidden_states[10:]):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss                        

                cls_loss = soft_cross_entropy(
                    student_logits / args.temperature, teacher_logits / args.temperature)
            
                loss = rep_loss + att_loss + cls_loss
            else:
                loss = student_loss
                assert loss, "The switch of loss computation is closed because of kd"                

            if args.sift:
                if args.sift_version == 1:
                    loss += adv.loss(student_logits, logits_fn, **batch)
                else:
                    loss += adv.loss(student_logits, teacher_logits, logits_fn, **batch)

            # loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            tr_att_loss += att_loss.item()
            tr_rep_loss += rep_loss.item()
            tr_cls_loss += cls_loss.item()
            tr_loss += loss.item()

            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

                if pruner:
                   pruner.prune()

            if completed_steps % args.eval_step == 0:
                if accelerator.is_main_process:
	                    # optimized step, loss, att_loss, rep_loss, cls loss, tr_loss, tr_att_loss, tr_rep_loss, tr_cls_loss 
                    logger.info("{:0>6d}/{:0>6d}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
                        completed_steps,
                        args.max_train_steps, 
                        loss.item(),
                        att_loss.item(),
                        rep_loss.item(),
                        cls_loss.item(),
                        tr_loss / completed_steps,
                        tr_att_loss / completed_steps,
                        tr_rep_loss / completed_steps,
                        tr_cls_loss / completed_steps)
                    )

                if pruner and accelerator.is_main_process:
                    layer_sparse_rate, total_sparse_rate = pruner.sparsity()
                    logger.info('\nepoch %d; step=%d; weight sparsity=%s; layer weight sparsity=%s\n' % (epoch, completed_steps, total_sparse_rate, layer_sparse_rate))

                model.eval()
                metric = None
                if args.task_name is not None:
	                metric = load_metric("glue", args.task_name)
                else:
	                metric = load_metric("accuracy")
                for step, batch in enumerate(eval_dataloader):
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch["labels"]),
                    )
                eval_metric = metric.compute()
                logger.info(f"epoch {epoch}, step {completed_steps}/{args.max_train_steps}: {eval_metric}")
                model.train()

            if completed_steps >= args.max_train_steps:
                break

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)

    model.eval()
    metric = None
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")
    for step, batch in enumerate(eval_train_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
    eval_metric = metric.compute()
    logger.info(f"Train Result: {eval_metric}")  

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    metric = None
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")
    for step, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
    eval_metric = metric.compute()
    logger.info(f"DEV Result: {eval_metric}")    

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training")

    metric = None
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy") 
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

if __name__ == "__main__":
    main()
