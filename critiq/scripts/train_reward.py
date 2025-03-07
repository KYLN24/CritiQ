import argparse
import json
import os
import random

import datasets
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardConfig, RewardTrainer

from critiq.utils import is_pair_data, reverse_ab

Qwen2ForRewardModel = None


def get_qwen2_rm_class():
    global Qwen2ForRewardModel  # pylint: disable=W0603:global-statement
    if Qwen2ForRewardModel is not None:
        return
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-Math-RM-72B", trust_remote_code=True
    )
    config.hidden_size = 1
    config.num_hidden_layers = 1
    config.num_attention_heads = 1
    config.intermediate_size = 1
    config.vocab_size = 1
    model = AutoModel.from_config(config, trust_remote_code=True)
    Qwen2ForRewardModel = model.__class__
    del model
    del config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_random_seed(args.seed)

    training_args = RewardConfig(
        output_dir=f"{args.output_dir}/{args.job_name}",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.accum,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,
        logging_dir=f"{args.output_dir}/{args.job_name}/logs",
        logging_steps=1,
        logging_first_step=True,
        logging_strategy="steps",
        bf16=True,
        tf32=True,
        greater_is_better=True,
        metric_for_best_model="accuracy",
        seed=args.seed,
        deepspeed=(
            {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "zero_optimization": {"stage": args.zero_stage},
                "bf16": {"enabled": "auto"},
            }
            if args.zero_stage
            else None
        ),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        report_to="tensorboard",
        max_length=args.max_length,
        remove_unused_columns=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = args.max_length
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if args.use_qwen2_rm or "QWEN2" in args.model.upper():
        get_qwen2_rm_class()
        model = Qwen2ForRewardModel.from_pretrained(
            args.model,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    print(model.__class__)

    if not model.config.pad_token_id:
        model.config.pad_token_id = model.config.eos_token_id

    def preprocess_function(examples):
        chosen = []
        rejected = []
        for idx, answer in enumerate(examples["answer"]):
            chosen.append(examples[answer][idx])
            rejected.append(examples[reverse_ab(answer)][idx])

        chosen_tokenized = tokenizer(chosen, truncation=True)
        rejected_tokenized = tokenizer(rejected, truncation=True)
        return {
            "input_ids_chosen": chosen_tokenized.input_ids,
            "attention_mask_chosen": chosen_tokenized.attention_mask,
            "input_ids_rejected": rejected_tokenized.input_ids,
            "attention_mask_rejected": rejected_tokenized.attention_mask,
        }

    with open(args.data, "r", encoding="utf8") as f:
        data = [json.loads(line) for line in f]
    n = len(data)

    data = list(filter(is_pair_data, data))
    n1 = len(data)
    data = (
        datasets.Dataset.from_list(data)
        .map(
            preprocess_function,
            batched=True,
        )
        .filter(
            lambda d: len(d["input_ids_chosen"]) <= args.max_length
            and len(d["input_ids_rejected"]) <= args.max_length
        )
    )
    n2 = len(data)

    print(
        f"Filtered out {n - n2} examples ({n-n1} + {n1-n2}), dataset size: {len(data)}"
    )

    if not args.only_eval:
        data = data.train_test_split(test_size=0.05, seed=args.seed)

    if args.only_eval:
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=data,  # Make RewardTrainer happy.
            eval_dataset=data,
            processing_class=tokenizer,
        )
    else:
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(os.path.join(args.output_dir, f"{args.job_name}"))

    print(trainer.evaluate())


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="tmp",
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--data", type=str)
    parser.add_argument("--seed", type=int, default=100745534)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--use_qwen2_rm", action="store_true")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
