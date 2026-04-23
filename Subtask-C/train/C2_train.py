import argparse
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    set_seed,
)

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


class GPUUsageCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and logs is not None:
            logs["gpu_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024 ** 3), 2
            )
            logs["gpu_reserved_gb"] = round(
                torch.cuda.memory_reserved() / (1024 ** 3), 2
            )


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        current = metrics.get("eval_f1_macro")
        if current is None:
            return

        if self.best is None or current > self.best:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stopping: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            print(f"🛑 Early stopping triggered (best F1={self.best:.4f})")
            control.should_training_stop = True


class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length=512, random_crop=False):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_crop = random_crop

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = int(self.labels[idx])

        if not (0 <= label <= 3):
            raise ValueError(f"Invalid label {label} at index {idx}")

        enc = self.tokenizer(
            code,
            truncation=not self.random_crop,
            max_length=None if self.random_crop else self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if self.random_crop and len(input_ids) > self.max_length:
            start = random.randint(0, len(input_ids) - self.max_length)
            input_ids = input_ids[start:start + self.max_length]
            attention_mask = attention_mask[start:start + self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy": (preds == labels).mean(),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
            loss = loss_fct(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def balance_by_lang_label(df, cap=None):
    if cap is None:
        return df

    def cap_group(x):
        return x.sample(n=min(len(x), cap), random_state=42)

    return (
        df.groupby(["language", "label"])
        .apply(cap_group)
        .reset_index(drop=True)
    )


def load_data(args):
    train_df = pd.read_parquet("Task_C/train.parquet")
    val_df = pd.read_parquet("Task_C/validation.parquet")

    if args.train_size < len(train_df):
        train_df = train_df.sample(args.train_size, random_state=42)

    train_df = balance_by_lang_label(train_df, args.max_group_size)

    if args.val_size < len(val_df):
        val_df = val_df.sample(args.val_size, random_state=42)

    return train_df, val_df

def create_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=4,
        trust_remote_code=True,
        problem_type="single_label_classification",
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer


def setup_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"{args.output_dir}/logs",
        report_to="none",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=args.tf32 and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        remove_unused_columns=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_random_crop", action="store_true")

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--max_group_size", type=int, default=None)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    set_seed(args.seed)

    train_df, val_df = load_data(args)

    class_weights = None
    if args.use_class_weights:
        class_weights = torch.tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(train_df["label"]),
                y=train_df["label"],
            ),
            dtype=torch.float,
        )

    model, tokenizer = create_model_and_tokenizer(args)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CodeDataset(
        train_df["code"].values,
        train_df["label"].values,
        tokenizer,
        args.max_length,
        args.use_random_crop,
    )

    val_dataset = CodeDataset(
        val_df["code"].values,
        val_df["label"].values,
        tokenizer,
        args.max_length,
        False,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 or args.bf16 else None,
    )

    trainer = WeightedTrainer(
        model=model,
        args=setup_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[
            GPUUsageCallback(),
            EarlyStoppingCallback(args.early_stopping_patience),
        ],
        class_weights=class_weights,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()