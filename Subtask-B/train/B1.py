import argparse
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

from sklearn.metrics import f1_score, classification_report, confusion_matrix

TASK_B_LABELS = 11

LABEL_MAP = {
    "Human": 0,
    "DeepSeek-AI": 1,
    "Qwen": 2,
    "01-ai": 3,
    "BigCode": 4,
    "Gemma": 5,
    "Phi": 6,
    "Meta-LLaMA": 7,
    "IBM-Granite": 8,
    "Mistral": 9,
    "OpenAI": 10
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = list(LABEL_MAP.keys())

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

class TaskBDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = int(self.labels[idx])

        if not (0 <= label <= 10):
            raise ValueError(f"Invalid label {label} at index {idx} (Task B: 0-10)")

        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
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

def load_data(args):
    train_df = pd.read_parquet("Task_B/train.parquet")
    val_df = pd.read_parquet("Task_B/validation.parquet")
    test_sample_df = pd.read_parquet("Task_B/test_sample.parquet")
    test_df = pd.read_parquet("Task_B/test.parquet")
    
    print(f"Task B - Dataset sizes:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    
    return train_df, val_df, test_sample_df, test_df

def create_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=TASK_B_LABELS,
        trust_remote_code=True,
        problem_type="single_label_classification",
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
    parser.add_argument("--max_length", type=int, default=512)
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
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--submission_file", default="submission_task_b.csv")

    args = parser.parse_args()
    set_seed(args.seed)
    
    train_df, val_df, test_sample_df, test_df = load_data(args)
    model, tokenizer = create_model_and_tokenizer(args)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = TaskBDataset(train_df["code"].values, train_df["label"].values, tokenizer, args.max_length)
    val_dataset = TaskBDataset(val_df["code"].values, val_df["label"].values, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 or args.bf16 else None,
    )

    trainer = Trainer(
        model=model,
        args=setup_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)],
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")

    if args.do_eval:
        eval_ds = TaskBDataset(test_sample_df["code"].values, test_sample_df["label"].values, tokenizer, args.max_length)
        preds = trainer.predict(eval_ds)
        y_pred = np.argmax(preds.predictions, axis=1)
        y_true = preds.label_ids
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    if args.do_predict:
        pred_ds = TaskBDataset(test_df["code"].values, np.zeros(len(test_df), dtype=int), tokenizer, args.max_length)
        preds = trainer.predict(pred_ds)
        labels = np.argmax(preds.predictions, axis=1)
        
        pd.DataFrame({"ID": test_df["ID"], "label": labels}).to_csv(args.submission_file, index=False)
        print(f"Predictions saved to: {args.submission_file}")

if __name__ == "__main__":
    main()