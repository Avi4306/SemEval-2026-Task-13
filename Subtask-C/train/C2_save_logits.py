import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, codes, tokenizer, max_length=512):
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.tokenizer(
            str(self.codes[idx]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

def extract_logits(model_path, texts, batch_size=32, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=4,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    dataset = CodeDataset(texts, tokenizer, max_length)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return np.vstack(all_logits)

def main():
    parser = argparse.ArgumentParser(description="Extract logits from trained models")

    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--val_data", default="Task_C/validation.parquet")
    parser.add_argument("--test_data", default="Task_C/test.parquet")
    parser.add_argument("--output_dir", default="./logits")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()

    if len(args.model_paths) != len(args.model_names):
        raise ValueError("model_paths and model_names must have same length")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    val_df = pd.read_parquet(args.val_data)
    test_df = pd.read_parquet(args.test_data)

    if args.sample_size:
        val_df = val_df.sample(min(args.sample_size, len(val_df)), random_state=42)
        test_df = test_df.sample(min(args.sample_size, len(test_df)), random_state=42)

    for model_path, model_name in zip(args.model_paths, args.model_names):
        print("\n" + "=" * 60)
        print(f"Processing model: {model_name}")
        print("=" * 60)

        print("Extracting validation logits...")
        val_logits = extract_logits(
            model_path,
            val_df["code"].tolist(),
            args.batch_size,
            args.max_length
        )
        np.save(f"{args.output_dir}/{model_name}_val_logits.npy", val_logits)

        print("Extracting test logits...")
        test_logits = extract_logits(
            model_path,
            test_df["code"].tolist(),
            args.batch_size,
            args.max_length
        )
        np.save(f"{args.output_dir}/{model_name}_test_logits.npy", test_logits)

        if model_name == args.model_names[0]:
            np.save(
                f"{args.output_dir}/val_labels.npy",
                val_df["label"].values
            )

    np.save(f"{args.output_dir}/test_ids.npy", test_df["ID"].values)

    metadata = {
        "model_names": args.model_names,
        "model_paths": args.model_paths,
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }

    with open(f"{args.output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nLogit extraction complete!")

if __name__ == "__main__":
    main()