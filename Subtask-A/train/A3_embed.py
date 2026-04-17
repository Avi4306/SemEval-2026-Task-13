#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys

# =========================
# DATASET CLASS
# =========================
class TaskADataset(Dataset):
    def __init__(self, codes, tokenizer, max_length=512):
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code_text = str(self.codes[idx]) if self.codes[idx] is not None else ""
        enc = self.tokenizer(
            code_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

# =========================
# EMBEDDING EXTRACTION
# =========================
@torch.no_grad()
def extract_embeddings(model, loader, desc, use_pooling=True):
    model.eval()
    model.cuda()
    embs = []

    for batch in tqdm(loader, desc=desc, file=sys.stdout):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        out = model(**batch, output_hidden_states=True)
        
        if use_pooling:
            # 🧠 LAYER POOLING: Average [CLS] from last 4 layers.
            last_4_layers = out.hidden_states[-4:] 
            cls_pool = torch.stack([layer[:, 0, :] for layer in last_4_layers]).mean(dim=0)
            embs.append(cls_pool.cpu().numpy())
        else:
            cls_last = out.hidden_states[-1][:, 0, :]
            embs.append(cls_last.cpu().numpy())

    return np.vstack(embs)

# =========================
# MAIN
# =========================
def main(args):
    print("🚀 Starting Pre-trained Embedding Extraction", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Tokenizer and Model
    print("🔄 Loading tokenizer & model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True).cuda()

    # Load Data
    print("📦 Loading datasets from parquet...", flush=True)
    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)
    
    # Handle optional test_sample
    test_sample_df = None
    if args.test_sample_path and os.path.exists(args.test_sample_path):
        test_sample_df = pd.read_parquet(args.test_sample_path)
        print(f"Test sample loaded: {len(test_sample_df)} samples", flush=True)

    # Create Loaders
    def get_loader(df):
        return DataLoader(
            TaskADataset(df["code"].values, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    train_loader = get_loader(train_df)
    test_loader = get_loader(test_df)

    # Extraction
    print("\n🧠 Extracting TRAIN embeddings...", flush=True)
    train_embs = extract_embeddings(model, train_loader, "Train", args.use_pooling)
    np.save(os.path.join(args.output_dir, "train_embeddings.npy"), train_embs)

    print("🧠 Extracting TEST embeddings...", flush=True)
    test_embs = extract_embeddings(model, test_loader, "Test", args.use_pooling)
    np.save(os.path.join(args.output_dir, "test_embeddings.npy"), test_embs)
    
    # Extract test_sample if provided
    if test_sample_df is not None:
        print("🧠 Extracting TEST_SAMPLE embeddings...", flush=True)
        sample_loader = get_loader(test_sample_df)
        sample_embs = extract_embeddings(model, sample_loader, "Test Sample", args.use_pooling)
        np.save(os.path.join(args.output_dir, "test_sample_embeddings.npy"), sample_embs)

    # Save IDs
    if "ID" in test_df.columns:
        np.save(os.path.join(args.output_dir, "test_ids.npy"), test_df["ID"].values)

    print(f"✅ Done! Files saved to {args.output_dir}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--test_sample_path", help="Path to test_sample.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_pooling", action="store_true")

    args = parser.parse_args()
    main(args)