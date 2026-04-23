#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def get_adversarial_auc_and_importances(X_train, X_test, n_jobs):
    """Calculates current AUC and returns feature importance scores."""
    X = np.vstack([X_train, X_test])
    y = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42
    )
    
    for tr, va in skf.split(X, y):
        model.fit(X[tr], y[tr])
        oof_preds[va] = model.predict_proba(X[va])[:, 1]
    
    auc = roc_auc_score(y, oof_preds)
    model.fit(X, y)
    return auc, model.feature_importances_

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("📦 Loading data...", flush=True)
    
    print("Loading X_train...", flush=True)
    X_train = np.load(args.train_embeddings)
    print("Loaded X_train", X_train.shape, flush=True)

    print("Loading X_test...", flush=True)
    X_test = np.load(args.test_embeddings)
    print("Loaded X_test", X_test.shape, flush=True)

    print("Loading y_train...", flush=True)
    y_train = pd.read_parquet(args.train_parquet)["label"].values.astype(int)
    print("Loaded y_train", y_train.shape, flush=True)

    print("Loading test_ids...", flush=True)
    test_ids = np.load(args.test_ids)
    print("Loaded test_ids", test_ids.shape, flush=True)

    # ==========================================
    # ITERATIVE SURGICAL DROPPING (FEATURE MASK)
    # ==========================================
    current_auc = 1.0
    feature_mask = np.ones(X_train.shape[1], dtype=bool)
    
    print(f"🚀 Starting iterative dropping to reach AUC < {args.target_auc}...")
    
    while current_auc > args.target_auc:
        current_auc, importances = get_adversarial_auc_and_importances(
            X_train[:, feature_mask],
            X_test[:, feature_mask],
            args.n_jobs
        )
        print(f"📊 Current Adversarial AUC: {current_auc:.4f}")
        
        if current_auc <= args.target_auc:
            print(f"✅ Target reached! Total features dropped: {feature_mask.size - feature_mask.sum()}")
            break
            
        remaining = feature_mask.sum()
        max_safe_drop = remaining - 10
        if max_safe_drop <= 0:
            print("🛑 Reached minimum feature floor (10). Stopping drops.")
            break

        drop_count = min(args.step_size, max_safe_drop)
        bad_local = np.argsort(importances)[-drop_count:]
        global_bad = np.where(feature_mask)[0][bad_local]
        feature_mask[global_bad] = False
        
        print(f"🔪 Dropped {len(bad_local)} features. Remaining dims: {feature_mask.sum()}")

    # Apply final mask ONCE
    X_train = X_train[:, feature_mask]
    X_test  = X_test[:, feature_mask]

    # ==========================================
    # FINAL TASK TRAINING
    # ==========================================
    print("\n🧠 Training Final Classifier on 'Cleaned' Embeddings...", flush=True)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    print(f"🔮 Predicting with threshold={args.threshold}...", flush=True)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    print(f"Mean Train Prob: {model.predict_proba(X_train_scaled)[:, 1].mean():.4f}")
    print(f"Mean Test Prob:  {probs.mean():.4f}")

    submission = pd.DataFrame({"ID": test_ids, "prediction": preds})
    out_path = f"{args.output_dir}/adv_surgical_drop.csv"
    submission.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings", required=True)
    parser.add_argument("--test_embeddings", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--test_ids", required=True)
    parser.add_argument("--output_dir", required=True)
    
    parser.add_argument("--target_auc", type=float, default=0.7)
    parser.add_argument("--step_size", type=int, default=50)
    
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_jobs", type=int, default=32)

    args = parser.parse_args()
    main(args)