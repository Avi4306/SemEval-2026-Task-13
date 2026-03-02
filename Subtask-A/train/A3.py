#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MIN_FEATURES_FOR_FALLBACK = 10

def get_adversarial_auc_and_importances(X_train, X_test, n_jobs):
    X = np.vstack([X_train, X_test])
    y = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
    )

    for tr, va in skf.split(X, y):
        model.fit(X[tr], y[tr])
    model.fit(X, y)
    return model.feature_importances_

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    X_train = np.load(args.train_embeddings)
    X_test = np.load(args.test_embeddings)
    y_train = pd.read_parquet(args.train_parquet, columns=["label"])["label"].values.astype(int)
    test_ids = np.load(args.test_ids)

    total_dropped = []
    prev_X_train = X_train.copy()
    prev_X_test = X_test.copy()
    prev_total_dropped = total_dropped.copy()

    while True:
        importances = get_adversarial_auc_and_importances(X_train, X_test, args.n_jobs)
        bad_indices = np.argsort(importances)[-args.step_size:]
        n_features = X_train.shape[1]
        max_drop = max(0, n_features - 1)
        if max_drop == 0:
            break

        n_drop = min(len(bad_indices), max_drop)
        bad_indices = bad_indices[-n_drop:]
        total_dropped.extend(bad_indices.tolist())

        X_train = np.delete(X_train, bad_indices, axis=1)
        X_test = np.delete(X_test, bad_indices, axis=1)

        if X_train.shape[1] < MIN_FEATURES_FOR_FALLBACK:
            break

    if X_train.shape[1] == 0:
        if 'prev_X_train' in locals():
            X_train = prev_X_train
            X_test = prev_X_test
            total_dropped = prev_total_dropped
        else:
            raise ValueError("No features left after adversarial dropping.")

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
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    sub_05 = pd.DataFrame({"ID": test_ids, "prediction": (test_probs >= 0.5).astype(int)})
    path_05 = f"{args.output_dir}/A3_thr_0.5.csv"
    sub_05.to_csv(path_05, index=False)
    print(f"✅ Saved: {path_05}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings", required=True)
    parser.add_argument("--test_embeddings", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--test_ids", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    main(args)