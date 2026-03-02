import argparse
import os
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

def load_logits(model_names, logits_dir="./logits", split="val"):
    all_logits = []
    for model_name in tqdm(model_names, desc=f"{split} logits"):
        logits_path = os.path.join(logits_dir, f"{model_name}_{split}_logits.npy")
        if not os.path.exists(logits_path):
            raise FileNotFoundError(f"Logits file not found: {logits_path}")
        logits = np.load(logits_path)
        all_logits.append(logits)
    return np.hstack(all_logits)

def load_labels(logits_dir="./logits"):
    return np.load(os.path.join(logits_dir, "val_labels.npy"))

def load_test_ids(logits_dir="./logits"):
    return np.load(os.path.join(logits_dir, "test_ids.npy"))

def create_features(logits, n_models, n_classes=4, temp=1.0):
    n_samples = logits.shape[0]
    per_model_logits = logits.reshape(n_samples, n_models, n_classes)
    features_list = []

    for i in range(n_models):
        model_logits = per_model_logits[:, i, :]
        features_list.append(model_logits)

        exp_logits = np.exp((model_logits - np.max(model_logits, axis=1, keepdims=True)) / temp)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        features_list.append(probs)

        features_list.append(np.mean(model_logits, axis=1).reshape(-1, 1))
        features_list.append(np.std(model_logits, axis=1).reshape(-1, 1))
        features_list.append(np.max(model_logits, axis=1).reshape(-1, 1))
        features_list.append(np.min(model_logits, axis=1).reshape(-1, 1))

        confidence = np.max(probs, axis=1)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        top2_diff = sorted_probs[:, 0] - sorted_probs[:, 1]

        features_list.append(confidence.reshape(-1, 1))
        features_list.append(entropy.reshape(-1, 1))
        features_list.append(top2_diff.reshape(-1, 1))

        ranks = np.argsort(np.argsort(-probs, axis=1), axis=1)
        features_list.append(ranks)

    return np.hstack(features_list), per_model_logits.argmax(axis=2)

def tune_hyperparameters(X_train, y_train, n_trials=50, cv_folds=3):
    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for t_idx, v_idx in skf.split(X_train, y_train):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train[t_idx], y_train[t_idx])
            preds = model.predict(X_train[v_idx])
            scores.append(f1_score(y_train[v_idx], preds, average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def train_xgb_ensemble(model_names, logits_dir, output_dir, tune=True, n_trials=50):
    os.makedirs(output_dir, exist_ok=True)
    X_val_logits = load_logits(model_names, logits_dir, "val")
    y_val = load_labels(logits_dir)

    X_val_features, _ = create_features(X_val_logits, len(model_names))
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val_features)

    if tune:
        params = tune_hyperparameters(X_val_scaled, y_val, n_trials=n_trials)
    else:
        params = {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 200}

    model = xgb.XGBClassifier(**params, objective='multi:softprob', num_class=4, random_state=42)
    model.fit(X_val_scaled, y_val)

    preds = model.predict(X_val_scaled)
    print(f"Validation F1: {f1_score(y_val, preds, average='macro'):.4f}")

    with open(os.path.join(output_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump({'model': model, 'scaler': scaler, 'n_models': len(model_names)}, f)
    return model, scaler

def predict_and_submit(model_names, logits_dir, output_dir, submission_name):
    with open(os.path.join(output_dir, "xgb_model.pkl"), "rb") as f:
        saved = pickle.load(f)
    
    X_test_logits = load_logits(model_names, logits_dir, "test")
    X_test_features, _ = create_features(X_test_logits, saved['n_models'])
    X_test_scaled = saved['scaler'].transform(X_test_features)
    
    preds = saved['model'].predict(X_test_scaled)
    df = pd.DataFrame({"ID": load_test_ids(logits_dir), "label": preds})
    df.to_csv(os.path.join(output_dir, submission_name), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs='+', required=True)
    parser.add_argument("--logits_dir", default="./logits")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    if args.train:
        train_xgb_ensemble(args.model_names, args.logits_dir, args.output_dir, n_trials=args.n_trials)
    if args.predict:
        predict_and_submit(args.model_names, args.logits_dir, args.output_dir, "submission.csv")

if __name__ == "__main__":
    main()