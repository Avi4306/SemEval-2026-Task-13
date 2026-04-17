import pandas as pd
import numpy as np
import re
from scipy import stats
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

def extract_robust_features(code):
    code_str = str(code)
    lines = code_str.splitlines()
    total_lines = len(lines)
    non_empty = len([l for l in lines if l.strip()])
    
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_str)
    word_lengths = [len(w) for w in words]
    kurtosis = stats.kurtosis(word_lengths) if len(word_lengths) >= 4 else 0.0

    operators = re.findall(r'[+\-*/%=<>!&|^~]+', code_str)
    n1, n2 = len(set(operators)), len(set(words))
    N1, N2 = len(operators), len(words)
    halstead = (N1 + N2) * np.log2(n1 + n2 + 1e-10) if (n1 + n2) > 0 else 0

    single_line_comments = len(re.findall(r'^\s*(?://|#)', code_str, re.MULTILINE))
    multi_line_comments = len(re.findall(r'/\*.*?\*/', code_str, re.DOTALL))
    comment_lines = single_line_comments + multi_line_comments
    
    return {
        'word_length_kurtosis': float(kurtosis),
        'halstead_volume': float(halstead),
        'total_lines': total_lines,
        'non_empty_lines': non_empty,
        'single_line_comment_ratio': single_line_comments / max(1, comment_lines) if comment_lines > 0 else 0
    }

train = pd.read_parquet("Task_A/train.parquet")
val = pd.read_parquet("Task_A/validation.parquet")

tqdm.pandas(desc="Extracting Train Features")
train_feats = train['code'].progress_apply(extract_robust_features).apply(pd.Series)

tqdm.pandas(desc="Extracting Validation Features")
val_feats = val['code'].progress_apply(extract_robust_features).apply(pd.Series)

cols_to_norm = ['total_lines', 'non_empty_lines', 'halstead_volume', 'single_line_comment_ratio']

norm_stats = {}
for col in cols_to_norm:
    mean = train_feats[col].mean()
    std = train_feats[col].std() + 1e-9
    norm_stats[col] = (mean, std)
    train_feats[f'{col}_norm'] = (train_feats[col] - mean) / std
    val_feats[f'{col}_norm'] = (val_feats[col] - mean) / std

features = [
    'word_length_kurtosis',
    'total_lines_norm',
    'non_empty_lines_norm',
    'halstead_volume_norm',
    'single_line_comment_ratio_norm'
]

X_train, y_train = train_feats[features], train['label']
X_val, y_val = val_feats[features], val['label']

xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

importance = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Feature Importance ---")
print(importance)

y_val_probs = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_probs >= 0.5).astype(int)
print("\n--- Validation Classification Report (Threshold 0.5) ---")
print(classification_report(y_val, y_val_pred))

final_test = pd.read_parquet("test.parquet")
final_test_feats = pd.DataFrame([extract_robust_features(c) for c in tqdm(final_test['code'], desc="Final Test Extraction")])

for col in cols_to_norm:
    mean, std = norm_stats[col]
    final_test_feats[f'{col}_norm'] = (final_test_feats[col] - mean) / std

X_final = final_test_feats[features]
final_probs = xgb_model.predict_proba(X_final)[:, 1]

id_col = 'ID' if 'ID' in final_test.columns else 'id'
submission_thresholds = [0.5, 0.84]

for t in submission_thresholds:
    sub_df = pd.DataFrame({
        id_col: final_test[id_col],
        'label': (final_probs >= t).astype(int)
    })
    filename = f"A1_thr_{int(t*100)}.csv"
    sub_df.to_csv(filename, index=False)
    print(f"✅ Saved {filename} (Threshold: {t:.2f}, Positive Count: {sub_df['label'].sum()})")