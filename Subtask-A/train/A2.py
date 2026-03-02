import pandas as pd
import numpy as np
import re
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier

def extract_robust_features(code):
    code_str = str(code)
    lines = code_str.splitlines()
    total_lines = len(lines)
    non_empty = len([l for l in lines if l.strip()])

    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_str)
    word_lengths = [len(w) for w in words]
    if len(word_lengths) >= 4 and np.std(word_lengths) > 1e-6:
        kurtosis = stats.kurtosis(word_lengths, fisher=True, bias=False)
    else:
        kurtosis = 0.0

    operators = re.findall(r'[+\-*/%=<>!&|^~]+', code_str)
    n1, n2 = len(set(operators)), len(set(words))
    N1, N2 = len(operators), len(words)
    halstead = (N1 + N2) * np.log2(n1 + n2 + 1e-10) if (n1 + n2) > 0 else 0

    single_line_comments = len(re.findall(r'^\s*(?://|#)', code_str, re.MULTILINE))
    multi_line_comments = len(re.findall(r'/\*.*?\*/', code_str, re.DOTALL))
    comment_lines = single_line_comments + multi_line_comments

    return {
        "word_length_kurtosis": float(kurtosis),
        "halstead_volume": float(halstead),
        "total_lines": total_lines,
        "non_empty_lines": non_empty,
        "single_line_comment_ratio":
            single_line_comments / max(1, comment_lines) if comment_lines > 0 else 0
    }

class ModernBERTMeanPoolingExtractor:
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, code):
        enc = self.tokenizer(
            str(code),
            truncation=True,
            max_length=1024,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        out = self.model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"]
        weights = mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)

        return pooled.squeeze(0).cpu().numpy()

train = pd.read_parquet("data/train.parquet")
val = pd.read_parquet("data/validation.parquet")
test = pd.read_parquet("data/test.parquet")

train_feats = train["code"].progress_apply(extract_robust_features).apply(pd.Series)
val_feats = val["code"].progress_apply(extract_robust_features).apply(pd.Series)
test_feats = test["code"].progress_apply(extract_robust_features).apply(pd.Series)

cols_to_norm = ["total_lines", "non_empty_lines", "halstead_volume", "single_line_comment_ratio"]
norm_stats = {}

for col in cols_to_norm:
    mean = train_feats[col].mean()
    std = train_feats[col].std() + 1e-9
    norm_stats[col] = (mean, std)
    train_feats[col + "_norm"] = (train_feats[col] - mean) / std
    val_feats[col + "_norm"]   = (val_feats[col] - mean) / std
    test_feats[col + "_norm"]  = (test_feats[col] - mean) / std

handcrafted_features = [
    "word_length_kurtosis",
    "total_lines_norm",
    "non_empty_lines_norm",
    "halstead_volume_norm",
    "single_line_comment_ratio_norm"
]

bert = ModernBERTMeanPoolingExtractor()
train_bert = np.vstack(train["code"].progress_apply(bert.encode))
val_bert = np.vstack(val["code"].progress_apply(bert.encode))
test_bert = np.vstack(test["code"].progress_apply(bert.encode))

X_train = np.hstack([train_bert, train_feats[handcrafted_features].values])
X_val = np.hstack([val_bert, val_feats[handcrafted_features].values])
X_test = np.hstack([test_bert, test_feats[handcrafted_features].values])

y_train = train["label"].values
y_val = val["label"].values

model = XGBClassifier(
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

model.fit(X_train, y_train)

test_probs = model.predict_proba(X_test)[:, 1]
pred_05 = (test_probs >= 0.5).astype(int)
df_05 = pd.DataFrame({"ID": test["ID"], "label": pred_05})
df_05.to_csv("A2_threshold_0.5.csv", index=False)
pred_95 = (test_probs >= 0.95).astype(int)
df_95 = pd.DataFrame({"ID": test["ID"], "label": pred_95})
df_95.to_csv("A2_threshold_0.95.csv", index=False)