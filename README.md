# SemEval-2026-Task-13

Code for the paper **"TransformerTrio at SemEval-2026 Task 13: Navigating Domain Shift and Representation Instability in Machine-Generated Code Detection"** (under review).

## Repository Structure

```
SemEval-2026-Task-13/
│
├── Subtask-A/
│   ├── train/
│   │   ├── A1.py              # XGBoost on handcrafted features
│   │   ├── A2.py              # XGBoost + frozen ModernBERT-Base embeddings
│   │   ├── A3_train.py        # Fine-tune UniXcoder
│   │   ├── A3_embed.py        # Extract embeddings from fine-tuned UniXcoder
│   │   └── A3.py              # Adversarial validation + XGBoost with dimension dropping
│   └── slurms/
│       ├── A1.sh
│       ├── A2.sh
│       ├── A3_train.sh
│       ├── A3_embed.sh
│       └── A3.sh
│
├── Subtask-B/
│   ├── train/
│   │   └── B1.py              # Fine-tune ModernBERT-Large for 11-class attribution
│   └── slurms/
│       └── B1.sh
│
├── Subtask-C/
│   ├── train/
│   │   ├── C1.py              # Fine-tune ModernBERT-Large for 4-class detection
│   │   ├── C2_train.py        # Fine-tune UniXcoder-Base (stage 1 of stacking)
│   │   ├── C2_save_logits.py  # Extract & save logits from stage-1 model
│   │   └── C2_stacking.py     # Train XGBoost meta-classifier on logit features
│   └── slurms/
│       ├── C1.sh
│       ├── C2_train.sh
│       ├── C2_save_logits.sh
│       └── C2_stacking.sh
│
└── README.md
```