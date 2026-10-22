"""
RMIT GenAI & Cybersecurity Hackathon 2026 -- Challenge 1
Multilingual Multi-Class Prompt-Attack Classification (3 classes)
Approved Models:
  - distilbert-base-multilingual-cased (fastest baseline)
  - jhu-clsp/mmBERT-small / mmBERT-base
  - FacebookAI/xlm-roberta-base / xlm-roberta-large (strong multilingual)
  - google/rembert
  - protectai/deberta-v3-base-prompt-injection-v2
  - patronus-studio/wolf-defender-prompt-injection
"""
import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import emoji
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from tqdm.auto import tqdm

from src.utils import seed_everything, detailed_metrics
from src.dataset import PromptDataset
from src.preprocessing import clean_and_deobfuscate_text


# ---------------------------------------------------------------------------
# Approved Models Registry (RMIT Hackathon 2026 Rules)
# ---------------------------------------------------------------------------
APPROVED_MODELS = {
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "distilbert-multilingual": "distilbert/distilbert-base-multilingual-cased",
    "deberta-v3-prompt-injection": "protectai/deberta-v3-base-prompt-injection-v2",
    "wolf-defender": "patronus-studio/wolf-defender-prompt-injection",
    "rembert": "google/rembert",
    "mmbert-base": "jhu-clsp/mmBERT-base",
    "mmbert-small": "jhu-clsp/mmBERT-small",
}

LABEL_MAP = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}


def get_data_paths(data_dir: str = "./data"):
    """Tu dong tim train va test csv theo data_dir cua nguoi dung hoac Kaggle input."""
    if os.path.exists("./data/train.csv"):
        return "./data/train.csv", "./data/test.csv"
    if os.path.exists(r"D:\Code\rmit\data\Data-rmit2026\train.csv"):
        return r"D:\Code\rmit\data\Data-rmit2026\train.csv", r"D:\Code\rmit\data\Data-rmit2026\test.csv"

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    return train_path, test_path


def run_training_2026(
    model_key: str = "xlm-roberta-base",
    data_dir: str = r"D:\Code\rmit\data\Data-rmit2026",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    n_folds: int = 5,
    use_class_weights: bool = True,
    quick_run: bool = False,
):
    """
    Pipeline huan luyen da ngon ngu 3-class cho RMIT Hackathon 2026:
    - 3 classes: BENIGN (62%), JAILBREAK (31%), PROMPT_INJECTION (7%)
    - Weighted Cross-Entropy Loss de giai quyet class imbalance (nang f1 cua PROMPT_INJECTION).
    - Stratified K-Fold theo ca (label + language).
    - Tinh Macro-F1 chuan xac theo tieu chi danh gia cua Kaggle.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = APPROVED_MODELS.get(model_key, model_key)
    model_alias = model_key.replace("/", "_").replace("-", "_")

    print(f"\n{'='*70}")
    print(f"RMIT Hackathon 2026 -- Challenge 1: Multilingual Prompt-Attack")
    print(f"Model backbone : {model_name}")
    print(f"Device         : {device}")
    print(f"Folds          : {'1 (quick_run)' if quick_run else n_folds}")
    print(f"Classes        : BENIGN (0), JAILBREAK (1), PROMPT_INJECTION (2)")
    print(f"{'='*70}\n")

    train_file, test_file = get_data_paths(data_dir)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Khong tim thay tap train tai: {train_file}")

    # 1. Load data
    df_train = pd.read_csv(train_file)
    print(f"[Data] Train samples: {len(df_train)}")
    print(f"[Data] Ngon ngu: {dict(df_train['language'].value_counts())}")

    # De-obfuscate text
    df_train["clean_text"] = df_train["text"].apply(clean_and_deobfuscate_text)
    df_train["label_num"] = df_train["label"].map(lambda l: LABEL_MAP.get(str(l).strip().upper(), 0))
    print(f"[Data] Ti le nhan:\n{df_train['label'].value_counts(normalize=True)}\n")

    # Load test if exists
    df_test = None
    if os.path.exists(test_file):
        df_test = pd.read_csv(test_file)
        df_test["clean_text"] = df_test["text"].apply(clean_and_deobfuscate_text)
        print(f"[Data] Test samples: {len(df_test)}")

    # 2. Stratified Folds (stratify theo ca label va language de fold can bang)
    df_train["strat_key"] = df_train["language"].astype(str) + "_" + df_train["label_num"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    df_train["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train["strat_key"])):
        df_train.loc[val_idx, "fold"] = fold

    # 3. Class Weights cho Loss (giai quyet bat can bang nhan 62% vs 7%)
    weights = None
    if use_class_weights:
        cls_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=df_train["label_num"].values,
        )
        weights = torch.tensor(cls_weights, dtype=torch.float).to(device)
        print(f"[Loss] Class weights (balanced): {cls_weights.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # 4. Tokenizer & model save directory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    save_dir = f"./models_2026/{model_alias}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./models_2026", exist_ok=True)

    oof_probs = np.zeros((len(df_train), 3))
    test_probs = np.zeros((len(df_test), 3)) if df_test is not None else None

    test_loader = None
    if df_test is not None:
        test_ds = PromptDataset(df_test["clean_text"].values, tokenizer=tokenizer, max_length=max_length)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    num_folds = 1 if quick_run else n_folds

    # 5. Training Loop
    for fold in range(num_folds):
        print(f"\n{'─'*55}")
        print(f"FOLD {fold + 1} / {num_folds}")
        print(f"{'─'*55}")

        train_df = df_train[df_train["fold"] != fold].reset_index(drop=True)
        val_df   = df_train[df_train["fold"] == fold].reset_index(drop=True)
        val_idx  = df_train[df_train["fold"] == fold].index.values

        if quick_run:
            train_df = train_df.head(64)
            val_df   = val_df.head(32)
            val_idx  = val_idx[:32]

        train_loader = DataLoader(
            PromptDataset(train_df["clean_text"].values, train_df["label_num"].values, tokenizer, max_length),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            PromptDataset(val_df["clean_text"].values, val_df["label_num"].values, tokenizer, max_length),
            batch_size=batch_size * 2, shuffle=False, num_workers=0,
        )

        save_fold_path = f"{save_dir}/{model_alias}_fold{fold}.pt"
        config = AutoConfig.from_pretrained(model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)

        # Check resume
        if os.path.exists(save_fold_path):
            print(f"[Resume] Fold {fold + 1} da ton tai checkpoint: {save_fold_path}. Dang load...")
            model.load_state_dict(torch.load(save_fold_path, map_location=device))
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    val_preds_list.extend(probs)
            val_preds_arr = np.array(val_preds_list)
            oof_probs[val_idx] = val_preds_arr
            metrics = detailed_metrics(val_df["label_num"].values, val_preds_arr)
            print(f"[Resume] Fold {fold + 1} Loaded: Macro-F1={metrics['macro_f1']} (Benign:{metrics['f1_benign']}, Jailbreak:{metrics['f1_jailbreak']}, Injection:{metrics['f1_prompt_injection']})")

            if test_loader is not None and test_probs is not None:
                with torch.no_grad():
                    test_fold_list = []
                    for batch in test_loader:
                        logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                        test_fold_list.extend(torch.softmax(logits, dim=-1).cpu().numpy())
                    test_probs += np.array(test_fold_list) / num_folds
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * (1 if quick_run else epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )

        best_val_f1 = 0.0
        for epoch in range(1 if quick_run else epochs):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1} Train", leave=False):
                optimizer.zero_grad()
                out = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                loss = criterion(out.logits, batch["label"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    val_preds_list.extend(probs)

            val_preds_arr = np.array(val_preds_list)
            metrics = detailed_metrics(val_df["label_num"].values, val_preds_arr)
            avg_loss = train_loss / len(train_loader)

            print(
                f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} "
                f"| Macro-F1: {metrics['macro_f1']:.4f} "
                f"[B: {metrics['f1_benign']:.4f}, JB: {metrics['f1_jailbreak']:.4f}, PI: {metrics['f1_prompt_injection']:.4f}]"
            )

            if metrics["macro_f1"] >= best_val_f1:
                best_val_f1 = metrics["macro_f1"]
                torch.save(model.state_dict(), save_fold_path)

        oof_probs[val_idx] = val_preds_arr

        # Test inference for this fold
        if test_loader is not None and test_probs is not None:
            model.eval()
            test_fold_list = []
            with torch.no_grad():
                for batch in test_loader:
                    logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                    test_fold_list.extend(torch.softmax(logits, dim=-1).cpu().numpy())
            test_probs += np.array(test_fold_list) / num_folds

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # 6. Save OOF & Test probs
    np.save(f"./models_2026/oof_{model_alias}.npy", oof_probs)
    print(f"\n[Save] Da luu OOF probs -> ./models_2026/oof_{model_alias}.npy")

    overall_metrics = detailed_metrics(df_train["label_num"].values, oof_probs)
    print(f"[Done] Overall OOF Macro-F1 = {overall_metrics['macro_f1']:.4f}")
    print(f"       Per-Class: Benign={overall_metrics['f1_benign']:.4f}, Jailbreak={overall_metrics['f1_jailbreak']:.4f}, Injection={overall_metrics['f1_prompt_injection']:.4f}")

    if test_probs is not None:
        np.save(f"./models_2026/test_{model_alias}.npy", test_probs)
        pred_labels = [ID2LABEL[idx] for idx in np.argmax(test_probs, axis=1)]
        sub = pd.DataFrame({
            "id": df_test["id"],
            "label": pred_labels
        })
        sub_file = f"submission_{model_alias}.csv"
        sub.to_csv(sub_file, index=False)
        sub.to_csv("submission.csv", index=False)
        print(f"\n[Done] Da tao file nop bai: {sub_file} va submission.csv (6,000 dong)")
        print(f"[Done] Ti le du doan:\n{sub['label'].value_counts(normalize=True)}")

    return overall_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMIT Hackathon 2026 -- C1 Multilingual Classifier")
    parser.add_argument("--model", default="xlm-roberta-base", choices=list(APPROVED_MODELS.keys()),
                        help="Approved model backbone")
    parser.add_argument("--data_dir", default=r"D:\Code\rmit\data\Data-rmit2026")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--quick", action="store_true", help="Quick run 64 samples")
    args = parser.parse_args()

    run_training_2026(
        model_key=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_length=args.maxlen,
        n_folds=args.folds,
        quick_run=args.quick,
    )
