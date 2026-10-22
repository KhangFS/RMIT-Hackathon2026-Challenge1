"""
Utility module -- Member A (Huy)
Seed, metrics, and helper functions for model training.
"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


def seed_everything(seed: int = 42) -> None:
    """Fix tat ca random seeds de ket qua co the tai tao."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred) -> dict:
    """
    Tinh Macro-F1 tu logits va labels cho 3 classes (BENIGN, JAILBREAK, PROMPT_INJECTION).
    eval_pred: tuple (logits, labels) tuong thich voi HuggingFace Trainer.
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=1)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    return {
        "macro_f1": macro_f1,
        "f1_benign": per_class_f1[0] if len(per_class_f1) > 0 else 0.0,
        "f1_jailbreak": per_class_f1[1] if len(per_class_f1) > 1 else 0.0,
        "f1_prompt_injection": per_class_f1[2] if len(per_class_f1) > 2 else 0.0,
    }


def detailed_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """
    Bao cao day du cac chi so phan loai 3-class (Macro-F1, Precision, Recall per class).
    y_pred_probs: mang xac suat shape (N, 3) hoac logits (N, 3).
    """
    if y_pred_probs.ndim == 2:
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = y_pred_probs

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "macro_f1": round(float(macro_f1), 4),
        "f1_benign": round(float(per_class_f1[0]), 4) if len(per_class_f1) > 0 else 0.0,
        "f1_jailbreak": round(float(per_class_f1[1]), 4) if len(per_class_f1) > 1 else 0.0,
        "f1_prompt_injection": round(float(per_class_f1[2]), 4) if len(per_class_f1) > 2 else 0.0,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "confusion_matrix": cm.tolist(),
    }
