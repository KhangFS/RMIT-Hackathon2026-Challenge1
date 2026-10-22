"""
Script du doan (Inference) cho RMIT Hackathon 2026.
Chay tren test.csv (6,000 mau) bang cac fold models da luu va xuat submission.csv chuan.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

from src.dataset import PromptDataset
from src.preprocessing import clean_and_deobfuscate_text

ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}


def predict_2026(
    models_dir: str = "../models_2026",
    test_file: str = "./data/test.csv" if os.path.exists("./data/test.csv") else r"D:\Code\rmit\data\Data-rmit2026\test.csv",
    output_file: str = "submission.csv",
    batch_size: int = 32,
    max_length: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Dang su dung: {device}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Khong tim thay test.csv tai: {test_file}")

    df_test = pd.read_csv(test_file)
    print(f"[Data] So mau test: {len(df_test)}")
    df_test["clean_text"] = df_test["text"].apply(clean_and_deobfuscate_text)

    # Tim cac checkpoint .pt
    ckpt_files = sorted(glob.glob(os.path.join(models_dir, "**", "*.pt"), recursive=True))
    if not ckpt_files:
        # Fallback tim trong ../models
        ckpt_files = sorted(glob.glob(os.path.join("../models", "**", "*.pt"), recursive=True))

    if not ckpt_files:
        raise FileNotFoundError(f"Khong tim thay checkpoint .pt nao trong {models_dir}")

    print(f"[Models] Tim thay {len(ckpt_files)} checkpoints:")
    for f in ckpt_files:
        print(f" - {os.path.basename(f)}")

    # Lay tokenizer tu model dau tien hoac xlm-roberta-base / mdeberta
    model_name = "FacebookAI/xlm-roberta-base" if "xlm" in ckpt_files[0].lower() else "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_ds = PromptDataset(df_test["clean_text"].values, tokenizer=tokenizer, max_length=max_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_fold_probs = []

    for ckpt_path in ckpt_files:
        fold_name = os.path.basename(ckpt_path)
        print(f"\n[Inference] Dang suy luan checkpoint: {fold_name}...")
        
        # Kiem tra so luong labels tu checkpoint state_dict
        state_dict = torch.load(ckpt_path, map_location=device)
        classifier_keys = [k for k in state_dict.keys() if "classifier.weight" in k or "classifier.out_proj.weight" in k]
        num_labels = state_dict[classifier_keys[0]].shape[0] if classifier_keys else 3
        
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  {fold_name}", leave=False):
                logits = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                fold_probs.extend(probs)

        all_fold_probs.append(np.array(fold_probs))
        del model
        torch.cuda.empty_cache()

    # Ensemble
    ensemble_probs = np.mean(all_fold_probs, axis=0)

    if ensemble_probs.shape[1] == 3:
        pred_indices = np.argmax(ensemble_probs, axis=1)
        pred_labels = [ID2LABEL[idx] for idx in pred_indices]
    else:
        # 2 classes fallback
        p_jailbreak = ensemble_probs[:, 1]
        pred_labels = np.where(p_jailbreak >= 0.5, "JAILBREAK", "BENIGN")

    sub = pd.DataFrame({
        "id": df_test["id"],
        "label": pred_labels
    })

    sub.to_csv(output_file, index=False)
    print(f"\n[Done] Da tao file nop bai: {output_file} ({len(sub)} dong)")
    print(f"[Done] Phan bo du doan:")
    print(sub["label"].value_counts(normalize=True))
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMIT Hackathon 2026 -- Predict")
    parser.add_argument("--models_dir", default="../models_2026")
    parser.add_argument("--test_file", default=r"D:\Code\rmit\data\Data-rmit2026\test.csv")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    predict_2026(
        models_dir=args.models_dir,
        test_file=args.test_file,
        output_file=args.output,
        batch_size=args.batch,
    )
