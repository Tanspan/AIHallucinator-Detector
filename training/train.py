"""
training/train.py
──────────────────────────────────────────────────────────────────
Fine-tunes BiomedBERT on combined MedHallu + MedNLI + PubMedQA
for 3-class medical hallucination detection.

BASE MODEL (borrowed — cite this paper):
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  Paper: Gu et al. 2020, arXiv:2007.15779
  Pretrained on PubMed abstracts + PubMedCentral full text.
  State-of-the-art on BLURB medical benchmark.

YOUR CONTRIBUTION:
  Fine-tuning on 3-class hallucination task with
  INSUFFICIENT_EVIDENCE as a distinct output — not present
  in any existing fine-tuned version of this model.

HOW TO RUN:
  Option A — Google Colab (RECOMMENDED, free GPU):
    1. Upload this entire medverify/ folder to Google Drive
    2. Open new Colab notebook
    3. Mount Drive:
         from google.colab import drive
         drive.mount('/content/drive')
    4. cd /content/drive/MyDrive/medverify
    5. pip install -r requirements.txt
    6. python training/train.py

  Option B — Local (needs NVIDIA GPU):
    pip install -r requirements.txt
    python training/train.py

EXPECTED TRAINING TIME:
  Google Colab T4 GPU: ~40-60 minutes for 3 epochs
  No GPU (CPU only):   ~8-12 hours (not recommended)

OUTPUT:
  Saved to: ./saved_model/
  This is YOUR fine-tuned model. Load it in the pipeline.

EXPECTED ACCURACY (from literature):
  BiomedBERT fine-tuned on MedNLI alone: ~83-84%
    Source: Kanakarajan et al. 2021
  BiomedBERT fine-tuned on combined data: ~85-88%
    (improvement from larger + diverse training set)
  DO NOT CLAIM 95%+ — that is not honest for this task.
  MedHallu paper (EMNLP 2025) shows GPT-4o achieves only
  ~62-72% on hard hallucination subset.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# Add parent to path so imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_datasets import build_dataset, samples_to_hf_dataset, LABEL2ID, ID2LABEL


# ── Configuration ────────────────────────────────────────────────────────────

# BASE MODEL — borrowed from Microsoft Research
# Cite: Gu et al. 2020, arXiv:2007.15779
BASE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# YOUR fine-tuned model will be saved here
SAVE_DIR = "./saved_model"

# Training hyperparameters
# Standard values from BERT fine-tuning paper (Devlin et al. 2019)
HYPERPARAMS = {
    "max_length":   256,    # max tokens per input (claim + evidence)
    "batch_size":   16,     # reduce to 8 if you get CUDA out of memory
    "epochs":       3,      # standard for BERT fine-tuning
    "learning_rate": 2e-5,  # standard for BERT fine-tuning
    "warmup_ratio": 0.1,    # 10% warmup — standard practice
    "weight_decay": 0.01,   # L2 regularisation
    "seed":         42,
}

NUM_LABELS = 3  # SUPPORTED / HALLUCINATED / INSUFFICIENT_EVIDENCE


# ── Dataset class ────────────────────────────────────────────────────────────

class MedVerifyDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset.
    Input format: [CLS] claim [SEP] evidence [SEP]
    This is standard BERT sentence-pair input format.
    """

    def __init__(self, samples, tokenizer, max_length: int = 256):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Standard BERT pair encoding:
        # Sentence A = claim (hypothesis)
        # Sentence B = evidence (premise)
        encoding = self.tokenizer(
            s.claim,
            s.evidence if s.evidence else " ",  # empty evidence → single space
            max_length   = self.max_length,
            padding      = "max_length",
            truncation   = True,         # truncates evidence if too long
            return_tensors = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get("token_type_ids",
                              torch.zeros(self.max_length, dtype=torch.long)).squeeze(),
            "labels":         torch.tensor(s.label, dtype=torch.long),
        }


# ── Evaluation function ──────────────────────────────────────────────────────

def evaluate(model, dataloader, device) -> dict:
    """Standard evaluation — no fake numbers."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "macro_f1":  f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds,
                                     average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds,
                                  average="macro", zero_division=0),
        "report":    classification_report(
                         all_labels, all_preds,
                         target_names=label_names, zero_division=0
                     ),
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ── Main training function ───────────────────────────────────────────────────

def train():
    torch.manual_seed(HYPERPARAMS["seed"])
    np.random.seed(HYPERPARAMS["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Training will be very slow.")
        print("Use Google Colab for free GPU access.\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading datasets")
    print("=" * 60)
    train_samples, val_samples, test_samples = build_dataset()

    # ── Load tokenizer and model ──────────────────────────────────────────
    print("=" * 60)
    print(f"STEP 2: Loading base model: {BASE_MODEL}")
    print("  (This is the borrowed model — cite Gu et al. 2020)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # YOUR fine-tuning: add classification head with 3 labels
    # num_labels=3 is YOUR task definition
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels  = NUM_LABELS,
        id2label    = ID2LABEL,
        label2id    = LABEL2ID,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Create dataloaders ────────────────────────────────────────────────
    train_ds = MedVerifyDataset(train_samples, tokenizer, HYPERPARAMS["max_length"])
    val_ds   = MedVerifyDataset(val_samples,   tokenizer, HYPERPARAMS["max_length"])
    test_ds  = MedVerifyDataset(test_samples,  tokenizer, HYPERPARAMS["max_length"])

    train_loader = DataLoader(train_ds, batch_size=HYPERPARAMS["batch_size"],
                              shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=HYPERPARAMS["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=HYPERPARAMS["batch_size"])

    # ── Optimiser and scheduler ───────────────────────────────────────────
    # AdamW — standard for BERT fine-tuning (Devlin et al. 2019)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = HYPERPARAMS["learning_rate"],
        weight_decay = HYPERPARAMS["weight_decay"],
    )

    total_steps  = len(train_loader) * HYPERPARAMS["epochs"]
    warmup_steps = int(total_steps * HYPERPARAMS["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Fine-tuning YOUR model")
    print("=" * 60)

    best_val_f1   = 0.0
    history       = []

    for epoch in range(HYPERPARAMS["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                labels         = labels,  # cross-entropy loss computed internally
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping — standard for BERT fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches  += 1

            if step % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {step}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Train loss:  {avg_loss:.4f}")
        print(f"  Val accuracy:{val_metrics['accuracy']:.4f}")
        print(f"  Val macro-F1:{val_metrics['macro_f1']:.4f}")

        history.append({
            "epoch":    epoch + 1,
            "loss":     round(avg_loss, 4),
            "val_acc":  round(val_metrics["accuracy"], 4),
            "val_f1":   round(val_metrics["macro_f1"], 4),
        })

        # Save best model (based on validation F1)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            print(f"  ✓ New best model (F1={best_val_f1:.4f}) — saving...")
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  Saved to {SAVE_DIR}/")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Final evaluation on TEST set")
    print("  (These are YOUR real accuracy numbers)")
    print("=" * 60)

    # Load best saved model for final evaluation
    from transformers import AutoModelForSequenceClassification as AMSC
    best_model = AMSC.from_pretrained(SAVE_DIR).to(device)
    test_metrics = evaluate(best_model, test_loader, device)

    print(f"\nFINAL TEST RESULTS (real numbers — use these in your thesis):")
    print(f"  Accuracy:   {test_metrics['accuracy']:.4f} "
          f"({test_metrics['accuracy']*100:.1f}%)")
    print(f"  Macro F1:   {test_metrics['macro_f1']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")
    print(f"  Recall:     {test_metrics['recall']:.4f}")
    print(f"\nPer-class breakdown:")
    print(test_metrics["report"])

    # Save results to JSON — use these numbers in your frontend
    results = {
        "model":         BASE_MODEL,
        "your_model":    SAVE_DIR,
        "hyperparams":   HYPERPARAMS,
        "training_history": history,
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_precision":round(test_metrics["precision"], 4),
        "test_recall":   round(test_metrics["recall"], 4),
        "classification_report": test_metrics["report"],
        "datasets_used": [
            "MedHallu/MedHallu",
            "bigbio/mednli",
            "qiaojin/PubMedQA (pqa_labeled)",
        ],
        "label_scheme": ID2LABEL,
    }

    results_path = os.path.join(SAVE_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print("Use test_accuracy and test_macro_f1 in your frontend dashboard.")
    print("\nTraining complete. Your model is at:", SAVE_DIR)


if __name__ == "__main__":
    train()
