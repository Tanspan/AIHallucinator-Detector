from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random
from collections import Counter
from datasets import load_dataset, Dataset

LABEL2ID = {
    "SUPPORTED":             0,
    "HALLUCINATED":          1,
    "INSUFFICIENT_EVIDENCE": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class Sample:
    claim:    str
    evidence: str
    label:    int


# ─────────────────────────────────────────────
# MedHallu
# UTAustin-AIHealth/MedHallu
# ─────────────────────────────────────────────

def load_medhallu(max_samples=None):
    print("Loading MedHallu...")
    samples = []

    for config in ["pqa_labeled", "pqa_artificial"]:
        try:
            ds = load_dataset("UTAustin-AIHealth/MedHallu", config, split="train")
            for row in ds:
                question     = str(row.get("Question", "")).strip()
                knowledge    = row.get("Knowledge", [])
                ground       = str(row.get("Ground Truth", "")).strip()
                hallucinated = str(row.get("Hallucinated Answer", "")).strip()

                if isinstance(knowledge, list):
                    evidence = " ".join(knowledge[:2])
                else:
                    evidence = str(knowledge)

                if len(question) > 10 and len(ground) > 10:
                    samples.append(Sample(
                        claim    = question + " " + ground,
                        evidence = evidence.strip(),
                        label    = LABEL2ID["SUPPORTED"]
                    ))

                if len(question) > 10 and len(hallucinated) > 10:
                    samples.append(Sample(
                        claim    = question + " " + hallucinated,
                        evidence = evidence.strip(),
                        label    = LABEL2ID["HALLUCINATED"]
                    ))

                if max_samples and len(samples) >= max_samples:
                    break

        except Exception as e:
            print(f"  MedHallu {config} failed:", e)

    print("  Loaded", len(samples), "MedHallu samples.")
    return samples


# ─────────────────────────────────────────────
# PubMedQA
# pubmed_qa / qiaojin/PubMedQA
# ─────────────────────────────────────────────

def load_pubmedqa(max_samples=None):
    print("Loading PubMedQA...")
    try:
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    except Exception:
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        except Exception as e:
            print("  PubMedQA failed:", e)
            return []

    samples = []
    for row in ds:
        decision = str(row.get("final_decision", "")).lower().strip()

        if decision == "yes":
            label = LABEL2ID["SUPPORTED"]
        elif decision == "no":
            label = LABEL2ID["HALLUCINATED"]
        else:
            label = LABEL2ID["INSUFFICIENT_EVIDENCE"]

        question = str(row.get("question", "")).strip()
        ctx      = row.get("context", {})
        if isinstance(ctx, dict):
            evidence = " ".join(ctx.get("contexts", [])[:2])
        else:
            evidence = ""

        answer = str(row.get("long_answer", "")).strip()
        claim  = (question + " " + answer).strip()

        if len(claim) < 10:
            continue

        samples.append(Sample(claim=claim, evidence=evidence.strip(), label=label))

        if max_samples and len(samples) >= max_samples:
            break

    print("  Loaded", len(samples), "PubMedQA samples.")
    return samples


# ─────────────────────────────────────────────
# MedQA
# medalpaca/medical_meadow_medqa
# ─────────────────────────────────────────────

def load_medqa(max_samples=3000):
    print("Loading MedQA...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_medqa", split="train")
    except Exception as e:
        print("  MedQA failed:", e)
        return []

    samples = []
    for row in ds:
        instruction = str(row.get("input", "")).strip()
        output      = str(row.get("output", "")).strip()

        if len(instruction) < 10 or len(output) < 5:
            continue

        samples.append(Sample(
            claim    = instruction + " " + output,
            evidence = instruction,
            label    = LABEL2ID["SUPPORTED"]
        ))

        if len(samples) >= max_samples:
            break

    print("  Loaded", len(samples), "MedQA samples.")
    return samples


# ─────────────────────────────────────────────
# MedFact — NEW DATASET
# ibragimovv/MedFact
# Simple true/false medical statements
# Fixes model bias towards HALLUCINATED on simple facts
# e.g. "Insulin regulates blood glucose" → SUPPORTED
# ─────────────────────────────────────────────

def load_medfact(max_samples=3000):
    print("Loading MedFact...")
    try:
        ds = load_dataset("ibragimovv/MedFact", split="train")
        print("  Columns:", ds.column_names)
        print("  Sample:", ds[0])
    except Exception as e:
        print("  MedFact failed:", e)
        return []

    samples = []
    for row in ds:
        # Try common column names for label
        label_raw = (
            str(row.get("label", "")) or
            str(row.get("answer", "")) or
            str(row.get("verdict", ""))
        ).lower().strip()

        if label_raw in ("true", "1", "correct", "yes", "supported"):
            label = LABEL2ID["SUPPORTED"]
        elif label_raw in ("false", "0", "incorrect", "no", "hallucinated"):
            label = LABEL2ID["HALLUCINATED"]
        else:
            continue

        # Try common column names for claim text
        claim = (
            str(row.get("claim", "")) or
            str(row.get("statement", "")) or
            str(row.get("text", "")) or
            str(row.get("sentence", ""))
        ).strip()

        evidence = str(row.get("evidence", "") or
                       row.get("context", "") or "").strip()

        if len(claim) < 10:
            continue

        samples.append(Sample(
            claim    = claim,
            evidence = evidence,
            label    = label
        ))

        if len(samples) >= max_samples:
            break

    print("  Loaded", len(samples), "MedFact samples.")
    return samples


# ─────────────────────────────────────────────
# Build combined dataset
# ─────────────────────────────────────────────

def build_dataset(seed=42, val_ratio=0.1, test_ratio=0.1):
    all_samples: List[Sample] = []

    all_samples += load_medhallu()
    all_samples += load_pubmedqa()
    all_samples += load_medqa()
    all_samples += load_medfact()   # ← NEW: fixes simple fact classification

    if not all_samples:
        raise RuntimeError(
            "No datasets loaded. "
            "Check internet and run: pip install datasets"
        )

    # Oversample INSUFFICIENT_EVIDENCE (minority class)
    insufficient = [s for s in all_samples
                    if s.label == LABEL2ID["INSUFFICIENT_EVIDENCE"]]
    if insufficient:
        all_samples += insufficient * 3

    random.seed(seed)
    random.shuffle(all_samples)

    dist = Counter(ID2LABEL[s.label] for s in all_samples)
    print("\nTotal samples:", len(all_samples))
    print("Label distribution:", dict(dist))

    n      = len(all_samples)
    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)

    train = all_samples[:n - n_val - n_test]
    val   = all_samples[n - n_val - n_test: n - n_test]
    test  = all_samples[n - n_test:]

    print(f"Split → train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test


def samples_to_hf_dataset(samples: List[Sample]) -> Dataset:
    return Dataset.from_dict({
        "claim":    [s.claim    for s in samples],
        "evidence": [s.evidence for s in samples],
        "label":    [s.label    for s in samples],
    })