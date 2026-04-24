from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.base_agent import BaseAgent

LABEL2ID = {
    "SUPPORTED":             0,
    "HALLUCINATED":          1,
    "INSUFFICIENT_EVIDENCE": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEFAULT_MODEL_PATH = "./saved_model"
FALLBACK_MODEL = "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"


@dataclass
class VerifyResult:
    label:            str
    label_id:         int
    confidence:       float
    scores:           dict
    using_your_model: bool


class MedVerifyAgent(BaseAgent):

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_length: int = 256):
        super().__init__()
        self.max_length = max_length
        self.device     = torch.device("cpu")  # CPU — avoids CUDA errors
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        if os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        ):
            self.log(f"Loading YOUR fine-tuned model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(self.device)
            self.using_your_model = True
            self.log("YOUR model loaded.")
        else:
            self.log(f"saved_model/ not found. Using fallback: {FALLBACK_MODEL}", "warning")
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                FALLBACK_MODEL
            ).to(self.device)
            self.using_your_model = False

        self.model.eval()

    def verify(self, claim: str, evidence: str) -> VerifyResult:
        if not evidence.strip():
            self.log("No evidence → INSUFFICIENT_EVIDENCE")
            return VerifyResult(
                label    = "INSUFFICIENT_EVIDENCE",
                label_id = LABEL2ID["INSUFFICIENT_EVIDENCE"],
                confidence = 1.0,
                scores   = {"SUPPORTED": 0.0, "HALLUCINATED": 0.0,
                            "INSUFFICIENT_EVIDENCE": 1.0},
                using_your_model = self.using_your_model,
            )

        inputs = self.tokenizer(
            claim, evidence,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1).squeeze()

        if self.using_your_model:
            scores = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(probs))}
        else:
            raw = probs.tolist()
            if len(raw) == 3:
                scores = {
                    "SUPPORTED":             round(raw[0], 4),
                    "INSUFFICIENT_EVIDENCE": round(raw[1], 4),
                    "HALLUCINATED":          round(raw[2], 4),
                }
            else:
                scores = {"SUPPORTED": 0.33, "HALLUCINATED": 0.33,
                          "INSUFFICIENT_EVIDENCE": 0.34}

        predicted_label = max(scores, key=scores.get)
        confidence      = scores[predicted_label]

        self.log(f"'{claim[:50]}' → {predicted_label} ({confidence:.3f})")

        return VerifyResult(
            label    = predicted_label,
            label_id = LABEL2ID[predicted_label],
            confidence = confidence,
            scores   = scores,
            using_your_model = self.using_your_model,
        )
