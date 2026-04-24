"""
vision/biovil_reader.py
────────────────────────
Vision Layer — BioViL-T Image Reader

Reads a medical image and generates text findings.
This runs BEFORE the main pipeline in image mode.

BioViL-T is a Microsoft Research model trained on
chest X-rays + radiology reports. It can:
  1. Generate findings text from an image
  2. Score visual entailment (image vs claim)

We use it here for (1) — generating the text that
then flows into the 9-agent verification pipeline.
"""

from __future__ import annotations
from typing import Optional
import logging

logger = logging.getLogger("BioViLReader")


class BioViLReader:
    """
    Reads a medical image and generates text findings using BioViL-T.

    In production: loads microsoft/BioViL-T from HuggingFace.
    Falls back to a descriptive placeholder if model unavailable.
    """

    def __init__(self, model_name: str = "microsoft/BioViL-T"):
        self.model_name = model_name
        self._model     = None
        self._processor = None
        self._load()

    def _load(self) -> None:
        try:
            # BioViL-T uses a custom processor + model
            from health_multimodal.image import get_image_inference
            from health_multimodal.text  import get_bert_inference
            self._image_inference = get_image_inference()
            self._text_inference  = get_bert_inference()
            logger.info("BioViL-T loaded successfully.")
        except Exception as e:
            logger.warning(
                f"BioViL-T not available ({e}). "
                "Install: pip install health-multimodal. "
                "Using fallback mode."
            )
            self._image_inference = None

    def generate_findings(self, image_path: str) -> str:
        """
        Generate text findings from a medical image.

        Parameters
        ----------
        image_path : str
            Path to the medical image file (JPEG/PNG).

        Returns
        -------
        str
            Generated text describing the image findings.
        """
        if self._image_inference is None:
            return self._fallback_findings(image_path)

        try:
            # BioViL-T visual grounding pipeline
            from health_multimodal.image.data.io import load_image
            from PIL import Image

            image = load_image(image_path)

            # Generate findings using image-text matching
            # BioViL-T scores candidate phrases against the image
            candidate_findings = [
                "No acute cardiopulmonary findings",
                "Consolidation present in the lung",
                "Pleural effusion observed",
                "Cardiomegaly present",
                "Pneumothorax detected",
                "Normal chest radiograph",
                "Bilateral infiltrates present",
                "Atelectasis noted",
            ]

            scores = []
            for finding in candidate_findings:
                score = self._score_finding(image_path, finding)
                scores.append((finding, score))

            # Return the highest-scoring finding
            scores.sort(key=lambda x: x[1], reverse=True)
            top_findings = [f for f, s in scores[:3] if s > 0.5]

            if top_findings:
                return ". ".join(top_findings) + "."
            return "No significant findings identified."

        except Exception as e:
            logger.error(f"BioViL-T inference failed: {e}")
            return self._fallback_findings(image_path)

    def _score_finding(self, image_path: str, finding: str) -> float:
        """Score how well a finding matches the image using BioViL-T."""
        try:
            image_emb = self._image_inference.get_projected_global_embedding(image_path)
            text_emb  = self._text_inference.get_projected_embeddings(
                [finding], normalize=True
            )
            import torch
            import torch.nn.functional as F
            score = F.cosine_similarity(image_emb, text_emb).item()
            return (score + 1) / 2   # normalise to [0, 1]
        except Exception:
            return 0.0

    def _fallback_findings(self, image_path: str) -> str:
        """
        Fallback when BioViL-T is unavailable.
        Returns a placeholder message that the pipeline can process.
        Used during development/testing without GPU.
        """
        logger.warning(
            "BioViL-T unavailable. Please install health-multimodal "
            "and ensure a compatible GPU is available for image mode."
        )
        return (
            "Image analysis requires BioViL-T model. "
            "Please install health-multimodal package and retry. "
            "Alternatively use text-only mode."
        )
