"""
api.py
──────────────────────────────────────────────────────────────────
Flask API server — bridge between React frontend and Python pipeline.

HOW TO RUN:
  1. Make sure you are in medverify/ folder
  2. pip install flask flask-cors
  3. python api.py
  4. Server runs on http://localhost:5000

ENDPOINTS:
  POST /verify      → runs pipeline on text input
  POST /verify-image → runs pipeline on image + optional text
  GET  /metrics     → returns real accuracy from eval_results.json
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import MedHallucinationPipeline, setup_logging
from config.settings import SystemConfig

# ── Setup ─────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger("API")

app    = Flask(__name__)
CORS(app)   # allows React on localhost:3000 to call this server

# Load pipeline ONCE when server starts (not on every request)
logger.info("Loading pipeline...")
config   = SystemConfig()
pipeline = MedHallucinationPipeline(config)
logger.info("Pipeline ready.")


# ── Helper ────────────────────────────────────────────────────────────────

def claim_to_dict(c) -> dict:
    """Convert ClaimResult to JSON-serializable dict."""
    return {
        "claim":       c.claim,
        "label":       c.label,
        "confidence":  round(c.confidence, 4),
        "scores":      c.scores,
        "pmid":        c.pmid,
        "evidence":    c.evidence_snippet,
        "explanation": c.explanation,
        "correction":  c.correction,
        "dependency":  {
            "type":             c.dependency.type,
            "depends_on_index": c.dependency.depends_on_index,
            "note":             c.dependency.note,
        } if c.dependency else None,
        "using_your_model": c.using_your_model,
    }


def output_to_dict(result) -> dict:
    """Convert PipelineOutput to JSON-serializable dict."""
    claims = [claim_to_dict(c) for c in result.claims]
    risks  = [c.confidence for c in result.claims]

    return {
        "input_mode":    result.input_mode,
        "input_text":    result.input_text,
        "image_text":    result.image_text,
        "final_verdict": result.final_verdict,
        "claims":        claims,
        "summary": {
            "total_claims":  len(claims),
            "supported":     sum(1 for c in result.claims if c.label == "SUPPORTED"),
            "hallucinated":  sum(1 for c in result.claims if c.label == "HALLUCINATED"),
            "insufficient":  sum(1 for c in result.claims if c.label == "INSUFFICIENT_EVIDENCE"),
            "avg_confidence": round(sum(risks) / len(risks), 4) if risks else 0,
            "min_confidence": round(min(risks), 4) if risks else 0,
        }
    }


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/verify", methods=["POST"])
def verify():
    """
    Verify text input.
    Body: { "text": "medical text to verify" }
    """
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    logger.info(f"Verifying text: {text[:80]}")

    try:
        result = pipeline.run_text(text)
        return jsonify(output_to_dict(result))
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/verify-image", methods=["POST"])
def verify_image():
    """
    Verify image + optional report text.
    Form data:
      image  → image file (JPEG/PNG)
      report → optional text report about the image
    """
    report = request.form.get("report", "").strip()
    image  = request.files.get("image")

    if not image and not report:
        return jsonify({"error": "No image or report provided"}), 400

    try:
        if image:
            # Save uploaded image temporarily
            image_path = f"/tmp/medverify_upload_{image.filename}"
            image.save(image_path)
            result = pipeline.run_image(
                image_path = image_path,
                report     = report if report else None,
            )
            # Clean up
            if os.path.exists(image_path):
                os.remove(image_path)
        else:
            # Report text only — treat as text mode
            result = pipeline.run_text(report)

        return jsonify(output_to_dict(result))

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Returns real accuracy numbers from eval_results.json.
    These are YOUR actual training results — not hardcoded.
    """
    results_path = os.path.join(config.model_path, "eval_results.json")

    if not os.path.exists(results_path):
        return jsonify({"error": "eval_results.json not found. Run training first."}), 404

    with open(results_path) as f:
        data = json.load(f)

    return jsonify({
        "accuracy":         data.get("test_accuracy",  data.get("accuracy",  0)),
        "macro_f1":         data.get("test_macro_f1",  data.get("macro_f1",  0)),
        "precision":        data.get("test_precision", data.get("precision", 0)),
        "recall":           data.get("test_recall",    data.get("recall",    0)),
        "classification_report": data.get("classification_report", ""),
        "datasets_used":    data.get("datasets_used", []),
        "label_scheme":     data.get("label_scheme",  {}),
        "training_history": data.get("training_history", []),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     pipeline.verifier.using_your_model,
        "model_path":       config.model_path,
    })


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("MedVerify API running at http://localhost:5000")
    print("React should call: http://localhost:5000/verify")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
