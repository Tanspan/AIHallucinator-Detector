from dataclasses import dataclass


@dataclass
class SystemConfig:

    # Your email for NCBI PubMed API
    # Register free at: https://www.ncbi.nlm.nih.gov/
    pubmed_email: str = "tanusrikaranam@gmail.com"   # ← CHANGE THIS

    # Groq API key — FREE at https://console.groq.com
    # Sign up → API Keys → Create → paste here
    # Free tier: 14,400 requests/day
    groq_api_key: str = "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"   # ← PASTE YOUR GROQ KEY HERE

    # Path to your fine-tuned model (from training/train.py)
    model_path: str = "./saved_model"

    # PubMed retrieval
    pubmed_max_results: int = 5

    # Must match training/train.py max_length
    max_length: int = 256

    # Corrector model (fallback if Groq unavailable)
    corrector_model: str = "google/flan-t5-base"

    # Vision model for image mode (chest X-ray only)
    # Cite: Bannur et al. 2023, MICCAI
    vision_model: str = "microsoft/BioViL-T"

    # Expand PubMed search if confidence below this
    expand_threshold: float = 0.65
    max_planner_iterations: int = 2

    # Upgrade to INSUFFICIENT_EVIDENCE if confidence below this
    uncertainty_threshold: float = 0.60