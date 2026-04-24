from dataclasses import dataclass


@dataclass
class SystemConfig:

    # Your email for NCBI PubMed API
    # Register free at: https://www.ncbi.nlm.nih.gov/
    

    # Groq API key — FREE at https://console.groq.com
    # Sign up → API Keys → Create → paste here
    # Free tier: 14,400 requests/day
    
    # Path to your fine-tuned model (from training/train.py)
    
    # PubMed retrieval
   
    # Must match training/train.py max_length
    
    # Corrector model (fallback if Groq unavailable)
    
    # Vision model for image mode (chest X-ray only)
    # Cite: Bannur et al. 2023, MICCAI
    
    # Expand PubMed search if confidence below this
   

    # Upgrade to INSUFFICIENT_EVIDENCE if confidence below this
    uncertainty_threshold: float = 0.60
