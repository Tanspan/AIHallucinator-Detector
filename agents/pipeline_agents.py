from __future__ import annotations
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import re
import os
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import requests
from Bio import Entrez

from core.base_agent import BaseAgent


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Evidence:
    abstracts: List[str] = field(default_factory=list)
    pmids:     List[str] = field(default_factory=list)

    def best(self) -> Tuple[Optional[str], Optional[str]]:
        if self.abstracts:
            return self.abstracts[0], self.pmids[0] if self.pmids else None
        return None, None

    def is_empty(self) -> bool:
        return len(self.abstracts) == 0


@dataclass
class DependencyInfo:
    type:             str
    depends_on_index: Optional[int]
    note:             Optional[str]


@dataclass
class ClaimResult:
    claim:            str
    label:            str
    confidence:       float
    scores:           dict
    pmid:             Optional[str]
    evidence_snippet: Optional[str]
    explanation:      Optional[str]
    correction:       Optional[str]
    dependency:       Optional[DependencyInfo]
    using_your_model: bool


@dataclass
class PipelineOutput:
    input_text:    str
    input_mode:    str
    image_text:    Optional[str]
    claims:        List[ClaimResult]
    final_verdict: str


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — ClaimDecomposerAgent
# Method: spaCy sentence segmentation
# Source: Honnibal et al., spaCy, https://spacy.io
# ══════════════════════════════════════════════════════════════════

class ClaimDecomposerAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self.log("spaCy loaded.")
        except Exception as e:
            self.log(f"spaCy unavailable, using regex: {e}", "warning")

    def decompose(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if self._nlp:
            doc = self._nlp(text)
            sentences = [s.text.strip() for s in doc.sents
                         if len(s.text.strip()) > 8]
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text)
                         if len(s.strip()) > 8]
        self.log(f"Decomposed into {len(sentences)} claim(s).")
        return sentences


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — DependencyAnalyserAgent  ← YOUR NOVEL AGENT
# FActScore (Min et al. 2023) treats each atomic fact independently.
# This agent detects logical dependencies between claims.
# No existing medical hallucination paper models this.
# ══════════════════════════════════════════════════════════════════

_CONCLUSIVE  = ["therefore", "thus", "hence", "consequently",
                "as a result", "it follows"]
_CAUSAL      = ["because", "since", "due to", "given that", "as a result of"]
_CONDITIONAL = ["if ", "when ", "provided that", "assuming that"]


class DependencyAnalyserAgent(BaseAgent):

    def analyse(self, claims: List[str]) -> List[DependencyInfo]:
        return [self._check(c, i) for i, c in enumerate(claims)]

    def _check(self, claim: str, idx: int) -> DependencyInfo:
        cl = claim.lower()
        for m in _CONCLUSIVE:
            if m in cl:
                return DependencyInfo(
                    type             = "CONCLUSIVE",
                    depends_on_index = idx - 1 if idx > 0 else None,
                    note = (
                        f"Claim uses '{m}' — presented as a logical conclusion. "
                        f"The inference chain itself must be verified independently. "
                        f"Reducing/managing a symptom does not imply curing a disease."
                    )
                )
        for m in _CAUSAL:
            if m in cl:
                return DependencyInfo(
                    type             = "CAUSAL",
                    depends_on_index = idx - 1 if idx > 0 else None,
                    note = (
                        f"Claim uses '{m}' — presents a causal relationship. "
                        f"Causal claims require stronger evidence than correlational."
                    )
                )
        for m in _CONDITIONAL:
            if cl.startswith(m):
                return DependencyInfo(
                    type             = "CONDITIONAL",
                    depends_on_index = None,
                    note = "Claim is conditional — validity depends on stated conditions."
                )
        return DependencyInfo(type="INDEPENDENT", depends_on_index=None, note=None)


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — MultiSourceRetrievalAgent
# THREE evidence sources — all free, no authentication:
# Source 1: PubMed (NCBI Entrez) — 35M papers
# Source 2: Semantic Scholar API — 200M papers
# Source 3: Europe PMC REST API  — 40M papers
# All merged → scored by key word match → best evidence selected
# ══════════════════════════════════════════════════════════════════

STOPWORDS = {
    "that","this","with","from","have","been","were","they",
    "their","there","when","which","what","also","into","more",
    "than","then","completely","works","used","about","does",
    "very","only","both","each","some","such","after","before",
    "while","where","just","against","these","those","other",
    "would","could","should","always","never","every","cured",
    "taking","treated","treatment","weeks","months","given",
    "patients","studies","study","results","data","analysis",
    "therefore","thus","hence","consequently","eliminates",
    "eliminate","cures","cure","causes","cause"
}


def build_query(claim: str) -> str:
    words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim.lower())
             if w not in STOPWORDS]
    words = sorted(set(words), key=len, reverse=True)
    return " ".join(words[:5])


class PubMedSource(BaseAgent):

    def __init__(self, email: str, max_results: int = 5):
        super().__init__()
        Entrez.email = email
        self.max_results = max_results

    def fetch(self, claim: str, n: int) -> Evidence:
        query = build_query(claim)
        self.log(f"PubMed: '{query}'")
        try:
            handle   = Entrez.esearch(db="pubmed", term=query, retmax=n)
            record   = Entrez.read(handle)
            handle.close()
            ids      = record.get("IdList", [])
            if not ids:
                return Evidence()
            fetch    = Entrez.efetch(db="pubmed", id=",".join(ids),
                                     rettype="abstract", retmode="xml")
            articles = Entrez.read(fetch)
            fetch.close()
            abstracts, pmids = [], []
            for i, art in enumerate(articles.get("PubmedArticle", [])):
                try:
                    txt   = art["MedlineCitation"]["Article"]["Abstract"][
                                "AbstractText"][0]
                    clean = re.sub(r'<[^>]+>', '', str(txt)).strip()
                    abstracts.append(clean)
                    pmids.append(str(ids[i]) if i < len(ids) else "")
                except (KeyError, IndexError):
                    continue
            return Evidence(abstracts=abstracts, pmids=pmids)
        except Exception as e:
            self.log(f"PubMed error: {e}", "warning")
            return Evidence()


class SemanticScholarSource(BaseAgent):

    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def fetch(self, claim: str, n: int) -> Evidence:
        query = build_query(claim)
        self.log(f"Semantic Scholar: '{query}'")
        try:
            resp = requests.get(
                self.BASE,
                params={"query": query, "limit": n,
                        "fields": "title,abstract,externalIds"},
                timeout=10,
                headers={"User-Agent": "MedVerify/1.0"}
            )
            if resp.status_code != 200:
                return Evidence()
            papers = resp.json().get("data", [])
            abstracts, pmids = [], []
            for p in papers:
                abstract = p.get("abstract", "")
                if not abstract or len(abstract) < 50:
                    continue
                clean = re.sub(r'<[^>]+>', '', abstract).strip()
                abstracts.append(clean)
                ext   = p.get("externalIds", {}) or {}
                pmids.append(str(ext.get("PubMed", "")))
            return Evidence(abstracts=abstracts, pmids=pmids)
        except Exception as e:
            self.log(f"Semantic Scholar error: {e}", "warning")
            return Evidence()


class EuropePMCSource(BaseAgent):

    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def fetch(self, claim: str, n: int) -> Evidence:
        query = build_query(claim)
        self.log(f"Europe PMC: '{query}'")
        try:
            resp = requests.get(
                self.BASE,
                params={"query": query, "resultType": "core",
                        "pageSize": n, "format": "json", "src": "MED"},
                timeout=10,
            )
            if resp.status_code != 200:
                return Evidence()
            results  = resp.json().get("resultList", {}).get("result", [])
            abstracts, pmids = [], []
            for r in results:
                abstract = r.get("abstractText", "")
                if not abstract or len(abstract) < 50:
                    continue
                clean = re.sub(r'<[^>]+>', '', abstract).strip()
                abstracts.append(clean)
                pmids.append(str(r.get("pmid", "")))
            return Evidence(abstracts=abstracts, pmids=pmids)
        except Exception as e:
            self.log(f"Europe PMC error: {e}", "warning")
            return Evidence()


class MultiSourceRetrievalAgent(BaseAgent):
    """
    Combines PubMed + Semantic Scholar + Europe PMC.
    All 3 searched → key word validation → best evidence returned.
    """

    def __init__(self, email: str, max_results: int = 5):
        super().__init__()
        self.pubmed    = PubMedSource(email, max_results)
        self.semantic  = SemanticScholarSource()
        self.europepmc = EuropePMCSource()
        self.max_results = max_results

    def retrieve(self, claim: str, expanded: bool = False) -> Evidence:
        n = self.max_results + 2 if expanded else self.max_results

        ev1 = self.pubmed.fetch(claim, n)
        ev2 = self.semantic.fetch(claim, n)
        ev3 = self.europepmc.fetch(claim, n)

        all_abstracts = ev1.abstracts + ev2.abstracts + ev3.abstracts
        all_pmids     = ev1.pmids     + ev2.pmids     + ev3.pmids

        if not all_abstracts:
            self.log("No evidence from any source.")
            return Evidence()

        self.log(f"Total: {len(all_abstracts)} "
                 f"(PubMed:{len(ev1.abstracts)} "
                 f"S2:{len(ev2.abstracts)} "
                 f"EPMC:{len(ev3.abstracts)})")

        # Key word validation
        # Top 3 longest words = most specific medical terms in claim
        claim_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim.lower())
            if w.lower() not in STOPWORDS
        )
        key_words = sorted(claim_words, key=len, reverse=True)[:3]

        # Score each abstract — must contain at least 2 of top 3 key words
        scored = []
        for ab, pid in zip(all_abstracts, all_pmids):
            ab_words  = set(re.findall(r'\b[a-zA-Z]{4,}\b', ab.lower()))
            key_score = sum(1 for kw in key_words if kw in ab_words)
            total     = len(claim_words & ab_words)
            if key_score >= 2:
                scored.append((ab, pid, key_score * 10 + total))

        if scored:
            scored.sort(key=lambda x: x[2], reverse=True)
            self.log(f"After validation: {len(scored)} relevant abstracts.")
            return Evidence(
                abstracts=[s[0] for s in scored],
                pmids    =[s[1] for s in scored]
            )

        # Fallback: try with 1 key word
        scored_fb = []
        for ab, pid in zip(all_abstracts, all_pmids):
            ab_words  = set(re.findall(r'\b[a-zA-Z]{4,}\b', ab.lower()))
            key_score = sum(1 for kw in key_words if kw in ab_words)
            if key_score >= 1:
                scored_fb.append((ab, pid, key_score))

        if scored_fb:
            scored_fb.sort(key=lambda x: x[2], reverse=True)
            self.log(f"Fallback: {len(scored_fb)} abstracts.")
            return Evidence(
                abstracts=[s[0] for s in scored_fb],
                pmids    =[s[1] for s in scored_fb]
            )

        self.log("No relevant evidence found.")
        return Evidence()


# ══════════════════════════════════════════════════════════════════
# AGENT 5 — OntologyGroundingAgent
# Uses scispaCy — no authentication required
# Cite: Neumann et al. 2019, ACL BioNLP Workshop
# ══════════════════════════════════════════════════════════════════

class OntologyGroundingAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_sci_lg")
            self.log("scispaCy en_core_sci_lg loaded.")
        except Exception as e:
            self.log(f"scispaCy not available: {e}", "warning")

    def ground(self, claim: str) -> float:
        if self._nlp is None:
            return 0.5
        doc      = self._nlp(claim)
        entities = [ent.text for ent in doc.ents]
        self.log(f"Medical entities: {entities}")
        return 1.0 if entities else 0.0


# ══════════════════════════════════════════════════════════════════
# AGENT 6 — ConfidenceAggregatorAgent
# Trusts your model's classification (94.3% accuracy).
# Only intervenes for low-confidence uncertain claims.
# ══════════════════════════════════════════════════════════════════

class ConfidenceAggregatorAgent(BaseAgent):

    def __init__(self, uncertainty_threshold: float = 0.60):
        super().__init__()
        self.threshold = uncertainty_threshold

    def adjust(self, verify_result, ontology_score: float,
               claim: str = "", evidence: str = "") -> str:
        label      = verify_result.label
        confidence = verify_result.confidence

        # Only upgrade to INSUFFICIENT_EVIDENCE if model is uncertain
        # AND no medical entity found — honest abstention
        if (confidence < self.threshold and
                ontology_score == 0.0 and
                label != "INSUFFICIENT_EVIDENCE"):
            self.log(
                f"Low confidence ({confidence:.3f}) + no medical entity "
                f"→ upgrading to INSUFFICIENT_EVIDENCE"
            )
            label = "INSUFFICIENT_EVIDENCE"

        self.log(f"Final label: {label} (confidence={confidence:.3f})")
        return label


# ══════════════════════════════════════════════════════════════════
# AGENT 7 — AdaptivePlannerAgent  ← YOUR NOVEL AGENT
# FActScore uses fixed top-k retrieval.
# This agent expands search for uncertain claims.
# ══════════════════════════════════════════════════════════════════

class AdaptivePlannerAgent(BaseAgent):

    def __init__(self, expand_threshold: float = 0.65, max_iterations: int = 2):
        super().__init__()
        self.expand_threshold = expand_threshold
        self.max_iterations   = max_iterations

    def should_expand(self, confidence: float, iteration: int) -> bool:
        if iteration >= self.max_iterations:
            self.log("Max iterations reached → STOP")
            return False
        if confidence < self.expand_threshold:
            self.log(f"Low confidence ({confidence:.3f}) → EXPAND search")
            return True
        self.log(f"Confidence {confidence:.3f} sufficient → STOP")
        return False


# ══════════════════════════════════════════════════════════════════
# AGENT 8 — ExplainerCorrectorAgent  ← YOUR NOVEL AGENT
# Existing systems output label only.
# Your system: label + explanation + correction + PMID.
# Uses Groq API (FREE) — get key at console.groq.com
# Falls back to evidence extraction if key not set.
# ══════════════════════════════════════════════════════════════════

class ExplainerCorrectorAgent(BaseAgent):

    def __init__(self, model_name: str = "google/flan-t5-base",
                 max_input: int = 512, max_output: int = 200,
                 groq_api_key: str = ""):
        super().__init__()
        self.model_name   = model_name
        self.max_input    = max_input
        self.max_output   = max_output
        self.groq_api_key = groq_api_key
        self.device       = torch.device("cpu")
        self._tok         = None
        self._model       = None

    def explain_and_correct(
        self, claim: str, label: str,
        evidence: Optional[str], pmid: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:

        if label != "HALLUCINATED" or not evidence:
            return None, None, None

        ev = evidence[:600]

        explanation, correction = self._groq_generate(claim, ev)

        if not explanation or not correction:
            self.log("Groq unavailable, using fallback", "warning")
            correction  = self._extract_correction(claim, evidence)
            explanation = self._flanT5_explain(claim, ev)

        if not explanation or len(explanation) < 20:
            explanation = (
                f"The claim is incorrect because the evidence states: "
                f"{correction[:150]}"
            )

        return explanation, correction, pmid

    def _groq_generate(self, claim: str, evidence: str):
        """
        Groq API — free 14,400 requests/day.
        Model: llama-3.3-70b-versatile
        Get key: https://console.groq.com
        Set in config/settings.py → groq_api_key
        """
        api_key = (self.groq_api_key or
                   os.environ.get("GROQ_API_KEY", "")).strip()

        if not api_key or api_key == "your_groq_key_here":
            self.log("Groq key not set. Using fallback.", "warning")
            return None, None

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a medical fact-checker. "
                                "Give accurate concise corrections "
                                "based only on provided evidence."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Evidence from peer-reviewed literature:\n"
                                f"{evidence}\n\n"
                                f"The following medical claim is INCORRECT:\n"
                                f"Claim: {claim}\n\n"
                                f"Based ONLY on the evidence, provide:\n"
                                f"1. EXPLANATION: One sentence starting with "
                                f"'The claim is incorrect because'\n"
                                f"2. CORRECTION: One accurate sentence fixing "
                                f"the claim. Must differ from original.\n\n"
                                f"Respond in exactly this format:\n"
                                f"EXPLANATION: <one sentence>\n"
                                f"CORRECTION: <one sentence>"
                            )
                        }
                    ],
                    "max_tokens":  200,
                    "temperature": 0.1,
                },
                timeout=15,
            )

            if resp.status_code != 200:
                self.log(f"Groq error {resp.status_code}", "warning")
                return None, None

            text = resp.json()["choices"][0]["message"]["content"].strip()
            self.log(f"Groq: {text[:80]}")

            explanation = ""
            correction  = ""
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()
                elif line.startswith("CORRECTION:"):
                    correction = line.replace("CORRECTION:", "").strip()

            if explanation and correction:
                return explanation, correction
            return None, None

        except Exception as e:
            self.log(f"Groq error: {e}", "warning")
            return None, None

    def _extract_correction(self, claim: str, evidence: str) -> str:
        """Fallback: extract most relevant sentence from evidence."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', evidence)
                     if len(s.strip()) > 20]
        if not sentences:
            return "See linked evidence for accurate information."

        stopwords = {
            "that","this","with","from","have","been","were","they",
            "their","there","when","which","what","also","into","more",
            "than","then","completely","works","used","about","does",
            "very","only","both","some","such","after","before","since",
            "while","just","always","never","every","all"
        }
        claim_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', claim)
            if w.lower() not in stopwords
        )
        best_sentence = None
        best_score    = -1
        for sent in sentences:
            sent_words = set(
                w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', sent)
            )
            overlap = len(claim_words & sent_words)
            if overlap > best_score:
                best_score    = overlap
                best_sentence = sent

        if best_sentence and best_score >= 2:
            return best_sentence
        return sentences[0]

    def _flanT5_explain(self, claim: str, evidence: str) -> str:
        """Fallback explanation using flan-t5-base."""
        self._load()
        prompt = (
            f"Evidence:\n{evidence}\n\n"
            f"Claim: {claim}\n\n"
            f"In one sentence starting with 'The claim is incorrect because', "
            f"explain what the evidence says instead:"
        )
        return self._generate(prompt)

    def _load(self):
        if self._model is not None:
            return
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        self.log(f"Loading {self.model_name}...")
        self._tok   = T5Tokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()

    def _generate(self, prompt: str) -> str:
        inputs = self._tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_input
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens       = self.max_output,
                num_beams            = 4,
                early_stopping       = True,
                no_repeat_ngram_size = 3,
            )
        return self._tok.decode(out[0], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════
# AGENT 9 — FinalAggregatorAgent
# ANY claim HALLUCINATED           → HALLUCINATED
# ALL claims SUPPORTED             → FACTUAL
# ALL claims INSUFFICIENT_EVIDENCE → INSUFFICIENT_EVIDENCE
# Mix SUPPORTED + INSUFFICIENT     → PARTIALLY_HALLUCINATED
# False inference chain detected   → HALLUCINATED
# ══════════════════════════════════════════════════════════════════

class FinalAggregatorAgent(BaseAgent):

    def aggregate(self, results: List[ClaimResult]) -> str:
        if not results:
            return "INSUFFICIENT_EVIDENCE"

        labels = [r.label for r in results]

        has_false_inference = any(
            r.dependency and r.dependency.type in ("CONCLUSIVE", "CAUSAL")
            and r.label == "HALLUCINATED"
            for r in results
        )
        if has_false_inference:
            return "HALLUCINATED"

        if any(l == "HALLUCINATED" for l in labels):
            return "HALLUCINATED"

        if all(l == "INSUFFICIENT_EVIDENCE" for l in labels):
            return "INSUFFICIENT_EVIDENCE"

        if any(l == "INSUFFICIENT_EVIDENCE" for l in labels):
            return "PARTIALLY_HALLUCINATED"

        return "FACTUAL"