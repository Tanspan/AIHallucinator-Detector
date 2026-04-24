from __future__ import annotations
import logging
from typing import Optional

from config.settings import SystemConfig
from agents.medverify_agent import MedVerifyAgent
from agents.pipeline_agents import (
    ClaimDecomposerAgent,
    DependencyAnalyserAgent,
    MultiSourceRetrievalAgent,
    OntologyGroundingAgent,
    ConfidenceAggregatorAgent,
    AdaptivePlannerAgent,
    ExplainerCorrectorAgent,
    FinalAggregatorAgent,
    ClaimResult,
    PipelineOutput,
)


def setup_logging():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt = "%H:%M:%S",
    )


class MedHallucinationPipeline:

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger("Pipeline")
        self.logger.info("Initialising pipeline...")

        self.decomposer  = ClaimDecomposerAgent()
        self.dependency  = DependencyAnalyserAgent()
        self.retriever   = MultiSourceRetrievalAgent(
            email       = config.pubmed_email,
            max_results = config.pubmed_max_results,
        )
        self.verifier    = MedVerifyAgent(
            model_path = config.model_path,
            max_length = config.max_length,
        )
        self.ontology    = OntologyGroundingAgent()
        self.confidence  = ConfidenceAggregatorAgent(
            uncertainty_threshold = config.uncertainty_threshold
        )
        self.planner     = AdaptivePlannerAgent(
            expand_threshold = config.expand_threshold,
            max_iterations   = config.max_planner_iterations,
        )
        self.explainer   = ExplainerCorrectorAgent(
            model_name   = config.corrector_model,
            groq_api_key = config.groq_api_key,
        )
        self.aggregator  = FinalAggregatorAgent()

        self.logger.info("All 9 agents ready.")

        if not self.verifier.using_your_model:
            self.logger.warning(
                "Using fallback model. Run python training/train.py first."
            )

    def run_text(self, text: str) -> PipelineOutput:
        return self._run(text=text, mode="text", image_text=None)

    def run_image(self, image_path: str,
                  report: Optional[str] = None) -> PipelineOutput:
        if report:
            image_text = report
        else:
            from vision.biovil_reader import BioViLReader
            reader     = BioViLReader()
            image_text = reader.generate_findings(image_path)
            self.logger.info(f"BioViL-T: {image_text[:100]}")
        return self._run(text=image_text, mode="image+text",
                         image_text=image_text)

    def _run(self, text: str, mode: str,
             image_text: Optional[str]) -> PipelineOutput:
        self.logger.info("=" * 55)
        self.logger.info(f"INPUT ({mode}): {text[:100]}")

        claims       = self.decomposer.decompose(text)
        dependencies = self.dependency.analyse(claims)

        claim_results = []
        prev_evidence = ""
        prev_pmid     = ""

        for i, (claim, dep) in enumerate(zip(claims, dependencies)):
            self.logger.info(f"Claim {i+1}/{len(claims)}: '{claim}'")
            result = self._process_claim(
                claim, dep,
                previous_evidence = prev_evidence,
                previous_pmid     = prev_pmid,
            )
            # Pass evidence to next claim (for dependency chains)
            prev_evidence = result.evidence_snippet or prev_evidence
            prev_pmid     = result.pmid or prev_pmid
            claim_results.append(result)

        verdict = self.aggregator.aggregate(claim_results)
        self.logger.info(f"FINAL VERDICT: {verdict}")

        return PipelineOutput(
            input_text    = text,
            input_mode    = mode,
            image_text    = image_text,
            claims        = claim_results,
            final_verdict = verdict,
        )

    def _process_claim(
        self, claim: str, dep,
        previous_evidence: str = "",
        previous_pmid:     str = "",
    ) -> ClaimResult:
        """
        Process one claim through agents 3-8.

        For CONCLUSIVE/CAUSAL dependency claims:
          Reuse previous claim's evidence — no new PubMed search needed.
          "Therefore X" should be verified against same evidence as
          the claim it derives from.

        For INDEPENDENT claims:
          Search PubMed + Semantic Scholar + Europe PMC as normal.
        """
        iteration = 0

        # CONCLUSIVE/CAUSAL — reuse previous evidence
        if dep.type in ("CONCLUSIVE", "CAUSAL") and previous_evidence:
            self.logger.info(
                f"Dependency ({dep.type}) → reusing previous evidence"
            )
            best_evidence = previous_evidence
            best_pmid     = previous_pmid

        else:
            # Normal retrieval for independent claims
            evidence                 = self.retriever.retrieve(claim)
            best_evidence, best_pmid = evidence.best()

            # Adaptive planning — expand if uncertain
            verify_result = self.verifier.verify(
                claim, best_evidence or ""
            )
            ont_score = self.ontology.ground(claim)

            while self.planner.should_expand(
                verify_result.confidence, iteration
            ):
                iteration += 1
                evidence   = self.retriever.retrieve(claim, expanded=True)
                best_evidence, best_pmid = evidence.best()
                verify_result = self.verifier.verify(
                    claim, best_evidence or ""
                )

            label = self.confidence.adjust(
                verify_result, ont_score,
                claim    = claim,
                evidence = best_evidence or ""
            )

            explanation, correction, pmid = self.explainer.explain_and_correct(
                claim, label, best_evidence, best_pmid
            )

            return ClaimResult(
                claim            = claim,
                label            = label,
                confidence       = verify_result.confidence,
                scores           = verify_result.scores,
                pmid             = pmid or best_pmid,
                evidence_snippet = (best_evidence[:200] + "…")
                                   if best_evidence and
                                   len(best_evidence) > 200
                                   else best_evidence,
                explanation      = explanation,
                correction       = correction,
                dependency       = dep,
                using_your_model = verify_result.using_your_model,
            )

        # CONCLUSIVE/CAUSAL — classify using previous evidence
        verify_result = self.verifier.verify(
            claim, best_evidence or ""
        )
        ont_score = self.ontology.ground(claim)
        label     = self.confidence.adjust(
            verify_result, ont_score,
            claim    = claim,
            evidence = best_evidence or ""
        )

        explanation, correction, pmid = self.explainer.explain_and_correct(
            claim, label, best_evidence, best_pmid
        )

        return ClaimResult(
            claim            = claim,
            label            = label,
            confidence       = verify_result.confidence,
            scores           = verify_result.scores,
            pmid             = pmid or best_pmid,
            evidence_snippet = (best_evidence[:200] + "…")
                               if best_evidence and len(best_evidence) > 200
                               else best_evidence,
            explanation      = explanation,
            correction       = correction,
            dependency       = dep,
            using_your_model = verify_result.using_your_model,
        )