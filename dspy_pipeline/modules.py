"""DSPy modules for the multi-agent perturbation prediction pipeline.

Each module corresponds to one reasoning stage:
  Characterize → RetrieveEvidence → Synthesize → Calibrate

The pipeline uses ChainOfThought with tool access for evidence retrieval
and a structured reasoning chain for synthesis and calibration.
"""

import dspy

from .signatures import Characterize, RetrieveEvidence, Synthesize, Calibrate
from .tools import TOOLS


class CharacterizeGene(dspy.Module):
    """Characterize a single gene's function using gene_info."""

    def __init__(self):
        super().__init__()
        self.characterize = dspy.ChainOfThought(Characterize)

    def forward(self, gene_symbol: str):
        return self.characterize(gene_symbol=gene_symbol)


class RetrieveEvidenceModule(dspy.Module):
    """Retrieve all evidence for a gene pair using available tools.

    Uses ReAct (Reason + Act) pattern so the LLM can decide which tools
    to call based on partial results. This is more flexible than a fixed
    sequence and stays within the 250-call budget.
    """

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.ReAct(RetrieveEvidence, tools=TOOLS, max_iters=12)

    def forward(self, gene_X: str, gene_Y: str):
        return self.retrieve(gene_X=gene_X, gene_Y=gene_Y)


class SynthesizeModule(dspy.Module):
    """Synthesize evidence into a biological reasoning chain with conflict resolution."""

    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(Synthesize)

    def forward(self, **kwargs):
        return self.synthesize(**kwargs)


class CalibrateModule(dspy.Module):
    """Map qualitative prediction to calibrated continuous probabilities."""

    def __init__(self):
        super().__init__()
        self.calibrate = dspy.ChainOfThought(Calibrate)

    def forward(self, **kwargs):
        return self.calibrate(**kwargs)


class PerturbationPredictor(dspy.Module):
    """Full multi-agent pipeline: Characterize → Retrieve → Synthesize → Calibrate."""

    def __init__(self):
        super().__init__()
        self.characterize_x = CharacterizeGene()
        self.characterize_y = CharacterizeGene()
        self.retrieve = RetrieveEvidenceModule()
        self.synthesize = SynthesizeModule()
        self.calibrate = CalibrateModule()

    def forward(self, gene_X: str, gene_Y: str):
        # Stage 1: Characterize both genes
        x_char = self.characterize_x(gene_symbol=gene_X)
        y_char = self.characterize_y(gene_symbol=gene_Y)

        # Stage 2: Retrieve evidence using tools
        evidence = self.retrieve(gene_X=gene_X, gene_Y=gene_Y)

        # Stage 3: Synthesize into reasoning chain with conflict resolution
        synthesis = self.synthesize(
            gene_X=gene_X,
            gene_Y=gene_Y,
            X_info=evidence.X_info,
            Y_info=evidence.Y_info,
            string_evidence=evidence.string_evidence,
            pathway_overlap=evidence.pathway_overlap,
            go_similarity=evidence.go_similarity,
            surrogate_prediction=evidence.surrogate_prediction,
            similar_perturbations=evidence.similar_perturbations,
        )

        # Stage 4: Calibrate to continuous probabilities
        result = self.calibrate(
            predicted_direction=synthesis.predicted_direction,
            confidence=synthesis.confidence,
            biological_reasoning=synthesis.biological_reasoning,
            surrogate_prediction=evidence.surrogate_prediction,
        )

        return dspy.Prediction(
            prediction_up=result.prediction_up,
            prediction_down=result.prediction_down,
            reasoning_trace=result.reasoning_trace,
        )


class FastPredictor(dspy.Module):
    """Lightweight predictor: retrieve evidence + synthesize in one ReAct step.

    Faster and uses fewer tokens. Good for budget-constrained scenarios.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ReAct(
            dspy.Signature(
                "gene_X, gene_Y -> prediction_up, prediction_down, reasoning_trace",
                "Predict whether CRISPRi knockdown of gene_X causes gene_Y to be "
                "up-regulated, down-regulated, or unchanged in mouse BMDMs. "
                "Use tools to gather evidence, reason about biology, and output "
                "calibrated probabilities (prediction_up + prediction_down ≤ 1.0).",
            ),
            tools=TOOLS,
            max_iters=8,
        )

    def forward(self, gene_X: str, gene_Y: str):
        return self.predict(gene_X=gene_X, gene_Y=gene_Y)
