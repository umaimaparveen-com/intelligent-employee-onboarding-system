"""
───────────────────────────────────────────────────────────────────────────────
RAGAS-style evaluation metrics implemented locally (no external API key).

Metrics implemented
────────────────────
1. Faithfulness      – Are all claims in the answer grounded in the retrieved
                       context?  Score ∈ [0, 1].
2. Answer Relevance  – Does the answer actually address the question asked?
                       Score ∈ [0, 1].

Both metrics use LLM-as-a-judge via the local Ollama model so that grading
stays consistent with the same model used for generation.

Hallucination Detection
────────────────────────
A sample is flagged as a hallucination risk when:
    faithfulness < 0.5  OR  answer_relevance < 0.5
This mirrors the RAGAS hallucination heuristic used in production systems.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="logs/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Data model ─────────────────────────────────────────────────────────────
@dataclass
class EvalSample:
    """One evaluation sample."""
    id: str
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_context: str
    category: str = ""
    source_doc: Optional[str] = None


@dataclass
class EvalResult:
    """Scores for one evaluation sample."""
    id: str
    question: str
    category: str
    faithfulness: float
    answer_relevance: float
    hallucination_risk: bool
    faithfulness_reason: str = ""
    relevance_reason: str = ""
    latency_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        """Harmonic mean of faithfulness and answer_relevance."""
        if self.faithfulness + self.answer_relevance == 0:
            return 0.0
        return (2 * self.faithfulness * self.answer_relevance) / (
            self.faithfulness + self.answer_relevance
        )


# ── Prompt templates ────────────────────────────────────────────────────────
_FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["context", "answer"],
    template="""You are an expert AI evaluator. Your job is to judge whether an 
AI-generated answer contains ONLY information that is directly supported by the 
provided context documents.

CONTEXT:
{context}

AI ANSWER:
{answer}

Instructions:
- Read every factual claim in the AI Answer.
- Check each claim against the Context.
- Score the answer from 0.0 to 1.0:
    1.0 = every claim is directly supported by the context
    0.5 = some claims are supported, some are not in the context
    0.0 = the answer is entirely made up / not in the context

Respond ONLY with valid JSON — no extra text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}}
""",
)

_RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an expert AI evaluator. Your job is to judge whether an 
AI-generated answer is relevant and directly addresses the question asked.

QUESTION:
{question}

AI ANSWER:
{answer}

Instructions:
- Score from 0.0 to 1.0:
    1.0 = the answer directly and completely addresses the question
    0.5 = the answer is partially relevant or misses key aspects
    0.0 = the answer does not address the question at all

Respond ONLY with valid JSON — no extra text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}}
""",
)


# ── Evaluator class ─────────────────────────────────────────────────────────
class RAGEvaluator:
    """
    Runs faithfulness + answer-relevance evaluation using LLM-as-a-judge.

    Usage
    ─────
        evaluator = RAGEvaluator()
        results   = evaluator.evaluate_dataset(samples)
        evaluator.print_report(results)
    """

    HALLUCINATION_THRESHOLD = 0.5  # flag if either metric is below this

    def __init__(self, model: str = "llama3", temperature: float = 0.0):
        self.llm = OllamaLLM(model=model, temperature=temperature)
        logger.info(f"RAGEvaluator initialised with model={model}")

    # ── Private helpers ─────────────────────────────────────────────────────
    def _call_judge(self, prompt: str) -> dict:
        """Call the LLM and parse the JSON response safely."""
        try:
            response = self.llm.invoke(prompt)
            # Extract JSON even if the model wraps it in prose
            match = re.search(r"\{.*?\}", response, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in response: {response}")
            return json.loads(match.group())
        except Exception as exc:
            logger.warning(f"Judge parse error: {exc}")
            return {"score": 0.0, "reason": f"Parse error: {exc}"}

    # ── Metric 1: Faithfulness ──────────────────────────────────────────────
    def score_faithfulness(self, context: str, answer: str) -> tuple[float, str]:
        """
        Faithfulness: fraction of answer claims supported by the context.

        Design decision: we use LLM-as-judge instead of NLI models because
        - No GPU required (runs on Ollama locally)
        - The same model that generated the answer does the grading
          which avoids cross-model calibration issues
        """
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        result = self._call_judge(prompt)
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        logger.info(f"Faithfulness score={score:.2f}  reason={reason}")
        return score, reason

    # ── Metric 2: Answer Relevance ─────────────────────────────────────────
    def score_answer_relevance(
        self, question: str, answer: str
    ) -> tuple[float, str]:
        """
        Answer Relevance: does the answer address the question?

        Design decision: we deliberately decouple this from faithfulness —
        an answer can be fully grounded in context but still miss the point
        of the question (e.g., answering the wrong question entirely).
        """
        prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
        result = self._call_judge(prompt)
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        logger.info(f"Answer relevance score={score:.2f}  reason={reason}")
        return score, reason

    # ── Hallucination detection ─────────────────────────────────────────────
    def is_hallucination_risk(
        self, faithfulness: float, relevance: float
    ) -> bool:
        """
        Flag as hallucination risk if EITHER metric is below threshold.

        Production strategy:
        - faithfulness < 0.5 → answer contains unsupported claims
        - relevance   < 0.5 → answer is off-topic (possible confabulation)
        Either condition is enough to warrant human review or a fallback.
        """
        return (
            faithfulness < self.HALLUCINATION_THRESHOLD
            or relevance < self.HALLUCINATION_THRESHOLD
        )

    # ── Evaluate one sample ─────────────────────────────────────────────────
    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Run both metrics on a single sample and return an EvalResult."""
        print(f"\n  📊 Evaluating [{sample.id}] — {sample.question[:60]}...")
        t0 = time.time()

        faithful_score, faithful_reason = self.score_faithfulness(
            sample.retrieved_context, sample.generated_answer
        )
        relevance_score, relevance_reason = self.score_answer_relevance(
            sample.question, sample.generated_answer
        )

        hallucination = self.is_hallucination_risk(faithful_score, relevance_score)
        latency = (time.time() - t0) * 1000

        result = EvalResult(
            id=sample.id,
            question=sample.question,
            category=sample.category,
            faithfulness=faithful_score,
            answer_relevance=relevance_score,
            hallucination_risk=hallucination,
            faithfulness_reason=faithful_reason,
            relevance_reason=relevance_reason,
            latency_ms=latency,
        )

        flag = "🚨 RISK" if hallucination else "✅ OK"
        print(
            f"     Faithfulness={faithful_score:.2f}  "
            f"Relevance={relevance_score:.2f}  "
            f"Overall={result.overall_score:.2f}  {flag}"
        )
        logger.info(
            f"EvalResult id={sample.id} faithfulness={faithful_score:.2f} "
            f"relevance={relevance_score:.2f} hallucination={hallucination}"
        )
        return result

    # ── Evaluate full dataset ───────────────────────────────────────────────
    def evaluate_dataset(self, samples: list[EvalSample]) -> list[EvalResult]:
        """Evaluate all samples in the dataset."""
        print(f"\n🔬 Starting evaluation of {len(samples)} samples...")
        results = [self.evaluate_sample(s) for s in samples]
        logger.info(f"Dataset evaluation complete: {len(results)} samples processed")
        return results

    # ── Summary report ──────────────────────────────────────────────────────
    def print_report(self, results: list[EvalResult]) -> dict:
        """Print a structured summary report and return the aggregated metrics."""
        n = len(results)
        avg_faith = sum(r.faithfulness for r in results) / n
        avg_rel = sum(r.answer_relevance for r in results) / n
        avg_overall = sum(r.overall_score for r in results) / n
        n_risk = sum(1 for r in results if r.hallucination_risk)
        avg_lat = sum(r.latency_ms for r in results) / n

        print("\n" + "=" * 70)
        print("📈  RAG PIPELINE EVALUATION REPORT")
        print("=" * 70)
        print(f"  Samples evaluated   : {n}")
        print(f"  Avg Faithfulness    : {avg_faith:.3f} / 1.000")
        print(f"  Avg Answer Relevance: {avg_rel:.3f} / 1.000")
        print(f"  Avg Overall Score   : {avg_overall:.3f} / 1.000")
        print(f"  Hallucination Risks : {n_risk} / {n} samples")
        print(f"  Avg Eval Latency    : {avg_lat:.0f} ms / sample")
        print("-" * 70)

        # Per-sample table
        print(f"  {'ID':<5} {'Category':<14} {'Faith':>6} {'Relev':>6} {'Overall':>8} {'Risk':<8}")
        print(f"  {'-'*5} {'-'*14} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        for r in results:
            flag = "🚨" if r.hallucination_risk else "✅"
            print(
                f"  {r.id:<5} {r.category:<14} {r.faithfulness:>6.2f} "
                f"{r.answer_relevance:>6.2f} {r.overall_score:>8.2f} {flag}"
            )

        # Hallucination section
        risks = [r for r in results if r.hallucination_risk]
        if risks:
            print("\n  🚨  HALLUCINATION RISK SAMPLES")
            print(f"  {'-'*60}")
            for r in risks:
                print(f"  [{r.id}] {r.question[:65]}")
                print(f"        Faithfulness reason : {r.faithfulness_reason}")
                print(f"        Relevance reason    : {r.relevance_reason}")
        else:
            print("\n  ✅  No hallucination risks detected.")

        print("=" * 70)

        summary = {
            "n_samples": n,
            "avg_faithfulness": round(avg_faith, 4),
            "avg_answer_relevance": round(avg_rel, 4),
            "avg_overall_score": round(avg_overall, 4),
            "hallucination_risk_count": n_risk,
            "avg_eval_latency_ms": round(avg_lat, 1),
        }
        logger.info(f"Evaluation report summary: {summary}")
        return summary
