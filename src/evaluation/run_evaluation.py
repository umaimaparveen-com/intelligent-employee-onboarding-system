"""
───────────────────────────────────────────────────────────────────────────────
End-to-end evaluation driver.

Steps
─────
1. Load the ground-truth evaluation dataset from data/eval_dataset.json
2. Load (or build) the RAG vector store
3. For each question, run the RAG chain to get a generated answer and the
   retrieved context chunks
4. Pass samples to RAGEvaluator for faithfulness + relevance scoring
5. Print summary report and save results to logs/eval_results.json

Run with:
    python -m src.evaluation.run_evaluation        (from project root)
  or:
    python src/evaluation/run_evaluation.py        (from project root)
"""

import json
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on PYTHONPATH when run as a script ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.pipeline import (
    load_documents,
    chunk_documents,
    create_vectorstore,
    load_vectorstore,
    build_rag_chain,
)
from src.evaluation.evaluator import EvalSample, RAGEvaluator

# ── Paths ───────────────────────────────────────────────────────────────────
EVAL_DATASET_PATH = Path("data/eval_dataset.json")
CHROMA_PATH       = Path("data/chroma_db")
RESULTS_PATH      = Path("logs/eval_results.json")

logging.basicConfig(
    filename="logs/run_evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helper: retrieve context chunks for a question ──────────────────────────
def get_context_chunks(vectorstore, question: str, k: int = 3) -> str:
    """Return the top-k retrieved chunks as a single string."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🔬  RAG PIPELINE EVALUATION  —  Amiseq Onboarding Assistant")
    print("=" * 70)

    # 1. Load eval dataset
    print(f"\n📂 Loading evaluation dataset from {EVAL_DATASET_PATH}...")
    with open(EVAL_DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"   ✅  {len(dataset)} samples loaded")

    # 2. Load or build vector store
    if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
        print("\n📦 Found existing ChromaDB — loading vector store...")
        vectorstore = load_vectorstore()
    else:
        print("\n🔨 No ChromaDB found — building from scratch...")
        docs   = load_documents()
        chunks = chunk_documents(docs)
        vectorstore = create_vectorstore(chunks)

    # 3. Build RAG chain
    rag_chain = build_rag_chain(vectorstore)

    # 4. Generate answers + collect retrieved context
    print("\n🤖 Generating answers for all evaluation samples...")
    samples: list[EvalSample] = []

    for item in dataset:
        q = item["question"]
        print(f"  ❓ [{item['id']}] {q}")
        t0 = time.time()

        # Get retrieved context (needed for faithfulness metric)
        context = get_context_chunks(vectorstore, q)

        # Get generated answer from RAG chain
        try:
            generated_answer = rag_chain.invoke(q)
        except Exception as exc:
            logger.error(f"RAG chain failed for {item['id']}: {exc}")
            generated_answer = f"[Error: {exc}]"

        elapsed = (time.time() - t0) * 1000
        print(f"     ✅  Answer generated in {elapsed:.0f}ms")
        logger.info(
            f"Generated answer for {item['id']} in {elapsed:.0f}ms: "
            f"{generated_answer[:100]}"
        )

        samples.append(
            EvalSample(
                id=item["id"],
                question=q,
                ground_truth=item["ground_truth"],
                generated_answer=generated_answer,
                retrieved_context=context,
                category=item.get("category", ""),
                source_doc=item.get("source_doc"),
            )
        )

    # 5. Evaluate
    evaluator = RAGEvaluator(model="llama3")
    results   = evaluator.evaluate_dataset(samples)

    # 6. Print report
    summary = evaluator.print_report(results)

    # 7. Save full results to JSON
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    output = {
        "summary": summary,
        "samples": [
            {
                "id":                r.id,
                "question":          r.question,
                "category":          r.category,
                "faithfulness":      r.faithfulness,
                "faithfulness_reason": r.faithfulness_reason,
                "answer_relevance":  r.answer_relevance,
                "relevance_reason":  r.relevance_reason,
                "overall_score":     r.overall_score,
                "hallucination_risk": r.hallucination_risk,
                "latency_ms":        r.latency_ms,
            }
            for r in results
        ],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾  Full results saved to {RESULTS_PATH}")
    print("\n✅  Evaluation complete.\n")


if __name__ == "__main__":
    main()
