from __future__ import annotations

from typing import Iterable


def token_count(text: str) -> int:
    return len(text.split())


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def overlap_f1(reference: str, candidate: str) -> float:
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    common = len(ref_set & cand_set)
    precision = common / len(cand_set)
    recall = common / len(ref_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def try_compute_rouge_l(references: list[str], predictions: list[str]) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    values = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions, strict=False)]
    return average(values)


def try_compute_bertscore_f1(references: list[str], predictions: list[str]) -> float | None:
    try:
        from bert_score import score
    except ImportError:
        return None
    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return float(f1.mean().item())
