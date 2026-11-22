"""Corrected order-preserving ROC AUC, EER and FAR-threshold metrics."""
from __future__ import annotations

import math
from typing import Iterable


def _validated(scores: Iterable[float], labels: Iterable[int]) -> tuple[list[float], list[int]]:
    score_values = [float(x) for x in scores]
    raw_labels = list(labels)
    if len(score_values) != len(raw_labels):
        raise ValueError("scores and labels must have equal length")
    if any(not math.isfinite(x) for x in score_values):
        raise ValueError("scores must contain only finite values")
    if any(x not in (0, 1, False, True) for x in raw_labels):
        raise ValueError("labels must contain only 0 and 1")
    return score_values, [int(x) for x in raw_labels]


def metric_selector_score(value, missing: float = -1.0) -> float:
    """Only undefined metrics map to the sentinel; a valid zero remains zero."""
    return float(missing) if value is None else float(value)


def auc_eer(scores: Iterable[float], labels: Iterable[int]) -> dict:
    """AUC uses aligned average ranks for exact ties; EER accepts score >= threshold.

    The EER sweep checks reject-all first and then distinct thresholds in
    descending order. Missing either class returns None for both metrics.
    """
    score_values, label_values = _validated(scores, labels)
    n_pos = sum(label == 1 for label in label_values)
    n_neg = len(label_values) - n_pos
    if not n_pos or not n_neg:
        return {"roc_auc": None, "eer": None, "eer_threshold": None,
                "eer_threshold_boundary": None, "positive_pair_count": n_pos,
                "negative_pair_count": n_neg,
                "undefined_reason": "both positive and negative labels are required"}

    ranked = sorted(zip(score_values, label_values), key=lambda row: row[0])
    rank_sum_positive = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(label == 1 for _, label in ranked[start:end])
        start = end
    auc = (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    ordered = sorted(zip(score_values, label_values), key=lambda row: row[0], reverse=True)
    thresholds = [None] + sorted(set(score_values), reverse=True)
    candidate_rates = []
    for threshold in thresholds:
        if threshold is None:
            true_accept = false_accept = 0.0
        else:
            accepted = [label for score, label in ordered if score >= threshold]
            true_accept = sum(label == 1 for label in accepted) / n_pos
            false_accept = sum(label == 0 for label in accepted) / n_neg
        false_reject = 1.0 - true_accept
        candidate_rates.append((abs(false_accept - false_reject),
                                (false_accept + false_reject) / 2.0, threshold))
    _, eer, threshold = min(candidate_rates, key=lambda row: row[0])
    boundary = threshold is None
    return {"roc_auc": float(auc), "eer": float(eer),
            "eer_threshold": None if boundary else float(threshold),
            "eer_threshold_boundary": "reject_all" if boundary else None,
            "positive_pair_count": n_pos, "negative_pair_count": n_neg}


def far_threshold(scores: Iterable[float], labels: Iterable[int], far_target: float = 0.01) -> float | None:
    score_values, label_values = _validated(scores, labels)
    if not math.isfinite(far_target) or not 0 <= far_target <= 1:
        raise ValueError("far_target must be finite and in [0, 1]")
    negatives = sorted((score for score, label in zip(score_values, label_values) if label == 0), reverse=True)
    if not negatives:
        return None
    allowed = int(math.floor(far_target * len(negatives)))
    if allowed >= len(negatives):
        return float("-inf")
    return math.nextafter(negatives[allowed], math.inf)


def verification_metrics(scores: Iterable[float], labels: Iterable[int],
                         threshold: float | None = None, far_target: float = 0.01) -> dict:
    score_values, label_values = _validated(scores, labels)
    curve = auc_eer(score_values, label_values)
    chosen = far_threshold(score_values, label_values, far_target) if threshold is None else float(threshold)
    positive = [score for score, label in zip(score_values, label_values) if label == 1]
    negative = [score for score, label in zip(score_values, label_values) if label == 0]
    accepts = (lambda score: chosen is not None and score >= chosen)
    return {
        "pair_count": len(score_values),
        "positive_pair_count": len(positive),
        "negative_pair_count": len(negative),
        "roc_auc": curve["roc_auc"],
        "eer": curve["eer"],
        "eer_threshold": curve["eer_threshold"],
        "decision_threshold": chosen,
        "tar_at_decision_threshold": sum(accepts(x) for x in positive) / len(positive) if positive else None,
        "far_at_decision_threshold": sum(accepts(x) for x in negative) / len(negative) if negative and chosen is not None else None,
        "requested_far_target": far_target,
        "undefined_reason": curve.get("undefined_reason"),
    }
