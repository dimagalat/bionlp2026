"""
Evaluation utilities for DMRS defense-level predictions.

Computes accuracy, macro / per-class precision, recall, F1,
and prints a readable confusion matrix — all via scikit-learn.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .config import DEFENSE_LEVELS

# Labels used in the task (0-8)
ALL_LABELS = sorted(DEFENSE_LEVELS.keys())
LABEL_NAMES = [f"{k}-{DEFENSE_LEVELS[k]}" for k in ALL_LABELS]


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Return a dict with:
      - accuracy
      - macro_precision, macro_recall, macro_f1  (over positive labels 1-8)
      - per_class: dict[label -> {precision, recall, f1, support}]
      - confusion_matrix: list[list[int]]
    """
    acc = accuracy_score(y_true, y_pred)

    # scikit-learn classification report as dict
    # Evaluate on positive labels (1-8) for macro averages, matching the paper
    positive_labels = list(range(1, 9))
    report = classification_report(
        y_true,
        y_pred,
        labels=ALL_LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Compute macro over positive (1-8) only, matching paper protocol
    pos_metrics = [report[LABEL_NAMES[k]] for k in positive_labels]
    macro_p = sum(m["precision"] for m in pos_metrics) / len(pos_metrics)
    macro_r = sum(m["recall"] for m in pos_metrics) / len(pos_metrics)
    macro_f1 = sum(m["f1-score"] for m in pos_metrics) / len(pos_metrics)

    per_class = {}
    for k in ALL_LABELS:
        name = LABEL_NAMES[k]
        per_class[k] = {
            "name": DEFENSE_LEVELS[k],
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS).tolist()

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def format_report(metrics: dict, title: str = "Evaluation Report") -> str:
    """Pretty-print evaluation metrics to a human-readable string."""
    lines = [
        f"\n{'=' * 70}",
        f"  {title}",
        f"{'=' * 70}",
        "",
        f"  Accuracy:         {metrics['accuracy']:.4f}",
        f"  Macro Precision:  {metrics['macro_precision']:.4f}  (over labels 1-8)",
        f"  Macro Recall:     {metrics['macro_recall']:.4f}  (over labels 1-8)",
        f"  Macro F1:         {metrics['macro_f1']:.4f}  (over labels 1-8)",
        "",
        "  Per-class breakdown:",
        f"  {'Level':<8} {'Name':<35} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}",
        f"  {'-'*8} {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*5}",
    ]
    for k in ALL_LABELS:
        c = metrics["per_class"][k]
        lines.append(
            f"  {k:<8} {c['name']:<35} {c['precision']:6.3f} {c['recall']:6.3f} "
            f"{c['f1']:6.3f} {c['support']:5d}"
        )

    # Confusion matrix
    lines += [
        "",
        "  Confusion matrix (rows=true, cols=predicted):",
        "  " + " " * 6 + "".join(f"{k:>6}" for k in ALL_LABELS),
    ]
    cm = metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        lines.append(
            f"  {ALL_LABELS[i]:>5} " + "".join(f"{v:>6}" for v in row)
        )
    lines.append(f"{'=' * 70}\n")
    return "\n".join(lines)


def format_distribution(labels: list[int], title: str = "Prediction Distribution") -> str:
    """Pretty-print label distribution."""
    total = len(labels)
    counts = Counter(labels)
    lines = [
        f"\n{'─' * 50}",
        f"  {title}  (n={total})",
        f"{'─' * 50}",
    ]
    for k in ALL_LABELS:
        n = counts.get(k, 0)
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        lines.append(f"  {k}: {n:>5} ({pct:5.1f}%)  {bar}")
    lines.append(f"{'─' * 50}\n")
    return "\n".join(lines)


def evaluate_predictions(
    gold_path: str | Path,
    pred_path: str | Path,
    title: str = "Evaluation",
) -> dict:
    """
    Load gold-standard and prediction files, compute and print metrics.

    Both files are JSON lists of ``{"id": ..., "label": int}``.
    Gold file may also be the original data format with ``"label"`` inside.
    """
    gold_data = json.loads(Path(gold_path).read_text())
    pred_data = json.loads(Path(pred_path).read_text())

    gold_map = {item["id"]: item["label"] for item in gold_data}
    pred_map = {item["id"]: item["label"] for item in pred_data}

    # Align by ID
    common_ids = sorted(set(gold_map) & set(pred_map))
    if not common_ids:
        raise ValueError("No overlapping IDs between gold and prediction files.")

    y_true = [gold_map[i] for i in common_ids]
    y_pred = [pred_map[i] for i in common_ids]

    metrics = compute_metrics(y_true, y_pred)
    print(format_report(metrics, title=title))
    return metrics
