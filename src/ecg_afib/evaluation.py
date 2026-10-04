"""Shared AFib-oriented scores, record aggregation, and held-out metrics."""
from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)


def aggregate_records(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["record_id"]].append(row)
    result = []
    for record_id, values in sorted(groups.items()):
        labels = {int(v["label"]) for v in values}
        scores = np.asarray([v["score"] for v in values], dtype=float)
        if len(labels) != 1 or not labels <= {0, 1} or not np.isfinite(scores).all():
            raise ValueError("Inconsistent labels or non-finite scores within a recording")
        result.append({"record_id": record_id, "label": labels.pop(), "score": float(scores.mean()),
                       "observations": len(values)})
    if not result:
        raise ValueError("No predictions to aggregate")
    return result


def arrays(rows):
    y = np.asarray([r["label"] for r in rows], dtype=int)
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    if set(y) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Evaluation requires both classes and finite AFib-oriented scores")
    return y, scores


def select_threshold(validation_rows):
    """Maximize validation balanced accuracy; ties favor higher specificity."""
    y, scores = arrays(validation_rows)
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    balanced = (tpr + 1 - fpr) / 2
    candidates = np.flatnonzero(np.isfinite(thresholds))
    best = max(candidates, key=lambda i: (round(float(balanced[i]), 12), -float(fpr[i]), float(thresholds[i])))
    return float(thresholds[best])


def compute_metrics(rows, threshold):
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    y, scores = arrays(rows)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"unit": "record", "n_records": len(y), "normal_records": int((y == 0).sum()),
            "afib_records": int((y == 1).sum()), "threshold": float(threshold),
            "auroc": float(roc_auc_score(y, scores)), "average_precision": float(average_precision_score(y, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "afib_f1": float(f1_score(y, predicted, zero_division=0)),
            "normal_specificity": float(tn / (tn + fp)), "afib_sensitivity": float(tp / (tp + fn)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def write_predictions(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write empty predictions")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(paths):
    results, signatures = [], set()
    for path in map(Path, paths):
        metadata = json.loads((path / "run.json").read_text())
        if metadata.get("status") != "complete":
            raise ValueError(f"Run is incomplete: {path}")
        signatures.add((metadata["cohort_hash"], metadata["split_signature"]))
        metrics = json.loads((path / "metrics.json").read_text())
        results.append({"experiment": metadata["config"]["name"], **metrics["test"]})
    if len(signatures) != 1:
        raise ValueError("Only runs using the same cohort and recording partitions can be compared")
    return results
