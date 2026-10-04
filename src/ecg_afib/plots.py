"""Reusable figures for notebooks and saved experiment reports."""
from pathlib import Path
import csv
import json

from .evaluation import compare_runs


def plot_roc_curves(run_paths, partition="test"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    if partition not in {"validation", "test"}:
        raise ValueError("Partition must be validation or test")
    paths = list(map(Path, run_paths))
    compare_runs(paths)
    figure, axis = plt.subplots(figsize=(6, 5))
    for path in paths:
        name = json.loads((path / "run.json").read_text())["config"]["name"]
        with (path / f"{partition}_records.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        y, scores = [int(r["label"]) for r in rows], [float(r["score"]) for r in rows]
        fpr, tpr, _ = roc_curve(y, scores)
        axis.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, scores):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Record-level ROC — {partition}")
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure, axis


def plot_confusion_matrix(run_path, partition="test"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    metrics = json.loads((Path(run_path) / "metrics.json").read_text())[partition]
    import numpy as np
    figure, axis = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=["Normal", "AFib"]).plot(
        ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"Record-level predictions — {partition}")
    figure.tight_layout()
    return figure, axis


def plot_training_history(run_path):
    import matplotlib.pyplot as plt
    with (Path(run_path) / "training_history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    figure, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 3.5), squeeze=False)
    for axis, stage in zip(axes[0], stages):
        selected = [row for row in rows if row["stage"] == stage]
        axis.plot([int(r["epoch"]) for r in selected], [float(r["loss"]) for r in selected])
        axis.set(xlabel="Epoch", ylabel="Training loss", title=stage.capitalize())
    figure.tight_layout()
    return figure, axes
