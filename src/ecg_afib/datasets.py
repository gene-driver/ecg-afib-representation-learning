"""Small dataset-inspection helpers and a clearly separated synthetic example."""
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Experiment, Model, Paths, Training
from .data import read_references, save_feature_store
from .splits import create_splits


def summarize_physionet(folder):
    rows = read_references(folder)
    counts = Counter(row["label"] for row in rows)
    return {"normal": counts[0], "afib": counts[1], "total_binary_records": len(rows)}


def inspect_wfdb(record_name, pn_dir=None):
    """Read metadata only, locally or from an explicitly specified WFDB dataset."""
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError('WFDB inspection requires: pip install -e ".[exploration]"') from exc
    record = wfdb.rdheader(str(record_name), pn_dir=pn_dir)
    return {"record": record.record_name, "sampling_rate": record.fs, "samples": record.sig_len,
            "seconds": record.sig_len / record.fs, "signals": getattr(record, "sig_name", None),
            "segments": getattr(record, "seg_name", None)}


def inspect_annotations(record_name, pn_dir=None, extension="atr"):
    """Summarize rhythm notes without guessing a record's sampling frequency."""
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError('WFDB inspection requires: pip install -e ".[exploration]"') from exc
    annotation = wfdb.rdann(str(record_name), extension=extension, pn_dir=pn_dir)
    return dict(Counter(note for note in annotation.aux_note if note))


def make_synthetic_pairs(output, records_per_class=40, seed=42):
    """Gaussian 3D clouds and multiplicative views; these are not ECG records."""
    rng = np.random.default_rng(seed)
    def records():
        for label, center in ((0, 1.0), (1, -1.0)):
            for index in range(records_per_class):
                value = rng.normal(center, 0.5, 3).astype(np.float32)
                yield f"synthetic_{label}_{index:04d}", label, np.stack([value, value * 1.1])[None]
    return save_feature_store(output, records(), "synthetic", {"dimensions": 3, "view_scale": 1.1},
                              {"synthetic": True, "seed": seed, "description": "Gaussian mechanism demonstration; not ECG"})


def run_synthetic_demo(output, epochs=10, seed=42):
    from .experiments import run_experiment
    try:
        import torch  # noqa: F401 -- check before creating the demo files
    except ImportError as exc:
        raise ImportError('The synthetic neural demo requires: pip install -e ".[neural]"') from exc
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Demo directory is not empty: {root}")
    store = make_synthetic_pairs(root / "features", seed=seed)
    create_splits(store, root / "splits.json", seed=seed)
    config = Experiment(name="synthetic_contrastive_demo", method="contrastive", seed=seed, root=root,
                        paths=Paths("features", "splits.json", "run"),
                        training=Training(epochs=epochs, classifier_epochs=epochs),
                        model=Model(embedding_dim=2, hidden_dims=[2], classifier_hidden=[]))
    return run_experiment(config)
