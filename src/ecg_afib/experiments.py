"""One orchestration path for all experiment presets."""
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import logging
import platform
import time

import numpy as np

from . import __version__
from .baselines import fit_classical
from .config import Experiment, load_config
from .data import FeatureStore, file_digest, write_json
from .evaluation import aggregate_records, compute_metrics, select_threshold, write_predictions
from .splits import load_splits, split_signature

LOGGER = logging.getLogger(__name__)


def environment():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "neurokit2", "fcwt", "joblib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "project_version": __version__, "packages": packages}


def run_experiment(config: Experiment | str | Path):
    """Fit on train, select a record-level threshold on validation, then score test."""
    config = (config if isinstance(config, Experiment) else load_config(config)).validate()
    store = FeatureStore(config.path("dataset"))
    splits = load_splits(store, config.path("splits"))
    neural = config.method in {"contrastive", "one_class", "autoencoder", "kpca_classifier"}
    if neural:
        try:
            from .training import fit_neural
        except ImportError as exc:
            raise ImportError('Neural experiments require: pip install -e ".[neural]"') from exc
    output = config.path("output")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}; choose another output")
    output.mkdir(parents=True, exist_ok=True)
    run = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "config": config.to_dict(), "cohort_hash": store.cohort_hash,
           "split_signature": split_signature(splits), "feature_manifest_sha256": file_digest(store.manifest_path),
           "representation": store.representation, "feature_parameters": store.manifest["parameters"],
           "data_provenance": store.manifest["provenance"], "score_direction": "higher means AFib",
           "threshold_selection": "maximum validation record-level balanced accuracy; specificity tie-break",
           "observation_unit": "pair" if config.method == "one_class" else "window"}
    write_json(output / "run.json", run)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "splits.json", splits)
    write_json(output / "environment.json", environment())
    started = time.monotonic()
    try:
        np.random.seed(config.seed)
        fitter = fit_classical if config.method in {"kpca", "kpca_classifier", "ocsvm"} else fit_neural
        predictor, history = fitter(store, splits["train"], config)
        validation_observations = predictor.scores(store, splits["validation"])
        validation = aggregate_records(validation_observations)
        threshold = select_threshold(validation)
        # Test data are first scored after training and threshold selection have finished.
        test_observations = predictor.scores(store, splits["test"])
        test = aggregate_records(test_observations)
        result = {"validation": compute_metrics(validation, threshold), "test": compute_metrics(test, threshold)}
        for name, rows in (("validation_observations", validation_observations), ("test_observations", test_observations),
                           ("validation_records", validation), ("test_records", test)):
            write_predictions(output / f"{name}.csv", rows)
        write_json(output / "metrics.json", result)
        if history:
            write_predictions(output / "training_history.csv", history)
        predictor.save(output)
        run.update(status="complete", elapsed_seconds=time.monotonic() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "run.json", run)
        LOGGER.info("Saved completed experiment to %s", output)
        return result
    except Exception as exc:
        run.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic() - started)
        write_json(output / "run.json", run)
        raise
