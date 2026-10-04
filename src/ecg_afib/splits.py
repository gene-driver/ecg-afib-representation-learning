"""Persisted recording-level partitions, reused by every representation."""
from pathlib import Path
import json

from sklearn.model_selection import train_test_split

from .data import FeatureStore, digest, write_json


def create_splits(store: FeatureStore, output, seed=42, test_fraction=0.2, validation_fraction=0.2):
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 - test_fraction:
        raise ValueError("Positive train, validation, and test fractions are required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Split file already exists: {output}")
    ids = sorted(store.by_id)
    def labels(names):
        return [store.by_id[name]["label"] for name in names]
    try:
        train_val, test = train_test_split(ids, test_size=test_fraction, stratify=labels(ids), random_state=seed)
        train, validation = train_test_split(train_val, test_size=validation_fraction / (1 - test_fraction),
                                            stratify=labels(train_val), random_state=seed)
    except ValueError as exc:
        raise ValueError("Not enough records of each class for the requested stratified partitions") from exc
    result = {"schema_version": 1, "unit": "record", "seed": seed, "cohort_hash": store.cohort_hash,
              "train": sorted(train), "validation": sorted(validation), "test": sorted(test)}
    validate_splits(store, result)
    write_json(output, result)
    return result


def validate_splits(store: FeatureStore, splits):
    if splits.get("schema_version") != 1 or splits.get("unit") != "record":
        raise ValueError("Expected a version-1 recording-level split file")
    groups = [splits[name] for name in ("train", "validation", "test")]
    flat = [record_id for group in groups for record_id in group]
    if len(flat) != len(set(flat)):
        raise ValueError("A recording occurs more than once or crosses split boundaries")
    if set(flat) != set(store.by_id) or splits.get("cohort_hash") != store.cohort_hash:
        raise ValueError("Split cohort differs from features; regenerate features using --cohort")
    for name, group in zip(("train", "validation", "test"), groups):
        if {store.by_id[record_id]["label"] for record_id in group} != {0, 1}:
            raise ValueError(f"The {name} partition needs both classes")
    return splits


def load_splits(store, path):
    return validate_splits(store, json.loads(Path(path).read_text()))


def split_signature(splits):
    return digest({name: sorted(splits[name]) for name in ("train", "validation", "test")})
