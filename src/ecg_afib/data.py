"""Record-aware feature storage and readers for the source ECG data."""
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np
from scipy.io import loadmat


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_references(folder: str | Path) -> list[dict]:
    """Read the headerless PhysioNet 2017 REFERENCE.csv, retaining N/A only."""
    folder = Path(folder)
    result, seen = [], set()
    with (folder / "REFERENCE.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError("REFERENCE.csv must contain recording ID and rhythm label")
            record_id, rhythm = (item.strip() for item in row)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
                raise ValueError(f"Invalid recording ID: {record_id!r}")
            if record_id in seen:
                raise ValueError(f"Duplicate recording ID: {record_id}")
            seen.add(record_id)
            if rhythm in {"N", "A"}:
                result.append({"record_id": record_id, "label": int(rhythm == "A")})
    if not result:
        raise ValueError("No normal or AFib recordings found in REFERENCE.csv")
    return result


def read_signal(path: str | Path) -> np.ndarray:
    values = np.asarray(loadmat(path)["val"])
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"Expected one ECG lead in {Path(path).name}, got {values.shape}")
    signal = np.asarray(values[0], dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite samples")
    return signal


class FeatureStore:
    """One numeric .npy file per recording, shaped (pairs, 2, *view_shape)."""

    def __init__(self, folder: str | Path):
        self.root = Path(folder).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported feature manifest version")
        self.records = self.manifest["records"]
        self.view_shape = tuple(self.manifest["view_shape"])
        self.representation = self.manifest["representation"]
        if self.representation not in {"rr", "wavelet", "synthetic"}:
            raise ValueError("Unsupported feature representation")
        if not self.records or not self.view_shape or min(self.view_shape) <= 0:
            raise ValueError("Feature store must contain records with a positive view shape")
        self.by_id = {row["record_id"]: row for row in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("Feature manifest contains duplicate recording IDs")
        for row in self.records:
            if row["label"] not in (0, 1) or row["pairs"] < 1:
                raise ValueError("Invalid label or pair count in feature manifest")
            self.record_path(row["record_id"])
        self.cohort_hash = digest(sorted((r["record_id"], r["label"]) for r in self.records))

    def record_path(self, record_id: str) -> Path:
        path = (self.root / self.by_id[record_id]["file"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Feature paths must remain inside the feature directory")
        return path

    def pairs(self, record_id: str) -> np.ndarray:
        row = self.by_id[record_id]
        array = np.load(self.record_path(record_id), mmap_mode="r", allow_pickle=False)
        expected = (row["pairs"], 2, *self.view_shape)
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"Invalid stored features for {record_id}: {array.shape}, {array.dtype}")
        return array

    def select(self, ids) -> list[dict]:
        return [self.by_id[record_id] for record_id in ids]

    def flat_views(self, ids) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Materialize small RR features or embeddings; avoid this for CWT stores."""
        data, labels, names = [], [], []
        for record_id in ids:
            values = self.pairs(record_id).reshape(-1, *self.view_shape)
            data.extend(values)
            labels.extend([self.by_id[record_id]["label"]] * len(values))
            names.extend([record_id] * len(values))
        if not data:
            raise ValueError("Requested feature subset is empty")
        return np.asarray(data, dtype=np.float32), np.asarray(labels), names


def save_feature_store(folder, records, representation, parameters=None, provenance=None):
    """Write an iterable of (ID, label, paired features); also used by small demos."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty feature directory: {folder}")
    (folder / "records").mkdir(parents=True, exist_ok=True)
    rows, shape, seen = [], None, set()
    for record_id, label, values in records:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", record_id) or record_id in seen:
            raise ValueError(f"Invalid or duplicate recording ID: {record_id}")
        seen.add(record_id)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 3 or values.shape[0] < 1 or values.shape[1] != 2:
            raise ValueError("Features must have shape (pairs, 2, *view_shape)")
        if not np.isfinite(values).all() or label not in (0, 1):
            raise ValueError("Features must be finite and labels must be 0 or 1")
        current = values.shape[2:]
        if shape is not None and current != shape:
            raise ValueError("All recordings must share the same view shape")
        shape = current
        relative = f"records/{record_id}.npy"
        np.save(folder / relative, values, allow_pickle=False)
        rows.append({"record_id": record_id, "label": int(label), "pairs": len(values),
                     "file": relative, "sha256": file_digest(folder / relative)})
    if not rows:
        raise ValueError("No eligible recordings; no complete manifest was written")
    manifest = {"schema_version": 1, "representation": representation, "view_shape": list(shape),
                "parameters": parameters or {}, "provenance": provenance or {}, "records": rows}
    write_json(folder / "manifest.json", manifest)
    return FeatureStore(folder)
