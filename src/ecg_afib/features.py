"""One implementation of RR extraction, paired CWTs, and frequency statistics."""
from pathlib import Path
import importlib.util
import logging

import numpy as np

from .data import FeatureStore, file_digest, read_references, read_signal, save_feature_store, write_json

LOGGER = logging.getLogger(__name__)


def paired_windows(signal, samples=4500):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or samples < 1 or len(signal) < 2 * samples:
        raise ValueError(f"A recording needs at least {2 * samples} samples")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite samples")
    return signal[:2 * samples].reshape(2, samples)


def intervals_from_peaks(peaks):
    peaks = np.asarray(peaks)
    if peaks.ndim != 1 or len(peaks) < 2 or not np.isfinite(peaks).all():
        raise ValueError("At least two finite R peaks are required")
    intervals = np.diff(peaks).astype(np.float32)
    if np.any(intervals <= 0):
        raise ValueError("R-peak indices must be strictly increasing")
    return intervals


def detect_peaks(signal, sampling_rate=300):
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError('RR preparation requires: pip install -e ".[signal]"') from exc
    return nk.ecg_peaks(signal, sampling_rate=sampling_rate)[1]["ECG_R_Peaks"]


def rr_pairs(signal, length=15, mode="windows", sampling_rate=300, samples=4500):
    """Return paired RR vectors, retaining values in sample units.

    windows: detect peaks in two fixed ECG windows, then truncate/zero-pad.
    chunks: detect over the whole recording and pair consecutive RR chunks.
    """
    if length < 1:
        raise ValueError("RR length must be positive")
    if mode == "chunks":
        rr = intervals_from_peaks(detect_peaks(signal, sampling_rate))
        count = len(rr) // (2 * length)
        if count < 1:
            raise ValueError(f"At least {2 * length} RR intervals are needed")
        return rr[:count * 2 * length].reshape(count, 2, length)
    if mode != "windows":
        raise ValueError("RR mode must be windows or chunks")
    result = []
    for window in paired_windows(signal, samples):
        rr = intervals_from_peaks(detect_peaks(window, sampling_rate))[:length]
        result.append(np.pad(rr, (0, length - len(rr))))
    return np.asarray(result, dtype=np.float32)[None, ...]


def wavelet_pairs(signal, frequencies=50, sampling_rate=300, samples=4500, f_min=1.0, f_max=20.0):
    if frequencies < 1 or not 0 < f_min < f_max < sampling_rate / 2:
        raise ValueError("Invalid CWT frequency settings")
    try:
        import fcwt
    except ImportError as exc:
        raise ImportError('CWT preparation requires: pip install -e ".[wavelet]"') from exc
    result = []
    for window in paired_windows(signal, samples):
        _, coefficients = fcwt.cwt(np.ascontiguousarray(window), sampling_rate, f_min, f_max, frequencies)
        result.append(np.abs(coefficients).astype(np.float32))
    return np.stack(result)[None, ...]


def augment_rr(values):
    """Prepend standard deviation and mean, excluding zero padding from moments."""
    values = np.asarray(values, dtype=np.float32)
    mask = values > 0
    counts = mask.sum(axis=-1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("Cannot summarize an RR vector without positive intervals")
    means = (values * mask).sum(axis=-1, keepdims=True) / counts
    variance = (((values - means) ** 2) * mask).sum(axis=-1, keepdims=True) / counts
    return np.concatenate([np.sqrt(variance), means, values], axis=-1).astype(np.float32)


def frequency_statistics(arrays):
    """Streaming population moments over (..., frequency, time) magnitude arrays."""
    count, total, squares, minimum, maximum = 0, None, None, None, None
    for values in arrays:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 2 or not np.isfinite(values).all():
            raise ValueError("Expected finite arrays with frequency and time axes")
        flat = np.moveaxis(values, -2, 0).reshape(values.shape[-2], -1)
        if total is None:
            total = np.zeros(len(flat))
            squares = np.zeros(len(flat))
            minimum = np.full(len(flat), np.inf)
            maximum = np.full(len(flat), -np.inf)
        if len(flat) != len(total):
            raise ValueError("Frequency dimensions do not match")
        count += flat.shape[1]
        total += flat.sum(axis=1)
        squares += (flat ** 2).sum(axis=1)
        minimum = np.minimum(minimum, flat.min(axis=1))
        maximum = np.maximum(maximum, flat.max(axis=1))
    if not count:
        raise ValueError("Cannot summarize an empty collection")
    mean = total / count
    return {"count_per_frequency": count, "minimum": minimum.tolist(), "maximum": maximum.tolist(),
            "mean": mean.tolist(), "std": np.sqrt(np.maximum(0, squares / count - mean ** 2)).tolist()}


def prepare_dataset(source, output, representation="rr", rr_length=15, rr_mode="windows",
                    frequencies=50, sampling_rate=300, samples=4500, cohort=None):
    required = "neurokit2" if representation == "rr" else "fcwt"
    if representation not in {"rr", "wavelet"}:
        raise ValueError("Preparation supports rr or wavelet")
    if importlib.util.find_spec(required) is None:
        extra = "signal" if representation == "rr" else "wavelet"
        raise ImportError(f'Install the preparation dependency with: pip install -e ".[{extra}]"')
    source, output = Path(source), Path(output)
    references = read_references(source)
    expected = None if cohort is None else FeatureStore(cohort)
    if expected is not None:
        labels = {r["record_id"]: r["label"] for r in references}
        if any(labels.get(r["record_id"]) != r["label"] for r in expected.records):
            raise ValueError("Requested cohort IDs or labels differ from the source references")
        references = [r for r in references if r["record_id"] in expected.by_id]
    excluded, raw_hashes = [], {}

    def records():
        for index, row in enumerate(references, 1):
            name = row["record_id"]
            path = source / f"{name}.mat"
            # Missing source files fail immediately instead of silently changing the cohort.
            signal = read_signal(path)
            raw_hashes[name] = file_digest(path)
            try:
                if representation == "rr":
                    features = rr_pairs(signal, rr_length, rr_mode, sampling_rate, samples)
                else:
                    features = wavelet_pairs(signal, frequencies, sampling_rate, samples)
            except (ValueError, IndexError) as exc:
                if expected is not None:
                    raise ValueError(f"Cannot preserve requested cohort: {name}: {exc}") from exc
                excluded.append({"record_id": name, "reason": str(exc)})
                continue
            if index % 100 == 0:
                LOGGER.info("Prepared %s / %s records", index, len(references))
            yield name, row["label"], features

    parameters = {"sampling_rate": sampling_rate, "window_samples": samples,
                  "rr_length": rr_length if representation == "rr" else None,
                  "rr_mode": rr_mode if representation == "rr" else None,
                  "frequencies": frequencies if representation == "wavelet" else None,
                  "frequency_range_hz": [1.0, 20.0] if representation == "wavelet" else None}
    provenance = {"dataset": "PhysioNet/CinC 2017", "reference_sha256": file_digest(source / "REFERENCE.csv"),
                  "raw_file_sha256": raw_hashes}
    store = save_feature_store(output, records(), representation, parameters, provenance)
    write_json(output / "exclusions.json", excluded)
    return store
