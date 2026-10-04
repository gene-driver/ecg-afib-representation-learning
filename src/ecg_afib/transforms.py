"""Input transforms fitted exclusively on the selected training records."""
from dataclasses import asdict, dataclass

import numpy as np

from .features import augment_rr


@dataclass
class InputTransform:
    rr_statistics: bool = False
    standardize: bool = False
    mean: list | None = None
    scale: list | None = None

    def _features(self, values):
        values = np.asarray(values, dtype=np.float32)
        return augment_rr(values) if self.rr_statistics else values

    def fit(self, values):
        values = self._features(values)
        if values.ndim != 2:
            raise ValueError("Fit normalization on a matrix of RR/synthetic vectors")
        if self.standardize:
            self.mean = values.mean(axis=0).tolist()
            self.scale = np.maximum(values.std(axis=0), 1e-6).tolist()
        return self

    def apply(self, values):
        values = self._features(values)
        if self.standardize:
            if self.mean is None or self.scale is None:
                raise ValueError("Normalization has not been fitted")
            values = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        return np.asarray(values, dtype=np.float32)

    def to_dict(self):
        return asdict(self)
