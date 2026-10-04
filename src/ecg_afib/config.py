"""Validated experiment settings, independent of notebooks and working directory."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib


@dataclass
class Paths:
    dataset: str = "data/processed/rr15"
    splits: str = "data/splits.json"
    output: str = "runs/experiment"


@dataclass
class Training:
    epochs: int = 6
    classifier_epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 0.001
    classifier_learning_rate: float = 0.001
    temperature: float = 0.1
    balance_classifier: bool = True
    device: str = "cpu"


@dataclass
class Model:
    embedding_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [13, 8, 5])
    autoencoder_hidden: list[int] = field(default_factory=lambda: [10, 5])
    classifier_hidden: list[int] = field(default_factory=lambda: [64])
    kpca_components: int = 21
    kpca_alpha: float = 0.1
    svm_nu: float = 0.05
    rr_statistics: bool = False
    standardize: bool = False
    normal_only: bool = False


@dataclass
class Experiment:
    name: str
    method: str
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    training: Training = field(default_factory=Training)
    model: Model = field(default_factory=Model)
    root: Path = field(default_factory=Path.cwd, repr=False)

    def validate(self):
        methods = {"kpca", "kpca_classifier", "ocsvm", "autoencoder", "contrastive", "one_class"}
        if self.method not in methods:
            raise ValueError(f"Unknown method {self.method!r}; choose from {sorted(methods)}")
        t, m = self.training, self.model
        if min(t.epochs, t.classifier_epochs) < 1 or t.batch_size < 2:
            raise ValueError("Epoch counts must be positive and batch_size must be at least 2")
        if min(t.learning_rate, t.classifier_learning_rate, t.temperature) <= 0:
            raise ValueError("Learning rates and temperature must be positive")
        if min([m.embedding_dim, m.kpca_components, *m.hidden_dims, *m.autoencoder_hidden, *m.classifier_hidden]) < 1:
            raise ValueError("Model dimensions must be positive")
        if not m.autoencoder_hidden or m.kpca_alpha <= 0 or not 0 < m.svm_nu <= 1:
            raise ValueError("Invalid autoencoder, KPCA, or SVM settings")
        return self

    def path(self, name: str) -> Path:
        path = Path(getattr(self.paths, name)).expanduser()
        return path if path.is_absolute() else self.root / path

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("root")
        return result


def load_config(path: str | Path) -> Experiment:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {"name", "method", "seed", "paths", "training", "model"}
    if unknown := set(raw) - allowed:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    return Experiment(
        name=raw["name"], method=raw["method"], seed=raw.get("seed", 42), root=root,
        paths=Paths(**raw.get("paths", {})), training=Training(**raw.get("training", {})),
        model=Model(**raw.get("model", {})),
    ).validate()
