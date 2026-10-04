"""Shared encoder, autoencoder, and classifier training without notebook state."""
from dataclasses import asdict
from pathlib import Path
import json
import logging
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import Model
from .data import write_json
from .models import BinaryClassifier, RRAutoencoder, build_encoder, nt_xent, pair_anomaly_scores
from .transforms import InputTransform

LOGGER = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use device='cpu'")
    return device


class RecordPairDataset(Dataset):
    """One pair per record per epoch prevents duplicate records in a minibatch."""

    def __init__(self, store, ids, transform, seed):
        self.store, self.ids, self.transform = store, list(ids), transform
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pairs = self.store.pairs(self.ids[index])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        pair = pairs[int(rng.integers(len(pairs)))]
        return torch.from_numpy(self.transform.apply(pair).copy())


def _loader(dataset, batch_size, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def fit_head(values, labels, config):
    seed_everything(config.seed)
    device = device_for(config.training.device)
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if set(labels) != {0, 1}:
        raise ValueError("Classifier training needs both rhythm classes")
    head = BinaryClassifier(values.shape[-1], config.model.classifier_hidden).to(device)
    positive_weight = float((labels == 0).sum() / (labels == 1).sum()) if config.training.balance_classifier else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    loader = _loader(TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
                     config.training.batch_size, config.seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.training.classifier_learning_rate)
    history = []
    for epoch in range(config.training.classifier_epochs):
        head.train()
        total, count = 0.0, 0
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(batch.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Classifier training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        history.append({"stage": "classifier", "epoch": epoch + 1, "loss": total / count})
        LOGGER.info("Classifier epoch %s: %.6f", epoch + 1, total / count)
    return head.eval(), history


def head_scores(head, values, device, batch_size=256):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            tensor = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(torch.sigmoid(head(tensor)).cpu().numpy())
    return np.concatenate(chunks)


def _encoded_records(encoder, store, ids, transform, device):
    encoder.eval()
    with torch.inference_mode():
        for record_id in ids:
            raw = store.pairs(record_id)
            # Each forward pass is one pair, limiting CWT inference memory.
            embeddings = []
            for pair in raw:
                values = torch.from_numpy(transform.apply(pair).copy()).to(device)
                embeddings.append(encoder(values).cpu().numpy())
            yield record_id, np.concatenate(embeddings)


def record_embeddings(predictor, store, ids):
    """Return mean latent vectors and labels by record for diagnostic plots."""
    if tuple(store.view_shape) != predictor.view_shape or store.manifest["parameters"] != predictor.parameters:
        raise ValueError("Embedding features differ from the trained preprocessing recipe")
    encoder = predictor.encoder.encoder if predictor.method == "autoencoder" else predictor.encoder
    names, labels, embeddings = [], [], []
    for record_id, values in _encoded_records(encoder, store, ids, predictor.transform, predictor.device):
        names.append(record_id)
        labels.append(store.by_id[record_id]["label"])
        embeddings.append(values.mean(axis=0))
    return names, np.asarray(labels), np.asarray(embeddings)


class NeuralPredictor:
    def __init__(self, encoder, head, transform, method, view_shape, model_settings, device, parameters):
        self.encoder, self.head, self.transform = encoder.eval(), head, transform
        self.method, self.view_shape, self.model_settings = method, tuple(view_shape), model_settings
        self.device, self.parameters = device, parameters

    def scores(self, store, ids):
        if tuple(store.view_shape) != self.view_shape or store.manifest["parameters"] != self.parameters:
            raise ValueError("Prediction features differ from the trained preprocessing recipe")
        rows = []
        with torch.inference_mode():
            for record_id in ids:
                scores = []
                for pair in store.pairs(record_id):
                    batch = torch.from_numpy(self.transform.apply(pair).copy()).to(self.device)
                    embedded = self.encoder(batch)
                    if self.method == "autoencoder":
                        current = ((embedded - batch) ** 2).mean(dim=1)
                    elif self.method == "one_class":
                        current = pair_anomaly_scores(embedded[:1], embedded[1:])
                    else:
                        current = torch.sigmoid(self.head(embedded))
                    scores.extend(current.cpu().numpy().tolist())
                rows.extend({"record_id": record_id, "label": store.by_id[record_id]["label"],
                             "observation": index, "score": float(score)} for index, score in enumerate(scores))
        return rows

    def save(self, folder):
        folder = Path(folder)
        torch.save({k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}, folder / "encoder.pt")
        if self.head is not None:
            torch.save({k: v.detach().cpu() for k, v in self.head.state_dict().items()}, folder / "classifier.pt")
        write_json(folder / "architecture.json", {"method": self.method, "view_shape": list(self.view_shape),
                   "model": asdict(self.model_settings), "transform": self.transform.to_dict(),
                   "parameters": self.parameters})

    @classmethod
    def load(cls, folder, device="cpu"):
        folder = Path(folder)
        metadata = json.loads((folder / "architecture.json").read_text())
        settings = Model(**metadata["model"])
        transform = InputTransform(**metadata["transform"])
        shape = tuple(metadata["view_shape"])
        effective = (shape[0] + 2,) if transform.rr_statistics else shape
        encoder = (RRAutoencoder(effective[0], settings.autoencoder_hidden, settings.embedding_dim)
                   if metadata["method"] == "autoencoder" else build_encoder(effective, settings))
        target = device_for(device)
        encoder.load_state_dict(torch.load(folder / "encoder.pt", map_location="cpu", weights_only=True))
        head = None
        if metadata["method"] == "contrastive":
            head = BinaryClassifier(settings.embedding_dim, settings.classifier_hidden)
            head.load_state_dict(torch.load(folder / "classifier.pt", map_location="cpu", weights_only=True))
            head = head.to(target).eval()
        return cls(encoder.to(target), head, transform, metadata["method"], shape, settings, target, metadata["parameters"])


def fit_neural(store, train_ids, config):
    seed_everything(config.seed)
    device = device_for(config.training.device)
    fit_ids = [record_id for record_id in train_ids
               if not (config.method == "one_class" or config.model.normal_only)
               or store.by_id[record_id]["label"] == 0]
    if len(fit_ids) < 2:
        raise ValueError("At least two encoder training records are required")
    transform = InputTransform(config.model.rr_statistics, config.model.standardize)
    if len(store.view_shape) == 2 and (transform.rr_statistics or transform.standardize):
        raise ValueError("RR statistics and vector standardization are not supported for CWT matrices")
    if store.representation != "rr" and transform.rr_statistics:
        raise ValueError("RR statistics require an RR feature store")
    if len(store.view_shape) == 1:
        transform.fit(store.flat_views(fit_ids)[0])
    shape = (store.view_shape[0] + 2,) if transform.rr_statistics else store.view_shape
    if config.method == "autoencoder":
        if len(shape) != 1:
            raise ValueError("The autoencoder experiment expects RR vectors")
        encoder = RRAutoencoder(shape[0], config.model.autoencoder_hidden, config.model.embedding_dim).to(device)
        values = transform.apply(store.flat_views(fit_ids)[0])
        dataset = TensorDataset(torch.from_numpy(values))
    else:
        encoder = build_encoder(shape, config.model).to(device)
        dataset = RecordPairDataset(store, fit_ids, transform, config.seed)
    loader = _loader(dataset, config.training.batch_size, config.seed)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.training.learning_rate)
    history = []
    for epoch in range(config.training.epochs):
        encoder.train()
        total, count = 0.0, 0
        if isinstance(dataset, RecordPairDataset):
            dataset.epoch = epoch
        for batch in loader:
            if config.method == "autoencoder":
                batch = batch[0].to(device)
                loss = nn.functional.mse_loss(encoder(batch), batch)
            else:
                if len(batch) < 2:
                    continue  # A lone pair has no contrastive negatives.
                batch = batch.to(device)
                loss = nt_xent(encoder(batch[:, 0]), encoder(batch[:, 1]), config.training.temperature)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        if count == 0:
            raise ValueError("No usable training batch")
        history.append({"stage": "encoder", "epoch": epoch + 1, "loss": total / count})
        LOGGER.info("Encoder epoch %s: %.6f", epoch + 1, total / count)
    encoder.eval()
    head = None
    if config.method == "contrastive":
        matrices, labels = [], []
        for record_id, matrix in _encoded_records(encoder, store, train_ids, transform, device):
            matrices.append(matrix)
            labels.extend([store.by_id[record_id]["label"]] * len(matrix))
        head, classifier_history = fit_head(np.concatenate(matrices), labels, config)
        history.extend(classifier_history)
    return NeuralPredictor(encoder, head, transform, config.method, store.view_shape,
                           config.model, device, store.manifest["parameters"]), history
