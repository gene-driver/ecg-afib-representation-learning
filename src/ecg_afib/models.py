"""Shared PyTorch architectures and paired-view objectives."""
import torch
from torch import nn
from torch.nn import functional as F


class RREncoder(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=(13, 8, 5), embedding_dim=2):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.extend([nn.Linear(previous, embedding_dim), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


class WaveletEncoder(nn.Module):
    """One CNN for any supported frequency count; infer the dense input size."""

    def __init__(self, frequencies=50, samples=4500, embedding_dim=128):
        super().__init__()
        if min(frequencies, samples) < 17:
            raise ValueError("Wavelet views need at least 17 frequency rows and time samples")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(), nn.MaxPool2d(3),
        )
        # Three stride-two convolutions followed by unpadded pooling of size 3.
        height, width = frequencies, samples
        for _ in range(3):
            height, width = (height + 1) // 2, (width + 1) // 2
        flattened = 64 * (height // 3) * (width // 3)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(flattened, 512), nn.ReLU(),
                                        nn.Linear(512, embedding_dim))

    def forward(self, values):
        if values.ndim == 3:
            values = values.unsqueeze(1)
        return self.projection(self.features(values))


class RRAutoencoder(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=(10, 5), embedding_dim=2):
        super().__init__()
        dimensions = [input_dim, *hidden_dims, embedding_dim]
        encoder = []
        for left, right in zip(dimensions, dimensions[1:]):
            encoder.extend([nn.Linear(left, right), nn.GELU()])
        decoder = []
        for index, (left, right) in enumerate(zip(dimensions[::-1], dimensions[-2::-1])):
            decoder.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                decoder.append(nn.GELU())
        self.encoder, self.decoder = nn.Sequential(*encoder), nn.Sequential(*decoder)

    def forward(self, values):
        return self.decoder(self.encoder(values))


class BinaryClassifier(nn.Module):
    """Return logits; sigmoid is applied only when producing AFib scores."""

    def __init__(self, input_dim, hidden_dims=(64,)):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def build_encoder(view_shape, settings):
    if len(view_shape) == 1:
        return RREncoder(view_shape[0], settings.hidden_dims, settings.embedding_dim)
    if len(view_shape) == 2:
        return WaveletEncoder(view_shape[0], view_shape[1], settings.embedding_dim)
    raise ValueError(f"Unsupported view shape: {view_shape}")


def nt_xent(first, second, temperature=0.1):
    """Symmetric NT-Xent with one positive counterpart for each view."""
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2 or temperature <= 0:
        raise ValueError("NT-Xent needs matching embedding matrices, two pairs, and positive temperature")
    embeddings = F.normalize(torch.cat([first, second], dim=0), dim=1)
    logits = embeddings @ embeddings.T / temperature
    mask = torch.eye(len(logits), device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = (torch.arange(len(logits), device=logits.device) + len(first)) % len(logits)
    return F.cross_entropy(logits, targets)


def pair_anomaly_scores(first, second):
    """Higher score means more anomalous: 1 minus paired cosine similarity."""
    return 1 - F.cosine_similarity(first, second, dim=-1).clamp(-1, 1)
