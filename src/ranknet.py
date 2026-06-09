"""
ranknet.py  —  Pairwise Learning-to-Rank Neural Network
========================================================
Input: 11 behavioral features (from feature_extractor.py)
Architecture: 11 → 64 → 32 → 1 (Sigmoid)
Loss: Pairwise Binary Cross-Entropy (RankNet)
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUM_FEATURES = 11
MODEL_PATH   = "model.pt"


class RankNet(nn.Module):
    def __init__(self, in_features: int = NUM_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


def pairwise_loss(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    label:   torch.Tensor,
) -> torch.Tensor:
    """
    RankNet pairwise BCE loss.
    label = 1  if i > j (i is better),
            0  if j > i,
            0.5 if tied.
    """
    diff = score_i - score_j
    return nn.functional.binary_cross_entropy(
        torch.sigmoid(diff), label, reduction="mean"
    )


def load_model(path: str, device: str = "cpu") -> RankNet:
    model = RankNet(NUM_FEATURES)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_scores(model: RankNet, normed_matrix: "np.ndarray") -> "np.ndarray":
    import numpy as np
    t = torch.tensor(normed_matrix, dtype=torch.float32)
    with torch.no_grad():
        return model.score(t).numpy()
