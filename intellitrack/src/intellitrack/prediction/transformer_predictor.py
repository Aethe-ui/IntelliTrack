"""Transformer trajectory predictor for IntelliTrack.

Mirror of the LSTM predictor using a small ``nn.TransformerEncoder`` so the
four-way mode comparison (reactive / Kalman / LSTM / Transformer) is complete.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from intellitrack.prediction.lstm_predictor import (
    _FEATURE_DIM,
    _centroids_to_features,
    _constant_velocity_extrapolate,
)

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x`` of shape ``(batch, seq, d_model)``."""
        return x + self.pe[:, : x.size(1)]


class TransformerTrajectoryPredictor(nn.Module):
    """Transformer encoder + linear head predicting future ``(x, y)``.

    Args:
        d_model: Embedding / model dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoder layers.
        input_size: Feature dimension (default 4).
        dim_feedforward: FFN hidden size inside each encoder layer.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        input_size: int = _FEATURE_DIM,
        dim_feedforward: int = 128,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: Tensor of shape ``(batch, seq_len, features)``.

        Returns:
            Tensor of shape ``(batch, 2)`` — normalised ``(x, y)``.
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last)


class TransformerPredictor:
    """Runtime wrapper with ``predict_next`` interface and checkpoint fallback.

    Args:
        sequence_length: Rolling history length required for inference.
        horizon_frames: Prediction horizon (also used by CV fallback).
        d_model: Transformer model dimension.
        nhead: Attention head count.
        num_layers: Encoder layer count.
        checkpoint_path: Path to a ``.pt`` state dict.
        frame_width: Pixel width for (de)normalisation.
        frame_height: Pixel height for (de)normalisation.
    """

    def __init__(
        self,
        sequence_length: int = 15,
        horizon_frames: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        checkpoint_path: str = "data/models/transformer_predictor.pt",
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> None:
        self.sequence_length = sequence_length
        self.horizon_frames = horizon_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._use_model = False

        self._model = TransformerTrajectoryPredictor(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )
        self._model.eval()

        path = Path(checkpoint_path)
        if path.is_file():
            try:
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    state = torch.load(path, map_location="cpu")
                self._model.load_state_dict(state)
                self._use_model = True
                logger.info("TransformerPredictor loaded checkpoint from %s", path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load Transformer checkpoint %s (%s); using CV fallback.",
                    path,
                    exc,
                )
        else:
            logger.warning(
                "Transformer checkpoint not found at %s — using constant-velocity fallback.",
                path,
            )

    def predict_next(
        self, history: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Predict the future centroid from a rolling history."""
        if not history:
            return (0.0, 0.0)

        if not self._use_model or len(history) < 2:
            return _constant_velocity_extrapolate(history, self.horizon_frames)

        seq = list(history)
        if len(seq) < self.sequence_length:
            pad = [seq[0]] * (self.sequence_length - len(seq))
            seq = pad + seq
        else:
            seq = seq[-self.sequence_length :]

        feats = _centroids_to_features(seq)
        feats[:, 0] /= self.frame_width
        feats[:, 1] /= self.frame_height
        feats[:, 2] /= self.frame_width
        feats[:, 3] /= self.frame_height

        with torch.no_grad():
            x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
            pred = self._model(x).squeeze(0).numpy()

        return (
            float(pred[0] * self.frame_width),
            float(pred[1] * self.frame_height),
        )
