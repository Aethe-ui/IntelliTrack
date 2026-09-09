"""LSTM trajectory predictor for IntelliTrack.

Predicts a target's future centroid from a short history of observed
positions.  Exposes the same ``predict_next(history)`` interface as
:class:`~intellitrack.prediction.no_prediction.NoPrediction` so the pipeline
can swap prediction stages without branching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Feature dim: (x, y, vx, vy)
_FEATURE_DIM = 4


def _centroids_to_features(
    centroids: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Convert a centroid sequence to normalised ``(x, y, vx, vy)`` features.

    Positions are left in pixel space (caller may scale). Velocities are
    frame-to-frame deltas.  The first velocity is zero.

    Args:
        centroids: Ordered ``(x, y)`` measurements, oldest first.

    Returns:
        Array of shape ``(len(centroids), 4)``.
    """
    arr = np.asarray(centroids, dtype=np.float64)
    feats = np.zeros((len(arr), _FEATURE_DIM), dtype=np.float64)
    feats[:, 0:2] = arr
    if len(arr) > 1:
        feats[1:, 2:4] = arr[1:] - arr[:-1]
    return feats


def _constant_velocity_extrapolate(
    history: List[Tuple[float, float]], horizon: int
) -> Tuple[float, float]:
    """Fallback: extrapolate assuming constant velocity over ``horizon`` steps."""
    if not history:
        return (0.0, 0.0)
    if len(history) == 1:
        return history[-1]
    vx = history[-1][0] - history[-2][0]
    vy = history[-1][1] - history[-2][1]
    return (history[-1][0] + vx * horizon, history[-1][1] + vy * horizon)


class TrajectoryDataset(Dataset):
    """Sliding-window trajectory dataset built from CSV or in-memory sequences.

    Each sample is an input window of ``sequence_length`` feature vectors and a
    target ``(x, y)`` at ``horizon_frames`` ahead of the window end.

    Args:
        sequences: List of per-track centroid sequences (each a list of
            ``(x, y)``).  Alternatively pass ``csv_path``.
        csv_path: Optional path to a trajectories CSV with columns
            ``track_id, centroid_x, centroid_y`` (and optional others).
        sequence_length: Number of past frames in each input window.
        horizon_frames: How many frames ahead to predict.
        frame_width: Used to normalise x into ``[0, 1]`` (default 640).
        frame_height: Used to normalise y into ``[0, 1]`` (default 480).
    """

    def __init__(
        self,
        sequences: Optional[List[List[Tuple[float, float]]]] = None,
        csv_path: Optional[str] = None,
        sequence_length: int = 15,
        horizon_frames: int = 5,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> None:
        self.sequence_length = sequence_length
        self.horizon_frames = horizon_frames
        self.frame_width = frame_width
        self.frame_height = frame_height

        if csv_path is not None:
            sequences = self._load_csv(csv_path)
        if not sequences:
            sequences = []

        self._samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for seq in sequences:
            self._samples.extend(self._windows_from_sequence(seq))

    @staticmethod
    def _load_csv(path: str) -> List[List[Tuple[float, float]]]:
        """Load per-track centroid sequences from a trajectories CSV."""
        import pandas as pd

        df = pd.read_csv(path)
        sequences: List[List[Tuple[float, float]]] = []
        for _, group in df.groupby("track_id"):
            group = group.sort_values("timestamp") if "timestamp" in group.columns else group
            seq = list(
                zip(
                    group["centroid_x"].astype(float).tolist(),
                    group["centroid_y"].astype(float).tolist(),
                )
            )
            sequences.append(seq)
        return sequences

    def _windows_from_sequence(
        self, seq: List[Tuple[float, float]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build (input, target) windows from one track sequence."""
        need = self.sequence_length + self.horizon_frames
        if len(seq) < need:
            return []

        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(seq) - need + 1):
            window = seq[i : i + self.sequence_length]
            target_pt = seq[i + self.sequence_length + self.horizon_frames - 1]
            feats = _centroids_to_features(window)
            # Normalise positions
            feats[:, 0] /= self.frame_width
            feats[:, 1] /= self.frame_height
            feats[:, 2] /= self.frame_width
            feats[:, 3] /= self.frame_height
            target = np.array(
                [target_pt[0] / self.frame_width, target_pt[1] / self.frame_height],
                dtype=np.float32,
            )
            samples.append((feats.astype(np.float32), target))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMTrajectoryPredictor(nn.Module):
    """LSTM + linear head predicting future ``(x, y)``.

    Args:
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        input_size: Feature dimension (default 4: x, y, vx, vy).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        input_size: int = _FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: Tensor of shape ``(batch, seq_len, features)``.

        Returns:
            Tensor of shape ``(batch, 2)`` — normalised ``(x, y)``.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class LSTMPredictor:
    """Runtime wrapper with ``predict_next`` interface and checkpoint fallback.

    Args:
        sequence_length: Rolling history length required for inference.
        horizon_frames: Prediction horizon used during training (and for
            constant-velocity fallback).
        hidden_size: LSTM hidden size (must match checkpoint).
        num_layers: LSTM layer count (must match checkpoint).
        checkpoint_path: Path to a ``.pt`` state dict.  If missing, falls back
            to constant-velocity extrapolation.
        frame_width: Pixel width used for (de)normalisation.
        frame_height: Pixel height used for (de)normalisation.
    """

    def __init__(
        self,
        sequence_length: int = 15,
        horizon_frames: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        checkpoint_path: str = "data/models/lstm_predictor.pt",
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> None:
        self.sequence_length = sequence_length
        self.horizon_frames = horizon_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._use_model = False

        self._model = LSTMTrajectoryPredictor(
            hidden_size=hidden_size, num_layers=num_layers
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
                logger.info("LSTMPredictor loaded checkpoint from %s", path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load LSTM checkpoint %s (%s); using constant-velocity fallback.",
                    path,
                    exc,
                )
        else:
            logger.warning(
                "LSTM checkpoint not found at %s — using constant-velocity fallback.",
                path,
            )

    def predict_next(
        self, history: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Predict the future centroid from a rolling history.

        Args:
            history: Recent ``(x, y)`` centroids, oldest first.

        Returns:
            Predicted ``(x, y)`` in pixel coordinates.
        """
        if not history:
            return (0.0, 0.0)

        if not self._use_model or len(history) < 2:
            return _constant_velocity_extrapolate(history, self.horizon_frames)

        # Pad / trim to sequence_length
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
