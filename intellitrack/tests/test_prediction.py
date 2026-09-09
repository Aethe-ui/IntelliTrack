"""Tests for LSTM and Transformer trajectory predictors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch

from intellitrack.prediction.lstm_predictor import (
    LSTMPredictor,
    LSTMTrajectoryPredictor,
    TrajectoryDataset,
)
from intellitrack.prediction.train import train_predictor
from intellitrack.prediction.transformer_predictor import (
    TransformerPredictor,
    TransformerTrajectoryPredictor,
)


def _synthetic_linear_sequences(
    n_tracks: int = 4, length: int = 80
) -> List[List[Tuple[float, float]]]:
    """Generate simple linear trajectories for quick training tests."""
    seqs: List[List[Tuple[float, float]]] = []
    for t in range(n_tracks):
        vx, vy = 2.0 + 0.3 * t, 1.0 + 0.2 * t
        x0, y0 = 50.0 * t, 30.0 * t
        seq = [(x0 + vx * i, y0 + vy * i) for i in range(length)]
        seqs.append(seq)
    return seqs


def test_lstm_training_loss_decreases(tmp_path: Path) -> None:
    """Train LSTM briefly on synthetic data; assert loss drops."""
    seqs = _synthetic_linear_sequences()
    ds = TrajectoryDataset(sequences=seqs, sequence_length=10, horizon_frames=3)
    assert len(ds) > 10
    model = LSTMTrajectoryPredictor(hidden_size=32, num_layers=1)
    ckpt = tmp_path / "lstm.pt"
    losses = train_predictor(
        model=model,
        dataset=ds,
        epochs=5,
        lr=1e-2,
        checkpoint_path=str(ckpt),
        batch_size=16,
        val_fraction=0.2,
    )
    assert len(losses) == 5
    assert losses[-1] < losses[0]
    assert ckpt.is_file()


def test_lstm_predict_next_fallback_without_checkpoint(tmp_path: Path) -> None:
    """Missing checkpoint must not raise; returns two floats via CV fallback."""
    pred = LSTMPredictor(
        sequence_length=5,
        horizon_frames=3,
        hidden_size=16,
        num_layers=1,
        checkpoint_path=str(tmp_path / "missing_lstm.pt"),
    )
    history = [(float(i), float(2 * i)) for i in range(8)]
    out = pred.predict_next(history)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(v, float) and math.isfinite(v) for v in out)


def test_transformer_training_loss_decreases(tmp_path: Path) -> None:
    """Train Transformer briefly on synthetic data; assert loss drops."""
    seqs = _synthetic_linear_sequences()
    ds = TrajectoryDataset(sequences=seqs, sequence_length=10, horizon_frames=3)
    model = TransformerTrajectoryPredictor(d_model=32, nhead=4, num_layers=1)
    ckpt = tmp_path / "transformer.pt"
    losses = train_predictor(
        model=model,
        dataset=ds,
        epochs=5,
        lr=1e-2,
        checkpoint_path=str(ckpt),
        batch_size=16,
        val_fraction=0.2,
    )
    assert losses[-1] < losses[0]
    assert ckpt.is_file()


def test_transformer_predict_next_fallback_without_checkpoint(tmp_path: Path) -> None:
    """Missing Transformer checkpoint uses constant-velocity fallback safely."""
    pred = TransformerPredictor(
        sequence_length=5,
        horizon_frames=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        checkpoint_path=str(tmp_path / "missing_tf.pt"),
    )
    history = [(float(i), float(i * 0.5)) for i in range(6)]
    out = pred.predict_next(history)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(v, float) and math.isfinite(v) for v in out)


def test_trajectory_dataset_from_sequences() -> None:
    """Dataset yields tensors with expected shapes."""
    seqs = _synthetic_linear_sequences(n_tracks=1, length=40)
    ds = TrajectoryDataset(sequences=seqs, sequence_length=8, horizon_frames=2)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (8, 4)
    assert y.shape == (2,)
