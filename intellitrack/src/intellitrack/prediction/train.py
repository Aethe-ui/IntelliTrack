"""Shared training loop for trajectory predictors (LSTM and Transformer)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


def train_predictor(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    lr: float,
    checkpoint_path: str,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    device: Optional[str] = None,
) -> List[float]:
    """Train a trajectory predictor with MSE loss and save the best checkpoint.

    Args:
        model: An ``nn.Module`` that maps ``(batch, seq, feats) → (batch, 2)``.
        dataset: Dataset yielding ``(input_tensor, target_tensor)`` pairs.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        checkpoint_path: Where to write the best ``state_dict``.
        batch_size: Mini-batch size.
        val_fraction: Fraction of data held out for validation.
        device: Torch device string; defaults to CUDA if available else CPU.

    Returns:
        List of mean training losses per epoch (length ``epochs``).
    """
    if len(dataset) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_val = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    if n_val > 0 and n_train > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ckpt = Path(checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    epoch_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        epoch_losses.append(train_loss)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    v_running += float(criterion(pred, yb).item())
                    v_batches += 1
            val_loss = v_running / max(v_batches, 1)

        logger.info(
            "Epoch %d/%d — train_loss=%.6f val_loss=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
            logger.info("Saved best checkpoint to %s (val_loss=%.6f)", ckpt, best_val)

    return epoch_losses
