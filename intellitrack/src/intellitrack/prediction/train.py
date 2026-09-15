"""Shared training loop for trajectory predictors (LSTM and Transformer)."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

logger = logging.getLogger(__name__)


def _scene_split(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset, dict]:
    """Split *dataset* at scene/video level using its ``scene_ids`` property.

    Returns ``(train_subset, val_subset, info_dict)``.
    """
    scene_ids: List[str] = dataset.scene_ids  # type: ignore[attr-defined]
    unique_scenes = sorted(set(scene_ids))
    n_scenes = len(unique_scenes)

    n_val_scenes = max(1, int(n_scenes * val_fraction))
    n_train_scenes = n_scenes - n_val_scenes

    rng = random.Random(seed)
    shuffled = list(unique_scenes)
    rng.shuffle(shuffled)

    train_scenes = set(shuffled[:n_train_scenes])
    val_scenes = set(shuffled[n_train_scenes:])

    train_indices: List[int] = []
    val_indices: List[int] = []
    for idx, sid in enumerate(scene_ids):
        if sid in train_scenes:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    info = {
        "split_strategy": "scene/video",
        "seed": seed,
        "n_scenes": n_scenes,
        "n_train_scenes": len(train_scenes),
        "n_val_scenes": len(val_scenes),
        "n_train_windows": len(train_indices),
        "n_val_windows": len(val_indices),
    }

    return Subset(dataset, train_indices), Subset(dataset, val_indices), info


def _check_data_scale(dataset: Dataset) -> None:
    """Log configured resolution and warn if coordinates appear out of bounds.

    This check catches the case where the CSV was generated at a *higher*
    resolution than the configured frame dimensions (normalised coordinates
    exceed 1.0).  The *opposite* mismatch — a lower-resolution CSV with a
    higher-resolution config — **cannot** be reliably detected from coordinate
    ranges alone, because all values would still normalise to ≤ 1.0.
    """
    fw = getattr(dataset, "frame_width", None)
    fh = getattr(dataset, "frame_height", None)
    if fw is None or fh is None:
        return

    logger.info(
        "Data-scale: assuming frame_width=%.0f, frame_height=%.0f for normalisation.",
        fw,
        fh,
    )

    max_x = max_y = 0.0
    n_checked = 0
    for i in range(len(dataset)):
        _, target = dataset[i]
        # target is normalised (x/fw, y/fh); check if any exceed 1.0
        tx = float(target[0])
        ty = float(target[1])
        max_x = max(max_x, tx)
        max_y = max(max_y, ty)
        n_checked += 1
        # Sample at most 1000 windows for speed
        if n_checked >= 1000:
            break

    logger.info(
        "Data-scale: observed normalised target ranges (over %d samples): "
        "x=[0, %.3f] y=[0, %.3f].",
        n_checked,
        max_x,
        max_y,
    )

    if max_x > 1.05 or max_y > 1.05:
        logger.warning(
            "Data-scale mismatch: normalised target coordinates exceed 1.0 "
            "(max_x_norm=%.3f, max_y_norm=%.3f). The CSV may have been generated "
            "at a different resolution than frame_width=%.0f, frame_height=%.0f. "
            "Check that the CSV scale matches the config.",
            max_x,
            max_y,
            fw,
            fh,
        )


def train_predictor(
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    lr: float,
    checkpoint_path: str,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    device: Optional[str] = None,
    seed: int = 42,
) -> List[float]:
    """Train a trajectory predictor with MSE loss and save the best checkpoint.

    Args:
        model: An ``nn.Module`` that maps ``(batch, seq, feats) → (batch, 2)``.
        dataset: Dataset yielding ``(input_tensor, target_tensor)`` pairs.
            If it exposes a ``scene_ids`` property (e.g.
            :class:`~intellitrack.prediction.lstm_predictor.TrajectoryDataset`),
            the split is performed at the scene/video level to prevent
            validation leakage.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        checkpoint_path: Where to write the best ``state_dict``.
        batch_size: Mini-batch size.
        val_fraction: Fraction of data held out for validation.
        device: Torch device string; defaults to CUDA if available else CPU.
        seed: Random seed for reproducible scene-level splitting.

    Returns:
        List of mean training losses per epoch (length ``epochs``).
    """
    if len(dataset) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data-scale sanity check ---
    _check_data_scale(dataset)

    # --- Train / validation split ---
    has_scene_ids = hasattr(dataset, "scene_ids") and len(getattr(dataset, "scene_ids", [])) > 0

    if has_scene_ids and len(dataset) > 1:
        train_ds, val_ds, split_info = _scene_split(dataset, val_fraction, seed)
        for k, v in split_info.items():
            logger.info("Split: %s = %s", k, v)
        if len(val_ds) == 0:
            logger.warning("Validation set is empty after scene split; all data used for training.")
            val_ds = None
    else:
        # Fallback: random split (e.g. in-memory sequences without scene IDs)
        if has_scene_ids is False and len(dataset) > 1:
            logger.warning(
                "Dataset has no scene_ids; falling back to random window-level split. "
                "This may cause validation leakage for overlapping windows."
            )
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

    # Frame dimensions for pixel-space evaluation
    fw = getattr(dataset, "frame_width", None)
    fh = getattr(dataset, "frame_height", None)

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
        val_fde_px = float("nan")
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_batches = 0
            sum_disp_px = 0.0
            n_val_samples = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    v_running += float(criterion(pred, yb).item())
                    v_batches += 1
                    # Pixel-space displacement error
                    if fw is not None and fh is not None:
                        diff_x = (pred[:, 0] - yb[:, 0]) * fw
                        diff_y = (pred[:, 1] - yb[:, 1]) * fh
                        disp = torch.sqrt(diff_x ** 2 + diff_y ** 2)
                        sum_disp_px += float(disp.sum().item())
                        n_val_samples += yb.size(0)
            val_loss = v_running / max(v_batches, 1)
            if n_val_samples > 0:
                val_fde_px = sum_disp_px / n_val_samples

        if math.isnan(val_fde_px):
            logger.info(
                "Epoch %d/%d — train_loss=%.6f val_loss=%.6f",
                epoch,
                epochs,
                train_loss,
                val_loss,
            )
        else:
            logger.info(
                "Epoch %d/%d — train_loss=%.6f val_loss=%.6f val_fde_px=%.2f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_fde_px,
            )

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
            logger.info("Saved best checkpoint to %s (val_loss=%.6f)", ckpt, best_val)

    return epoch_losses
