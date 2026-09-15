"""Shared training loop for trajectory predictors (LSTM and Transformer)."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


def _scene_split(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset, dict]:
    """Split *dataset* at scene/video level using its ``scene_ids`` property.

    Returns ``(train_subset, val_subset, info_dict)``.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    scene_ids = getattr(dataset, "scene_ids", None)
    if scene_ids is None or len(scene_ids) != len(dataset):
        raise ValueError("Dataset must provide one scene_id per window; random window splitting is unsafe.")
    unique_scenes = sorted(set(scene_ids))
    n_scenes = len(unique_scenes)
    if n_scenes < 2:
        raise ValueError("Scene/video validation requires at least two groups with usable windows.")

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

    Out-of-bounds coordinates can indicate that the CSV was generated at a
    higher resolution than configured, but do not prove a mismatch.
    The opposite mismatch — a lower-resolution CSV with a
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

    ranges = getattr(dataset, "coordinate_ranges", None)
    if ranges is None:
        logger.warning("Raw coordinate ranges unavailable for this dataset.")
        return
    min_x, max_x, min_y, max_y = ranges
    logger.info(
        "Data-scale: observed CSV/sequence coordinate ranges: x=[%.3f, %.3f] y=[%.3f, %.3f] px.",
        min_x, max_x, min_y, max_y,
    )

    if min_x < 0 or min_y < 0 or max_x > fw or max_y > fh:
        logger.warning(
            "Coordinates exceed configured bounds (%.0f x %.0f); check CSV scale. No coordinates were clipped or rescaled.",
            fw, fh,
        )


def _pixel_error(pred: torch.Tensor, target: torch.Tensor, fw: float, fh: float) -> torch.Tensor:
    """Euclidean error in pixels for each single future-point prediction."""
    return torch.linalg.vector_norm((pred - target) * pred.new_tensor([fw, fh]), dim=1)


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
            Must expose a ``scene_ids`` property (e.g.
            :class:`~intellitrack.prediction.lstm_predictor.TrajectoryDataset`),
            so the split is performed at the scene/video level to prevent
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
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data-scale sanity check ---
    _check_data_scale(dataset)

    # --- Train / validation split ---
    train_ds, val_ds, split_info = _scene_split(dataset, val_fraction, seed)
    for k, v in split_info.items():
        logger.info("Split: %s = %s", k, v)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
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
        n_train_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * yb.size(0)
            n_train_samples += yb.size(0)
        train_loss = running / n_train_samples
        epoch_losses.append(train_loss)

        val_loss = train_loss
        val_fde_px = float("nan")
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_samples = 0
            sum_disp_px = 0.0
            n_val_samples = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    v_running += float(criterion(pred, yb).item()) * yb.size(0)
                    v_samples += yb.size(0)
                    # Pixel-space displacement error
                    if fw is not None and fh is not None:
                        disp = _pixel_error(pred, yb, fw, fh)
                        sum_disp_px += float(disp.sum().item())
                        n_val_samples += yb.size(0)
            val_loss = v_running / v_samples
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
