"""Train LSTM or Transformer trajectory predictors.

Usage::

    python scripts/train_predictor.py --model lstm --dataset path/to.csv
    python scripts/train_predictor.py --model transformer --dataset path/to.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intellitrack.prediction.lstm_predictor import LSTMTrajectoryPredictor, TrajectoryDataset
from intellitrack.prediction.train import train_predictor
from intellitrack.prediction.transformer_predictor import TransformerTrajectoryPredictor
from intellitrack.utils.config import get, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train IntelliTrack trajectory predictor")
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--dataset", required=True, help="Path to trajectories CSV")
    parser.add_argument("--config", default=str(root / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    seq_len = get(config, "prediction.sequence_length", 15)
    horizon = get(config, "prediction.horizon_frames", 5)
    fw = float(get(config, "camera.width", 640))
    fh = float(get(config, "camera.height", 480))

    dataset = TrajectoryDataset(
        csv_path=args.dataset,
        sequence_length=seq_len,
        horizon_frames=horizon,
        frame_width=fw,
        frame_height=fh,
    )
    logger.info("Dataset size: %d windows", len(dataset))
    if len(dataset) == 0:
        logger.error(
            "No training windows found — need sequences at least as long as "
            "sequence_length + horizon_frames (%d).",
            seq_len + horizon,
        )
        sys.exit(1)

    if args.model == "lstm":
        model = LSTMTrajectoryPredictor(
            hidden_size=get(config, "prediction.lstm.hidden_size", 64),
            num_layers=get(config, "prediction.lstm.num_layers", 2),
        )
        ckpt = get(config, "prediction.lstm.checkpoint_path", "data/models/lstm_predictor.pt")
    else:
        model = TransformerTrajectoryPredictor(
            d_model=get(config, "prediction.transformer.d_model", 64),
            nhead=get(config, "prediction.transformer.nhead", 4),
            num_layers=get(config, "prediction.transformer.num_layers", 2),
        )
        ckpt = get(
            config,
            "prediction.transformer.checkpoint_path",
            "data/models/transformer_predictor.pt",
        )

    # Resolve relative checkpoint paths against project root
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    losses = train_predictor(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=str(ckpt_path),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    logger.info(
        "Training complete. First loss=%.6f last loss=%.6f → %s",
        losses[0],
        losses[-1],
        ckpt_path,
    )


if __name__ == "__main__":
    main()
