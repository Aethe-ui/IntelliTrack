"""Tests for LSTM and Transformer trajectory predictors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
import pytest

from intellitrack.prediction.lstm_predictor import (
    LSTMPredictor,
    LSTMTrajectoryPredictor,
    TrajectoryDataset,
)
from intellitrack.prediction.train import _pixel_error, _scene_split, _check_data_scale, train_predictor
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
    torch.manual_seed(42)
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
    runtime = LSTMPredictor(sequence_length=10, horizon_frames=3, hidden_size=32,
                            num_layers=1, checkpoint_path=str(ckpt))
    assert runtime._use_model
    assert all(math.isfinite(v) for v in runtime.predict_next(seqs[0][:10]))


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
    torch.manual_seed(42)
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
    runtime = TransformerPredictor(sequence_length=10, horizon_frames=3, d_model=32,
                                   nhead=4, num_layers=1, checkpoint_path=str(ckpt))
    assert runtime._use_model
    assert all(math.isfinite(v) for v in runtime.predict_next(seqs[0][:10]))


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


def test_scene_split_isolation(tmp_path: Path) -> None:
    """Windows from the same scene/video must never cross the train/val boundary."""
    from intellitrack.prediction.train import _scene_split

    # Build a dataset with 4 distinct scenes, each having multiple tracks
    # Simulate archive-format track_ids via CSV
    import csv

    csv_path = tmp_path / "multi_scene.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "centroid_x", "centroid_y", "timestamp"])
        for scene_idx in range(4):
            scene = f"scene{scene_idx}/video0"
            for track in range(3):
                tid = f"{scene}:{track}"
                for t in range(30):
                    writer.writerow([tid, 100.0 + t * 2.0, 50.0 + t * 1.0, t * 10])

    ds = TrajectoryDataset(
        csv_path=str(csv_path), sequence_length=8, horizon_frames=2
    )
    assert len(ds) == 4 * 3 * 21  # Numeric track IDs must not merge across videos.
    assert len(ds.scene_ids) == len(ds)

    train_sub, val_sub, info = _scene_split(ds, val_fraction=0.25, seed=42)

    # Collect scene IDs for each split
    train_scenes = {ds.scene_ids[i] for i in train_sub.indices}
    val_scenes = {ds.scene_ids[i] for i in val_sub.indices}

    # Core property: no scene overlap
    assert train_scenes & val_scenes == set(), (
        f"Scene leakage detected: {train_scenes & val_scenes}"
    )
    # Every window belongs to exactly one split
    assert len(train_sub) + len(val_sub) == len(ds)
    assert info["n_train_scenes"] + info["n_val_scenes"] == info["n_scenes"]
    again_train, again_val, again_info = _scene_split(ds, val_fraction=0.25, seed=42)
    assert train_sub.indices == again_train.indices
    assert val_sub.indices == again_val.indices
    assert info == again_info


def test_target_semantics() -> None:
    """Verify target is the single point at horizon_frames ahead of window end."""
    seq_len = 15
    horizon = 5
    fw, fh = 640.0, 480.0

    # Known linear trajectory: (i, 2*i)
    seq = [(float(i), float(2 * i)) for i in range(30)]
    ds = TrajectoryDataset(
        sequences=[seq],
        sequence_length=seq_len,
        horizon_frames=horizon,
        frame_width=fw,
        frame_height=fh,
    )

    # First window ends at 14; the fifth future observation is at 19.
    _, target = ds[0]
    expected_x = 19.0 / fw
    expected_y = 38.0 / fh
    assert abs(float(target[0]) - expected_x) < 1e-5
    assert abs(float(target[1]) - expected_y) < 1e-5


def test_pixel_error_calculation() -> None:
    """Verify pixel-space displacement error is computed correctly."""
    fw, fh = 640.0, 480.0

    # Normalised prediction and target
    pred = torch.tensor([[100.0 / fw, 200.0 / fh]])
    target = torch.tensor([[103.0 / fw, 204.0 / fh]])

    # Pixel-space error: sqrt((3)^2 + (4)^2) = 5.0
    disp = _pixel_error(pred, target, fw, fh)

    assert abs(float(disp[0]) - 5.0) < 1e-4


def test_scene_id_parsing_and_malformed_warning(caplog):
    assert TrajectoryDataset._extract_scene_id("bookstore/video0:0") == "bookstore/video0"
    assert TrajectoryDataset._extract_scene_id("001") == "001"
    assert TrajectoryDataset._extract_scene_id("1") == "1"
    assert TrajectoryDataset._extract_scene_id(":1") == ":1"
    assert "Malformed track ID" in caplog.text


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 1.2])
def test_invalid_split_fraction(fraction):
    ds = TrajectoryDataset(sequences=_synthetic_linear_sequences())
    with pytest.raises(ValueError, match="val_fraction"):
        _scene_split(ds, fraction, 42)


def test_training_rejects_unsafe_split(tmp_path):
    single_scene = TrajectoryDataset(sequences=_synthetic_linear_sequences(n_tracks=1))
    anonymous = torch.utils.data.TensorDataset(torch.zeros(3, 15, 4), torch.zeros(3, 2))
    for ds, message in [(single_scene, "at least two"), (anonymous, "scene_id")]:
        with pytest.raises(ValueError, match=message):
            train_predictor(LSTMTrajectoryPredictor(), ds, 1, 0.001, str(tmp_path / "unused.pt"))
    assert not (tmp_path / "unused.pt").exists()


def test_training_loaders_and_sample_weighted_metrics(tmp_path, monkeypatch, caplog):
    import intellitrack.prediction.train as training
    # Unequal final batch: known errors are 5, 10, 15 pixels, average 10.
    ds = TrajectoryDataset(sequences=[[(0., 0.), (3., 4.), (6., 8.), (9., 12.)]] * 2,
                           sequence_length=1, horizon_frames=1, frame_width=100, frame_height=100)
    loaders = []
    original_loader = training.DataLoader
    def capture_loader(subset, **kwargs):
        loaders.append(subset)
        return original_loader(subset, **kwargs)
    monkeypatch.setattr(training, "DataLoader", capture_loader)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    caplog.set_level("INFO")
    train_predictor(model, ds, 1, 0.0, str(tmp_path / "metric.pt"), batch_size=2, device="cpu")
    assert len(loaders) == 2
    assert {ds.scene_ids[i] for i in loaders[0].indices}.isdisjoint(
        {ds.scene_ids[i] for i in loaders[1].indices})
    epoch_record = next(r for r in caplog.records if r.msg.startswith("Epoch"))
    assert epoch_record.args[3] == pytest.approx((25 + 100 + 225) / (3 * 2 * 10000))
    assert epoch_record.args[4] == pytest.approx(10.0)


def test_csv_resolution_and_plain_ids(tmp_path, caplog):
    path = tmp_path / "metadata.csv"
    path.write_text("track_id,centroid_x,centroid_y,frame_width,frame_height\n"
                    "001,1,2,640,480\n001,2,3,640,480\n"
                    "1,4,5,640,480\n1,5,6,640,480\n")
    ds = TrajectoryDataset(csv_path=str(path), sequence_length=1, horizon_frames=1)
    assert set(ds.scene_ids) == {"001", "1"}
    assert "Scene isolation cannot be established" in caplog.text
    with pytest.raises(ValueError, match="frame_width metadata"):
        TrajectoryDataset(csv_path=str(path), frame_width=1280)


def test_scale_checks_all_observations_without_rescaling(caplog):
    ds = TrajectoryDataset(sequences=[[(10., 20.)] * 1050, [(-2., 800.)]])
    caplog.set_level("INFO")
    _check_data_scale(ds)
    assert ds.coordinate_ranges == (-2., 10., 20., 800.)
    assert "Coordinates exceed configured bounds" in caplog.text
    assert "x=[-2.000, 10.000] y=[20.000, 800.000]" in caplog.text
    torch.testing.assert_close(ds[0][1], torch.tensor([10 / 640, 20 / 480]))


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
@pytest.mark.parametrize("length", [4, 15, 25])
def test_runtime_features_and_pixel_output_match_training(tmp_path, kind, length):
    if kind == "lstm":
        model, wrapper = LSTMTrajectoryPredictor(), LSTMPredictor
    else:
        model, wrapper = TransformerTrajectoryPredictor(), TransformerPredictor
    path = tmp_path / f"{kind}.pt"
    torch.save(model.state_dict(), path)
    runtime = wrapper(checkpoint_path=str(path), frame_width=1280, frame_height=720)
    history = [(float(10 + i * 3), float(20 + i * 4)) for i in range(length)]
    observed = []
    handle = runtime._model.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
    actual = runtime.predict_next(history)
    handle.remove()
    window = ([history[0]] * max(0, 15 - length) + history)[-15:]
    ds = TrajectoryDataset(sequences=[window + [window[-1]] * 5], frame_width=1280, frame_height=720)
    features, _ = ds[0]
    torch.testing.assert_close(observed[0][0], features)
    assert features[0, 2:].tolist() == [0., 0.]
    assert features[-1].tolist() == pytest.approx([window[-1][0]/1280, window[-1][1]/720, 3/1280, 4/720])
    with torch.no_grad():
        expected = runtime._model(features.unsqueeze(0))[0] * torch.tensor([1280, 720])
    assert actual == pytest.approx(expected.tolist())
    assert runtime.predict_next([]) == (0., 0.)
    assert runtime.predict_next([(2., 3.)]) == (2., 3.)

