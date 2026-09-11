"""Convert archived benchmark annotations to IntelliTrack trajectories CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def convert(
    annotations_root: Path,
    output_path: Path,
    source_width: float,
    source_height: float,
    output_width: float,
    output_height: float,
    frame_stride: int,
    max_points_per_track: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    tracks_written = 0

    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["track_id", "centroid_x", "centroid_y", "timestamp", "label"])

        for annotation_path in sorted(annotations_root.glob("**/annotations.txt")):
            video_id = annotation_path.parent.relative_to(annotations_root).as_posix()
            tracks: dict[str, list[tuple[int, float, float, str]]] = defaultdict(list)

            with annotation_path.open(errors="replace") as annotation_file:
                for line in annotation_file:
                    fields = line.split()
                    if len(fields) < 10:
                        continue
                    frame = int(fields[5])
                    if frame % frame_stride:
                        continue
                    track_id = fields[0]
                    xmin, ymin, xmax, ymax = (float(value) for value in fields[1:5])
                    center_x = ((xmin + xmax) / 2.0) * output_width / source_width
                    center_y = ((ymin + ymax) / 2.0) * output_height / source_height
                    label = fields[-1].strip('"')
                    points = tracks[track_id]
                    if len(points) < max_points_per_track:
                        points.append((frame, center_x, center_y, label))

            for track_id, points in tracks.items():
                points.sort(key=lambda point: point[0])
                if not points:
                    continue
                namespaced_id = f"{video_id}:{track_id}"
                for frame, center_x, center_y, label in points:
                    writer.writerow([namespaced_id, center_x, center_y, frame, label])
                    rows_written += 1
                tracks_written += 1

    return rows_written, tracks_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-width", type=float, default=2048.0)
    parser.add_argument("--source-height", type=float, default=2048.0)
    parser.add_argument("--output-width", type=float, default=640.0)
    parser.add_argument("--output-height", type=float, default=480.0)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-points-per-track", type=int, default=120)
    args = parser.parse_args()

    rows, tracks = convert(
        annotations_root=args.annotations,
        output_path=args.output,
        source_width=args.source_width,
        source_height=args.source_height,
        output_width=args.output_width,
        output_height=args.output_height,
        frame_stride=args.frame_stride,
        max_points_per_track=args.max_points_per_track,
    )
    print(f"Wrote {rows} trajectory points across {tracks} tracks to {args.output}")


if __name__ == "__main__":
    main()