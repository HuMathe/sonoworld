"""Batch evaluation against the public SonoScene360 recordings."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from sonoworld.evaluation.foa import (
    center_clip,
    evaluate_foa_pair,
    read_foa,
    save_energy_map,
)
from sonoworld.stages.inference.default_renderer import SonoScene360Dataset


SCALAR_METRICS = (
    "spatial_cc",
    "auc_judd",
    "azimuth_error_degrees",
    "elevation_error_degrees",
    "angular_error_degrees",
    "clap_audio_similarity",
    "clap_text_similarity",
    "clap_text_ratio",
    "qa_accuracy",
)


class SonoScene360Evaluator:
    def __init__(
        self,
        dataset: SonoScene360Dataset,
        predictions_root: str | Path,
        output_root: str | Path,
        sample_rate: int = 48_000,
        estimated_clip_seconds: float | None = 5.0,
        ground_truth_clip_seconds: float | None = 8.0,
        angular_resolution_degrees: float = 2.0,
        clap_metrics: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.predictions_root = Path(predictions_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self.estimated_clip_seconds = estimated_clip_seconds
        self.ground_truth_clip_seconds = ground_truth_clip_seconds
        self.angular_resolution_degrees = float(angular_resolution_degrees)
        self.clap_metrics = clap_metrics

    def evaluate(
        self,
        scene_ids: Sequence[str],
        mic_ids: Sequence[str] | None = None,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for scene_id in scene_ids:
            selected_mics = list(mic_ids) if mic_ids else self.dataset.mic_ids(scene_id)
            for mic_id in selected_mics:
                try:
                    rows.extend(self.evaluate_microphone(scene_id, mic_id))
                except Exception as exc:
                    error = {
                        "scene_id": scene_id,
                        "mic_id": mic_id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    errors.append(error)
                    if fail_fast:
                        raise

        mic_summaries = self._aggregate(rows, ("scene_id", "mic_id"))
        scene_summaries = self._aggregate(rows, ("scene_id",))
        overall = self._mean_metrics(rows)
        summary = {
            "schema_version": 1,
            "status": "done" if not errors else "partial",
            "dataset_root": str(self.dataset.root),
            "predictions_root": str(self.predictions_root),
            "num_pairs": len(rows),
            "num_microphones": sum(
                len(scene_mics) for scene_mics in mic_summaries.values()
            ),
            "metric_conventions": {
                "channel_order": ["W", "Y", "Z", "X"],
                "normalization": "SN3D",
                "estimated_center_clip_seconds": self.estimated_clip_seconds,
                "ground_truth_center_clip_seconds": self.ground_truth_clip_seconds,
                "angular_resolution_degrees": self.angular_resolution_degrees,
                "azimuth": "image-left=-90, front=0, image-right=+90",
            },
            "overall": overall,
            "scenes": scene_summaries,
            "microphones": mic_summaries,
            "errors": errors,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_root / "summary.json", summary)
        _write_json(self.output_root / "pairs.json", rows)
        self._write_csv(self.output_root / "pairs.csv", rows)
        return summary

    def evaluate_microphone(self, scene_id: str, mic_id: str) -> list[dict[str, Any]]:
        import numpy as np

        prediction_path = self.predictions_root / scene_id / mic_id / "foa.wav"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {prediction_path}")
        ground_truth_paths = self.dataset.recordings(scene_id, mic_id)
        if not ground_truth_paths:
            raise FileNotFoundError(f"No ground-truth recordings for {scene_id}/{mic_id}")

        estimated, sample_rate = read_foa(prediction_path, self.sample_rate)
        estimated = center_clip(estimated, sample_rate, self.estimated_clip_seconds)
        metadata = self.dataset.mic_metadata(scene_id, mic_id)
        annotations = metadata.get("sound_source_annotations", [])
        qa_accuracy = None
        if self.clap_metrics is not None:
            qa_accuracy = float(self.clap_metrics.qa_accuracy(estimated, annotations))

        rows = []
        for ground_truth_path in ground_truth_paths:
            ground_truth, _ = read_foa(ground_truth_path, self.sample_rate)
            ground_truth = center_clip(ground_truth, self.sample_rate, self.ground_truth_clip_seconds)
            metrics, estimated_map, ground_truth_map = evaluate_foa_pair(
                estimated,
                ground_truth,
                self.angular_resolution_degrees,
            )
            if self.clap_metrics is not None:
                metrics.update(self.clap_metrics.pair_metrics(estimated, ground_truth, annotations))
                metrics["qa_accuracy"] = qa_accuracy

            sample_id = ground_truth_path.parent.name
            pair_root = self.output_root / scene_id / mic_id / sample_id
            pair_root.mkdir(parents=True, exist_ok=True)
            np.save(pair_root / "estimated_energy.npy", estimated_map)
            np.save(pair_root / "ground_truth_energy.npy", ground_truth_map)
            save_energy_map(pair_root / "estimated_energy.jpg", estimated_map)
            save_energy_map(pair_root / "ground_truth_energy.jpg", ground_truth_map)
            compose_evaluation_audit(
                output_path=pair_root / "evaluation_audit.jpg",
                reference_path=self.dataset.reference_view_path(scene_id, mic_id),
                inference_root=self.predictions_root / scene_id / mic_id,
                estimated_energy_path=pair_root / "estimated_energy.jpg",
                ground_truth_energy_path=pair_root / "ground_truth_energy.jpg",
                annotations=annotations,
                estimated_direction=metrics["estimated_direction"],
                ground_truth_direction=metrics["ground_truth_direction"],
                title=f"{scene_id} / {mic_id} / {sample_id}",
            )
            row = {
                "scene_id": scene_id,
                "mic_id": mic_id,
                "sample_id": sample_id,
                "prediction_path": str(prediction_path),
                "ground_truth_path": str(ground_truth_path),
                "audit_path": str(pair_root / "evaluation_audit.jpg"),
                **metrics,
            }
            _write_json(pair_root / "metrics.json", row)
            rows.append(row)
        _write_json(
            self.output_root / scene_id / mic_id / "summary.json",
            {"num_pairs": len(rows), "metrics": self._mean_metrics(rows)},
        )
        return rows

    def _mean_metrics(self, rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
        import numpy as np

        output: dict[str, float | None] = {}
        for metric in SCALAR_METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            output[metric] = float(np.mean(values)) if values else None
        return output

    def _aggregate(self, rows: Sequence[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
        output: dict[str, Any] = {}
        for group, members in sorted(groups.items()):
            cursor = output
            for component in group[:-1]:
                cursor = cursor.setdefault(component, {})
            cursor[group[-1]] = {"num_pairs": len(members), "metrics": self._mean_metrics(members)}
        return output

    def _write_csv(self, path: Path, rows: Sequence[dict[str, Any]]) -> None:
        fields = ["scene_id", "mic_id", "sample_id", *SCALAR_METRICS, "prediction_path", "ground_truth_path", "audit_path"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def compose_evaluation_audit(
    output_path: str | Path,
    reference_path: str | Path,
    inference_root: Path,
    estimated_energy_path: Path,
    ground_truth_energy_path: Path,
    annotations: Sequence[dict[str, str]],
    estimated_direction: dict[str, float],
    ground_truth_direction: dict[str, float],
    title: str,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (1800, 1440), (18, 23, 30))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(27, bold=True)
    panel_font = _load_font(18, bold=True)
    footer_font = _load_font(15)
    draw.text((28, 13), f"SonoScene360 evaluation audit: {title}", font=title_font, fill="white")
    reference = _fit_image(Path(reference_path), (1740, 720))
    _draw_direction_overlay(reference, annotations, estimated_direction, ground_truth_direction)
    canvas.paste(reference, (30, 55))

    panels = (
        ("Estimated FOA energy", estimated_energy_path),
        ("Ground-truth FOA energy", ground_truth_energy_path),
        ("Generated source point rasterization", inference_root / "pointcloud_rasterization.jpg"),
        ("Generated visual scene at mic pose", inference_root / "generated_view_panorama.jpg"),
    )
    panel_size = (855, 285)
    for index, (label, path) in enumerate(panels):
        x = 30 + (index % 2) * 885
        y = 820 + (index // 2) * 325
        draw.text((x, y), label, font=panel_font, fill=(214, 223, 235))
        canvas.paste(_fit_image(path, panel_size), (x, y + 25))
    draw.text(
        (30, 1405),
        "Azimuth convention: image-left=-90, front=0, image-right=+90.  Cyan=annotations, magenta=estimated, yellow=ground truth.",
        font=footer_font,
        fill=(164, 181, 201),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=93)


def _fit_image(path: Path, size: tuple[int, int]) -> Any:
    from PIL import Image, ImageDraw, ImageOps

    if path.is_file():
        with Image.open(path) as source:
            return ImageOps.fit(source.convert("RGB"), size)
    panel = Image.new("RGB", size, (35, 42, 53))
    ImageDraw.Draw(panel).text(
        (18, 18), f"Missing: {path}", font=_load_font(16), fill=(255, 135, 135)
    )
    return panel


def _draw_direction_overlay(
    image: Any,
    annotations: Sequence[dict[str, str]],
    estimated: dict[str, float],
    ground_truth: dict[str, float],
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    annotation_font = _load_font(17, bold=True)
    direction_font = _load_font(19, bold=True)
    width, height = image.size
    origin = (width / 2.0, height * 0.9)
    convention = {
        "front": (0.0, 0.0),
        "left": (-90.0, 0.0),
        "right": (90.0, 0.0),
        "behind": (179.0, 0.0),
        "up": (0.0, 90.0),
        "down": (0.0, -90.0),
    }
    grouped: dict[str, list[str]] = {}
    for annotation in annotations:
        direction = str(annotation.get("direction", ""))
        label = str(annotation.get("text_label", "")).strip()
        if direction in convention and label:
            grouped.setdefault(direction, []).append(label)
    for direction, labels in grouped.items():
        target = _direction_pixel(*convention[direction], image.size)
        _arrow(draw, origin, target, (0, 222, 255), 4)
        draw.text(
            (target[0] + 5, target[1] + 5),
            f"{direction}: {' / '.join(dict.fromkeys(labels))}",
            font=annotation_font,
            fill=(0, 222, 255),
            stroke_width=2,
            stroke_fill="black",
        )
    for label, direction, color in (
        ("EST", estimated, (255, 82, 171)),
        ("GT", ground_truth, (255, 201, 58)),
    ):
        target = _direction_pixel(direction["azimuth_degrees"], direction["elevation_degrees"], image.size)
        _arrow(draw, origin, target, color, 6)
        draw.text(
            (target[0] + 6, target[1] - 24),
            f"{label} az={direction['azimuth_degrees']:+.1f} el={direction['elevation_degrees']:+.1f}",
            font=direction_font,
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )


def _direction_pixel(azimuth_degrees: float, elevation_degrees: float, size: tuple[int, int]) -> tuple[int, int]:
    x = (azimuth_degrees + 180.0) / 360.0 * (size[0] - 1)
    y = (90.0 - elevation_degrees) / 180.0 * (size[1] - 1)
    return int(round(x)), int(round(y))


def _arrow(draw: Any, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int) -> None:
    draw.line((start, end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 18.0
    base_x, base_y = end[0] - ux * head, end[1] - uy * head
    draw.polygon((end, (base_x + px * 8, base_y + py * 8), (base_x - px * 8, base_y - py * 8)), fill=color)


def _load_font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    name = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
