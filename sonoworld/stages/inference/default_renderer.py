"""SonoScene360 inference renderer with persistent visual audit artifacts."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from sonoworld.stages.inference.base import (
    TARGET_SAMPLE_RATE,
    MicrophonePose,
    RenderSource,
    load_render_sources,
    reconstruct_microphone_pose,
    render_foa,
    scene_mics,
    write_foa,
)


SOURCE_COLORS = (
    (0, 222, 255),
    (255, 82, 171),
    (255, 201, 58),
    (87, 230, 139),
    (157, 123, 255),
    (255, 126, 77),
)


@dataclass(frozen=True)
class SonoScene360Dataset:
    root: Path
    metadata: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path) -> "SonoScene360Dataset":
        root = Path(root).expanduser().resolve()
        metadata_path = root / "data" / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"SonoScene360 metadata is missing: {metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a JSON object in {metadata_path}")
        return cls(root=root, metadata=metadata)

    @property
    def scene_ids(self) -> list[str]:
        mapping = self.metadata.get("scene_input_images_paths", {})
        return sorted(mapping) if isinstance(mapping, dict) else []

    def mic_ids(self, scene_id: str) -> list[str]:
        return scene_mics(self.metadata, scene_id)

    def resolve_dataset_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def mic_metadata_path(self, scene_id: str, mic_id: str) -> Path:
        value = self.metadata.get("scene_mic_metadata_paths", {}).get(scene_id, {}).get(mic_id)
        if not isinstance(value, str):
            raise KeyError(f"Unknown SonoScene360 microphone {scene_id}/{mic_id}")
        return self.resolve_dataset_path(value)

    def mic_metadata(self, scene_id: str, mic_id: str) -> dict[str, Any]:
        path = self.mic_metadata_path(scene_id, mic_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return value

    def input_panorama_path(self, scene_id: str) -> Path:
        value = self.metadata.get("scene_input_images_paths", {}).get(scene_id)
        if not isinstance(value, str):
            raise KeyError(f"Unknown SonoScene360 scene: {scene_id}")
        return self.resolve_dataset_path(value)

    def calibration_panorama_path(self, scene_id: str) -> Path:
        return self.root / "data" / scene_id / "images" / "panorama.jpg"

    def reference_view_path(self, scene_id: str, mic_id: str) -> Path:
        return self.root / "data" / scene_id / "images" / mic_id / "rendered.jpg"

    def recordings(self, scene_id: str, mic_id: str) -> list[Path]:
        values = (
            self.metadata.get("scene_mic_recording_paths", {})
            .get(scene_id, {})
            .get(mic_id, [])
        )
        return [self.resolve_dataset_path(value) for value in values] if isinstance(values, list) else []


class SonoScene360InferenceRenderer:
    """Render a generated scene from all selected SonoScene360 microphone poses."""

    def __init__(
        self,
        dataset: SonoScene360Dataset,
        sample_rate: int = TARGET_SAMPLE_RATE,
        duration_seconds: float = 8.0,
        device: str | None = "cuda",
        render_generated_view: bool = True,
        require_generated_view: bool = False,
        visualization_width: int = 2048,
        include_background: bool = True,
    ) -> None:
        self.dataset = dataset
        self.sample_rate = int(sample_rate)
        self.duration_seconds = float(duration_seconds)
        self.device = device
        self.render_generated_view = bool(render_generated_view)
        self.require_generated_view = bool(require_generated_view)
        self.visualization_width = int(visualization_width)
        self.include_background = bool(include_background)

    def render_scene(
        self,
        scene_id: str,
        scene_root: str | Path,
        output_root: str | Path,
        mic_ids: Sequence[str] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        import numpy as np
        from PIL import Image

        scene_root = Path(scene_root).expanduser().resolve()
        output_root = Path(output_root).expanduser().resolve()
        depth_path = scene_root / "visual_scene" / "panorama_depth.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Generated scene depth is missing: {depth_path}")
        depth = np.load(depth_path).astype(np.float32)
        sources = load_render_sources(scene_root, include_background=self.include_background)
        if not sources:
            raise RuntimeError(f"No selected sources in {scene_root / 'spatial' / 'summary.json'}")

        calibration_path = self.dataset.calibration_panorama_path(scene_id)
        with Image.open(calibration_path) as image:
            panorama_size = image.size

        selected_mics = list(mic_ids) if mic_ids else self.dataset.mic_ids(scene_id)
        if not selected_mics:
            raise RuntimeError(f"No microphones found for SonoScene360 scene {scene_id}")
        return [
            self.render_microphone(
                scene_id=scene_id,
                mic_id=mic_id,
                scene_root=scene_root,
                output_root=output_root,
                depth=depth,
                panorama_size=panorama_size,
                sources=sources,
                force=force,
            )
            for mic_id in selected_mics
        ]

    def render_microphone(
        self,
        scene_id: str,
        mic_id: str,
        scene_root: Path,
        output_root: Path,
        depth: Any,
        panorama_size: tuple[int, int],
        sources: Sequence[RenderSource],
        force: bool = False,
    ) -> dict[str, Any]:
        mic_root = output_root / scene_id / mic_id
        manifest_path = mic_root / "manifest.json"
        if manifest_path.is_file() and not force:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        mic_root.mkdir(parents=True, exist_ok=True)

        metadata = self.dataset.mic_metadata(scene_id, mic_id)
        pose = reconstruct_microphone_pose(metadata, panorama_size, depth)
        audio, source_records = render_foa(
            sources,
            pose.world_to_microphone,
            sample_rate=self.sample_rate,
            duration_seconds=self.duration_seconds,
        )
        foa_path = mic_root / "foa.wav"
        write_foa(foa_path, audio, self.sample_rate)

        pose_path = mic_root / "pose.json"
        pose_payload = {
            "scene_id": scene_id,
            "mic_id": mic_id,
            "coordinate_system": "panorama_world_to_microphone_opencv",
            **pose.to_dict(),
        }
        _write_json(pose_path, pose_payload)
        sources_path = mic_root / "sources.json"
        _write_json(sources_path, {"channel_order": ["W", "Y", "Z", "X"], "sources": source_records})

        visualizations: dict[str, str] = {}
        copied = (
            ("dataset_input_panorama", self.dataset.input_panorama_path(scene_id), mic_root / "dataset_input_panorama.jpg"),
            ("dataset_reference_view", self.dataset.reference_view_path(scene_id, mic_id), mic_root / "dataset_reference_view.jpg"),
            ("dataset_calibration_panorama", self.dataset.calibration_panorama_path(scene_id), mic_root / "dataset_calibration_panorama.jpg"),
        )
        for label, source, destination in copied:
            if source.is_file():
                shutil.copy2(source, destination)
                visualizations[label] = destination.name

        annotation_path = mic_root / "mic_pose_annotation.jpg"
        draw_mic_pose_annotation(
            self.dataset.calibration_panorama_path(scene_id), metadata, pose, annotation_path,
            output_width=self.visualization_width,
        )
        visualizations["mic_pose_annotation"] = annotation_path.name

        point_path = mic_root / "pointcloud_rasterization.jpg"
        point_projection_path = mic_root / "pointcloud_projection.npz"
        rasterize_source_points(
            sources,
            pose,
            point_path,
            projection_out_path=point_projection_path,
            background_path=self.dataset.reference_view_path(scene_id, mic_id),
            width=self.visualization_width,
        )
        visualizations["pointcloud_rasterization"] = point_path.name
        visualizations["pointcloud_projection"] = point_projection_path.name

        generated_view_error = None
        gaussian_path = scene_root / "visual_scene" / "representation" / "marble" / "gaussian.ply"
        if self.render_generated_view:
            if gaussian_path.is_file():
                try:
                    from sonoworld.utils.gaussian_splat_utils import render_marble_pinhole_view

                    render_metadata = render_marble_pinhole_view(
                        gaussian_path,
                        pose.world_to_microphone,
                        mic_root / "generated_view.jpg",
                        depth_out_path=mic_root / "generated_view_depth.npy",
                        depth_vis_out_path=mic_root / "generated_view_depth.jpg",
                        device=self.device,
                        panorama_rgb_out_path=mic_root / "generated_view_panorama.jpg",
                        panorama_depth_out_path=mic_root / "generated_view_panorama_depth.npy",
                        panorama_depth_vis_out_path=mic_root / "generated_view_panorama_depth.jpg",
                    )
                    _write_json(mic_root / "generated_view.json", render_metadata)
                    visualizations.update(
                        {
                            "generated_view": "generated_view.jpg",
                            "generated_view_depth": "generated_view_depth.jpg",
                            "generated_view_depth_array": "generated_view_depth.npy",
                            "generated_view_panorama": "generated_view_panorama.jpg",
                            "generated_view_panorama_depth": "generated_view_panorama_depth.jpg",
                            "generated_view_panorama_depth_array": "generated_view_panorama_depth.npy",
                        }
                    )
                except Exception as exc:  # Keep audio inference usable without gsplat/CUDA.
                    generated_view_error = f"{type(exc).__name__}: {exc}"
                    if self.require_generated_view:
                        raise
            else:
                generated_view_error = f"Missing Marble Gaussian: {gaussian_path}"
                if self.require_generated_view:
                    raise FileNotFoundError(generated_view_error)

        overview_path = mic_root / "inference_audit.jpg"
        compose_inference_audit(
            overview_path,
            scene_id,
            mic_id,
            [
                mic_root / "dataset_reference_view.jpg",
                annotation_path,
                point_path,
                mic_root / "generated_view_panorama.jpg",
            ],
            source_records,
            generated_view_error,
        )
        visualizations["audit_sheet"] = overview_path.name

        manifest = {
            "schema_version": 1,
            "status": "done",
            "scene_id": scene_id,
            "mic_id": mic_id,
            "scene_root": str(scene_root),
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "channel_order": ["W", "Y", "Z", "X"],
            "audio": foa_path.name,
            "pose": pose_path.name,
            "sources": sources_path.name,
            "num_sources": len(sources),
            "include_background": self.include_background,
            "visualizations": visualizations,
            "generated_view_error": generated_view_error,
            "ground_truth_recordings": [str(path) for path in self.dataset.recordings(scene_id, mic_id)],
        }
        _write_json(manifest_path, manifest)
        return manifest


def draw_mic_pose_annotation(
    panorama_path: str | Path,
    metadata: dict[str, Any],
    pose: MicrophonePose,
    output_path: str | Path,
    output_width: int = 2048,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(panorama_path) as source:
        image = source.convert("RGB")
    scale = output_width / float(image.width)
    image = image.resize((output_width, max(1, int(round(image.height * scale)))))
    draw = ImageDraw.Draw(image)
    label_font = _load_font(max(16, output_width // 90), bold=True)
    info_font = _load_font(max(16, output_width // 100))
    points = metadata["2d_points"]
    center = (
        float(points["mic_center_2d"]["x"]) * scale,
        float(points["mic_center_2d"]["y"]) * scale,
    )
    ground = (
        float(points["mic_location_2d"]["x"]) * scale,
        float(points["mic_location_2d"]["y"]) * scale,
    )
    line_width = max(3, output_width // 600)
    draw.line((ground, center), fill=(75, 235, 160), width=line_width)
    for xy, color, label in (
        (ground, (255, 208, 64), "mic_location_2d / sampled depth"),
        (center, (48, 232, 157), "mic_center_2d / reconstructed center"),
    ):
        radius = max(8, output_width // 180)
        draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), fill=color, outline="white", width=2)
        draw.text(
            (xy[0] + radius + 5, xy[1] - radius),
            label,
            font=label_font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
    draw.text(
        (18, 16),
        f"depth={pose.ground_depth:.3f}  depth_pixel={pose.depth_pixel}\n"
        f"mic_world={_format_vector(pose.microphone_center_world)}",
        font=info_font,
        fill="white",
        stroke_width=2,
        stroke_fill="black",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=94)


def rasterize_source_points(
    sources: Sequence[RenderSource],
    pose: MicrophonePose,
    output_path: str | Path,
    projection_out_path: str | Path | None = None,
    background_path: str | Path | None = None,
    width: int = 2048,
) -> None:
    """Project generated source point clouds into the microphone panorama."""
    import numpy as np
    from PIL import Image, ImageDraw

    height = width // 2
    background = None
    if background_path is not None and Path(background_path).is_file():
        with Image.open(background_path) as image:
            background = image.convert("RGB").resize((width, height))
    if background is None:
        background = Image.new("RGB", (width, height), (22, 27, 35))
    overlay = Image.new("RGBA", background.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    legend_font = _load_font(max(15, width // 95), bold=True)
    pose_matrix = np.asarray(pose.world_to_microphone, dtype=np.float64)
    saved: dict[str, Any] = {}

    legend_y = 15
    for source_index, source in enumerate(sources):
        if source.points_world is None:
            continue
        points = np.asarray(source.points_world, dtype=np.float64).reshape(-1, 3)
        camera = points @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]
        distance = np.linalg.norm(camera, axis=-1)
        valid = np.isfinite(camera).all(axis=1) & (distance > 1.0e-6)
        camera, distance = camera[valid], distance[valid]
        azimuth = np.arctan2(camera[:, 0], camera[:, 2])
        elevation = np.arctan2(-camera[:, 1], np.hypot(camera[:, 0], camera[:, 2]))
        x = (azimuth + math.pi) / (2.0 * math.pi) * (width - 1)
        y = (math.pi / 2.0 - elevation) / math.pi * (height - 1)
        saved[source.source_id] = {
            "camera_xyz": camera.astype(np.float32),
            "pixel_xy": np.stack([x, y], axis=-1).astype(np.float32),
            "distance": distance.astype(np.float32),
        }
        color = SOURCE_COLORS[source_index % len(SOURCE_COLORS)]
        radius = max(2, width // 700)
        # Far-to-near drawing provides a simple point-splat visibility cue.
        for point_index in np.argsort(distance)[::-1]:
            px, py = float(x[point_index]), float(y[point_index])
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*color, 205))
        label = f"{source.source_id}: {source.class_label}"
        draw.rectangle((12, legend_y, 34, legend_y + 18), fill=(*color, 235))
        draw.text(
            (42, legend_y),
            label,
            font=legend_font,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
        legend_y += max(25, width // 75)

    draw.line((width // 2, 0, width // 2, height), fill=(255, 255, 255, 100), width=1)
    draw.text(
        (width // 2 + 8, 8),
        "front",
        font=legend_font,
        fill=(255, 255, 255, 220),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    output = Image.alpha_composite(background.convert("RGBA"), overlay).convert("RGB")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, quality=94)
    if projection_out_path is not None:
        flat = {}
        for source_id, arrays in saved.items():
            for name, array in arrays.items():
                flat[f"{source_id}__{name}"] = array
        projection_out_path = Path(projection_out_path)
        projection_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(projection_out_path, **flat)


def compose_inference_audit(
    output_path: str | Path,
    scene_id: str,
    mic_id: str,
    panel_paths: Sequence[Path],
    source_records: Sequence[dict[str, Any]],
    generated_view_error: str | None,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    panel_size = (900, 450)
    canvas = Image.new("RGB", (1840, 1040), (18, 23, 30))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(28, bold=True)
    panel_font = _load_font(19, bold=True)
    small_font = _load_font(16)
    draw.text((22, 13), f"SonoWorld inference audit: {scene_id} / {mic_id}", font=title_font, fill="white")
    titles = ("Dataset mic reference", "Pose reconstruction", "Source point rasterization", "Generated scene at mic pose")
    for index, (title, path) in enumerate(zip(titles, panel_paths)):
        x = 20 + (index % 2) * 910
        y = 54 + (index // 2) * 485
        draw.text((x, y), title, font=panel_font, fill=(214, 223, 235))
        if path.is_file():
            with Image.open(path) as source:
                panel = ImageOps.fit(source.convert("RGB"), panel_size)
        else:
            panel = Image.new("RGB", panel_size, (35, 42, 53))
            ImageDraw.Draw(panel).text(
                (20, 20),
                generated_view_error or f"Missing {path.name}",
                font=small_font,
                fill=(255, 135, 135),
            )
        canvas.paste(panel, (x, y + 24))
    labels = ", ".join(f"{item['source_id']}={item['class_label']}" for item in source_records)
    draw.text((22, 1010), f"Sources: {labels}", font=small_font, fill=(164, 181, 201))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _format_vector(value: Any) -> str:
    return "[" + ", ".join(f"{float(item):+.3f}" for item in value) + "]"


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
