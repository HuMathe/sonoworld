"""Shared primitives for rendering generated SonoWorld scenes at real mic poses.

The public SonoScene360 release stores the AprilTag rotation and two panorama
points, but intentionally does not store the translation used by the internal
evaluation scripts.  The translation is reconstructed here from the generated
panorama depth with the same geometry as ``data_gen_mic_annotation.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TARGET_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class MicrophonePose:
    """Reconstructed OpenCV world-to-microphone pose."""

    world_to_microphone: Any
    microphone_center_world: Any
    ground_point_world: Any
    ground_depth: float
    ground_uv: tuple[float, float]
    center_uv: tuple[float, float]
    depth_pixel: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("world_to_microphone", "microphone_center_world", "ground_point_world"):
            value[key] = value[key].tolist()
        return value


@dataclass(frozen=True)
class RenderSource:
    """One mono source and its spatial support in panorama-world coordinates."""

    source_id: str
    class_label: str
    source_type: str
    audio_path: Path
    points_world: Any | None
    instance_id: str | None = None


def panorama_ray(u: float, v: float) -> Any:
    """Return a panorama-world ray for normalized equirectangular coordinates."""
    import numpy as np

    phi = float(v) * math.pi
    theta = (1.0 - float(u)) * 2.0 * math.pi
    return np.asarray(
        [
            math.cos(theta) * math.sin(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(phi),
        ],
        dtype=np.float64,
    )


def reconstruct_microphone_pose(
    mic_metadata: dict[str, Any],
    panorama_size: tuple[int, int],
    panorama_depth: Any,
) -> MicrophonePose:
    """Recover translation from the released 2-D annotations and scene depth.

    ``mic_location_2d`` is the microphone's ground projection, while
    ``mic_center_2d`` is its actual center.  Both coordinates refer to the raw
    calibration panorama; normalized UVs therefore remain valid for a depth map
    rendered at a different equirectangular resolution.
    """
    import numpy as np

    width, height = panorama_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid panorama size: {panorama_size}")

    depth = np.asarray(panorama_depth, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise ValueError(f"Expected an HxW panorama depth map, got {depth.shape}")

    points = mic_metadata.get("2d_points", {})
    ground = points.get("mic_location_2d")
    center = points.get("mic_center_2d")
    if not isinstance(ground, dict) or not isinstance(center, dict):
        raise ValueError("Mic metadata is missing mic_location_2d or mic_center_2d")

    ground_u = float(ground["x"]) / float(width)
    ground_v = float(ground["y"]) / float(height)
    center_u = float(center["x"]) / float(width)
    center_v = float(center["y"]) / float(height)
    for name, value in (
        ("ground_u", ground_u),
        ("ground_v", ground_v),
        ("center_u", center_u),
        ("center_v", center_v),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name}={value} lies outside normalized panorama bounds")

    row = min(depth.shape[0] - 1, max(0, int(ground_v * depth.shape[0] + 0.5)))
    col = min(depth.shape[1] - 1, max(0, int(ground_u * depth.shape[1] + 0.5)))
    ground_depth, sampled_row, sampled_col = _nearest_valid_depth(depth, row, col)

    ground_phi = ground_v * math.pi
    center_phi = center_v * math.pi
    center_sine = math.sin(center_phi)
    if abs(center_sine) < 1.0e-6:
        raise ValueError("mic_center_2d is too close to an equirectangular pole")

    # This is the same spherical-geometry correction used by the internal code.
    center_distance = ground_depth * math.sin(ground_phi) / center_sine
    ground_world = panorama_ray(ground_u, ground_v) * ground_depth
    center_world = panorama_ray(center_u, center_v) * center_distance

    rotation = np.asarray(mic_metadata.get("apriltag", {}).get("rotation"), dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 AprilTag rotation, got {rotation.shape}")
    if not np.isfinite(rotation).all():
        raise ValueError("AprilTag rotation contains non-finite values")
    ortho_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if ortho_error > 5.0e-2 or abs(determinant - 1.0) > 5.0e-2:
        raise ValueError(
            "AprilTag rotation is not a valid rotation "
            f"(orthogonality error={ortho_error:.4g}, determinant={determinant:.4g})"
        )

    world_to_microphone = np.eye(4, dtype=np.float64)
    world_to_microphone[:3, :3] = rotation
    world_to_microphone[:3, 3] = -rotation @ center_world
    return MicrophonePose(
        world_to_microphone=world_to_microphone.astype(np.float32),
        microphone_center_world=center_world.astype(np.float32),
        ground_point_world=ground_world.astype(np.float32),
        ground_depth=float(ground_depth),
        ground_uv=(ground_u, ground_v),
        center_uv=(center_u, center_v),
        depth_pixel=(sampled_col, sampled_row),
    )


def _nearest_valid_depth(depth: Any, row: int, col: int, max_radius: int = 32) -> tuple[float, int, int]:
    import numpy as np

    value = float(depth[row, col])
    if math.isfinite(value) and value > 0.0:
        return value, row, col

    for radius in (1, 2, 4, 8, 16, max_radius):
        y0, y1 = max(0, row - radius), min(depth.shape[0], row + radius + 1)
        x0, x1 = max(0, col - radius), min(depth.shape[1], col + radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid = np.isfinite(patch) & (patch > 0.0)
        if valid.any():
            yy, xx = np.nonzero(valid)
            distances = (yy + y0 - row) ** 2 + (xx + x0 - col) ** 2
            index = int(np.argmin(distances))
            sampled_row, sampled_col = int(yy[index] + y0), int(xx[index] + x0)
            return float(depth[sampled_row, sampled_col]), sampled_row, sampled_col
    raise ValueError(f"No valid depth near pixel ({col}, {row})")


def load_render_sources(
    scene_root: str | Path,
    include_background: bool = True,
) -> list[RenderSource]:
    """Load selected spatial sources and, optionally, generated ambience.

    The internal evaluation invoked ``audio_render_eval.py --load_global``.
    Public spatial configurations may omit ambience from ``selected_sources``,
    so the first generated background item is recovered from ``audio/summary``.
    """
    import json
    import numpy as np

    scene_root = Path(scene_root)
    summary_path = scene_root / "spatial" / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Missing spatial configuration: {summary_path}") from exc

    sources = {
        str(item.get("source_id")): item
        for item in summary.get("sources", [])
        if isinstance(item, dict) and item.get("source_id") is not None
    }
    selected = summary.get("selected_sources", [])
    if not isinstance(selected, list):
        raise ValueError(f"selected_sources must be a list in {summary_path}")
    coordinate_system = str(summary.get("coordinate_system", "pano"))

    output: list[RenderSource] = []
    for source_id in selected:
        item = sources.get(str(source_id))
        if item is None:
            continue
        audio_path = _resolve_ref(scene_root, item.get("audio"))
        if audio_path is None or not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio for selected source {source_id}: {audio_path}")

        point_path = None
        cloud = item.get("point_cloud")
        if isinstance(cloud, dict):
            point_path = _resolve_ref(scene_root, cloud.get("downsampled_points"))
            if point_path is None:
                point_path = _resolve_ref(scene_root, cloud.get("full_points"))

        points_world = None
        if point_path is not None and point_path.is_file():
            points_world = load_ply_points(point_path)
            cloud_coordinate_system = str(cloud.get("metadata", {}).get("coordinate_system", coordinate_system))
            if cloud_coordinate_system == "legacy_yzx":
                points_world = points_world[..., [2, 0, 1]]
        elif item.get("centroid") is not None:
            points_world = np.asarray(item["centroid"], dtype=np.float32).reshape(1, 3)
            if coordinate_system == "legacy_yzx":
                points_world = points_world[..., [2, 0, 1]]

        source_type = str(item.get("source_type", "area"))
        if source_type != "background" and points_world is None:
            raise FileNotFoundError(f"Selected source {source_id} has no point cloud or centroid")
        output.append(
            RenderSource(
                source_id=str(source_id),
                class_label=str(item.get("class_label") or item.get("grounding_label") or source_id),
                source_type=source_type,
                audio_path=audio_path,
                points_world=points_world,
                instance_id=str(item["instance_id"]) if item.get("instance_id") is not None else None,
            )
        )
    if include_background and not any(source.source_type == "background" for source in output):
        audio_summary_path = scene_root / "audio" / "summary.json"
        if audio_summary_path.is_file():
            audio_summary = json.loads(audio_summary_path.read_text(encoding="utf-8"))
            for item in audio_summary.get("items", []):
                if not isinstance(item, dict):
                    continue
                grounding_label = str(item.get("grounding_label", ""))
                source_type = str(item.get("source_type", ""))
                if grounding_label.lower() != "global" and source_type != "background":
                    continue
                audio_path = _resolve_ref(scene_root, item.get("primary"))
                if audio_path is None:
                    candidates = item.get("candidates", [])
                    first_candidate = candidates[0] if isinstance(candidates, list) and candidates else None
                    if isinstance(first_candidate, dict):
                        audio_path = _resolve_ref(scene_root, first_candidate.get("path"))
                if audio_path is None or not audio_path.is_file():
                    raise FileNotFoundError(f"Missing generated background audio: {audio_path}")
                output.append(
                    RenderSource(
                        source_id=f"background_{item.get('audio_id', 'global')}",
                        class_label=str(item.get("class_label") or "global"),
                        source_type="background",
                        audio_path=audio_path,
                        points_world=None,
                    )
                )
                break
    return output


def _resolve_ref(scene_root: Path, value: Any) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path")
        if isinstance(value, dict):
            value = value.get("path")
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else scene_root / path


def load_ply_points(path: str | Path) -> Any:
    import numpy as np
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    return np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=-1).astype(np.float32)


def ambisonic_coefficients(points_world: Any, world_to_microphone: Any) -> Any:
    """Match internal ``ClusteredPointSource.ambisonic_coefficients``.

    OpenCV mic coordinates are converted to hearing coordinates as
    ``[front, left, up] = [z, -x, -y]``.  The returned ACN/SN3D channel order is
    ``[W, Y, Z, X]``.
    """
    import numpy as np

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(world_to_microphone, dtype=np.float64)
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    hearing = np.stack([camera[:, 2], -camera[:, 0], -camera[:, 1]], axis=-1)
    distances = np.linalg.norm(hearing, axis=-1, keepdims=True)
    directions = hearing / np.maximum(distances, 1.0e-6)
    coefficients = np.stack(
        [
            np.ones(len(directions), dtype=np.float64),
            directions[:, 1],
            directions[:, 2],
            directions[:, 0],
        ],
        axis=-1,
    ) / np.maximum(distances, 1.0e-6)
    return coefficients.mean(axis=0).astype(np.float32)


def render_foa(
    sources: Sequence[RenderSource],
    world_to_microphone: Any,
    sample_rate: int = TARGET_SAMPLE_RATE,
    duration_seconds: float = 8.0,
) -> tuple[Any, list[dict[str, Any]]]:
    """Render selected mono sources to a fixed-duration four-channel FOA buffer."""
    import numpy as np

    frame_count = max(1, int(round(float(duration_seconds) * sample_rate)))
    output = np.zeros((4, frame_count), dtype=np.float32)
    source_records: list[dict[str, Any]] = []
    for source in sources:
        mono, source_rate = read_audio_mono(source.audio_path)
        mono = resample_audio(mono, source_rate, sample_rate)
        mono = fit_audio_duration(mono, frame_count, loop=source.source_type == "background")
        if source.source_type == "background":
            coefficients = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            point_count = 0
        else:
            coefficients = ambisonic_coefficients(source.points_world, world_to_microphone)
            point_count = int(len(source.points_world))
        output += coefficients[:, None] * mono[None, :]
        source_records.append(
            {
                "source_id": source.source_id,
                "class_label": source.class_label,
                "instance_id": source.instance_id,
                "source_type": source.source_type,
                "audio_path": str(source.audio_path),
                "num_points": point_count,
                "foa_coefficients_wyzx": coefficients.tolist(),
            }
        )
    return output, source_records


def read_audio_mono(path: str | Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError(f"Empty audio file: {path}")
    return np.mean(samples, axis=1, dtype=np.float32), int(sample_rate)


def resample_audio(samples: Any, source_rate: int, target_rate: int) -> Any:
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate:
        return samples
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


def fit_audio_duration(samples: Any, frame_count: int, loop: bool = False) -> Any:
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size >= frame_count:
        return samples[:frame_count]
    if loop and samples.size:
        return np.tile(samples, int(math.ceil(frame_count / samples.size)))[:frame_count]
    return np.pad(samples, (0, frame_count - samples.size))


def write_foa(path: str | Path, audio: Any, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    import numpy as np
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels_first = np.asarray(audio, dtype=np.float32)
    if channels_first.ndim != 2 or channels_first.shape[0] != 4:
        raise ValueError(f"Expected FOA shape 4xT, got {channels_first.shape}")
    sf.write(str(path), np.clip(channels_first.T, -1.0, 1.0), sample_rate, subtype="PCM_16")


def scene_mics(dataset_metadata: dict[str, Any], scene_id: str) -> list[str]:
    mapping = dataset_metadata.get("scene_mic_metadata_paths", {}).get(scene_id, {})
    if not isinstance(mapping, dict):
        return []
    return sorted(str(key) for key in mapping)


def iter_scene_recordings(
    dataset_metadata: dict[str, Any], scene_id: str, mic_id: str
) -> Iterable[str]:
    mapping = dataset_metadata.get("scene_mic_recording_paths", {}).get(scene_id, {}).get(mic_id, [])
    return [str(path) for path in mapping] if isinstance(mapping, list) else []
