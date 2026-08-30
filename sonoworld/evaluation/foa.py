"""First-order Ambisonics metrics adapted from the internal evaluation code."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


CHANNEL_ORDER = ("W", "Y", "Z", "X")


def read_foa(path: str | Path, target_sample_rate: int = 48_000) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[1] != 4 or samples.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 4-channel FOA file at {path}, got {samples.shape}")
    channels = samples.T
    if int(sample_rate) != int(target_sample_rate):
        from sonoworld.stages.inference.base import resample_audio

        channels = np.stack(
            [resample_audio(channel, int(sample_rate), target_sample_rate) for channel in channels],
            axis=0,
        )
        sample_rate = target_sample_rate
    return np.asarray(channels, dtype=np.float32), int(sample_rate)


def center_clip(audio: Any, sample_rate: int, duration_seconds: float | None) -> Any:
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if duration_seconds is None:
        return audio
    frames = min(audio.shape[1], max(1, int(round(duration_seconds * sample_rate))))
    start = max(0, (audio.shape[1] - frames) // 2)
    return audio[:, start : start + frames]


def dominant_direction(audio: Any) -> dict[str, float]:
    """Estimate FOA intensity direction in dataset display convention.

    The encoder uses +Y=left.  SonoScene360 panoramas use image-left=-90 and
    image-right=+90, hence ``display_azimuth = -foa_azimuth``.
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] != 4 or audio.shape[1] == 0:
        raise ValueError(f"Expected FOA shape 4xT, got {audio.shape}")
    w, y, z, x = audio
    intensity_x = float(np.mean(w * x))
    intensity_y = float(np.mean(y * w))
    intensity_z = float(np.mean(z * w))
    foa_azimuth = math.atan2(intensity_y, intensity_x)
    elevation = math.atan2(intensity_z, math.hypot(intensity_x, intensity_y))
    return {
        "azimuth_degrees": -math.degrees(foa_azimuth),
        "elevation_degrees": math.degrees(elevation),
        "intensity_x": intensity_x,
        "intensity_y": intensity_y,
        "intensity_z": intensity_z,
    }


def energy_map(audio: Any, angular_resolution_degrees: float = 2.0) -> Any:
    """Decode an FOA covariance matrix over an equirectangular direction grid."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] != 4 or audio.shape[1] == 0:
        raise ValueError(f"Expected FOA shape 4xT, got {audio.shape}")
    covariance = audio @ audio.T / float(audio.shape[1])
    azimuths = np.deg2rad(np.arange(-180.0, 180.0, angular_resolution_degrees, dtype=np.float64))
    elevations = np.deg2rad(np.arange(90.0, -90.0, -angular_resolution_degrees, dtype=np.float64))
    azimuth, elevation = np.meshgrid(azimuths, elevations, indexing="xy")
    cos_elevation = np.cos(elevation)
    # Internal AmbiDecomposition uses real first-order harmonics with sqrt(2)
    # horizontal decode terms.  Negating display azimuth maps +Y to image-left.
    basis = np.stack(
        [
            np.ones_like(azimuth),
            -math.sqrt(2.0) * np.sin(azimuth) * cos_elevation,
            np.sin(elevation),
            math.sqrt(2.0) * np.cos(azimuth) * cos_elevation,
        ],
        axis=-1,
    )
    energy_squared = np.einsum("...i,ij,...j->...", basis, covariance, basis, optimize=True)
    return np.sqrt(np.maximum(energy_squared, 0.0)).astype(np.float32)


def sphere_sample(values: Any) -> Any:
    """Area-balanced row sampling used by the internal CC/AUC implementation."""
    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    height, width = data.shape
    samples: list[float] = []
    for row in range(height):
        radius = math.sin(math.pi * (row + 0.5) / height)
        count = max(1, int(round(width * radius)))
        x = np.arange(count) * (width - 0.5 * (width / count)) / count
        sample = np.interp(x=x, xp=np.arange(width), fp=data[row], period=width)
        samples.extend(sample.tolist())
    return np.asarray(samples, dtype=np.float64)


def correlation_coefficient(first: Any, second: Any) -> float:
    """Compute CC using the ViSAGe spatial-evaluation implementation.

    Source: https://github.com/jaeyeonkim99/visage/blob/main/evaluate_spatial.py
    """
    import numpy as np

    first_values = sphere_sample(first)
    second_values = sphere_sample(second)
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = math.sqrt(float(np.sum(first_values**2) * np.sum(second_values**2)))
    return float(np.sum(first_values * second_values) / denominator) if denominator > 0.0 else float("nan")


def auc_judd(saliency_map: Any, fixation_map: Any) -> float:
    """Compute AUC-Judd using the ViSAGe spatial-evaluation implementation.

    Source: https://github.com/jaeyeonkim99/visage/blob/main/evaluate_spatial.py
    """
    import numpy as np

    saliency = sphere_sample(saliency_map)
    fixation = sphere_sample(fixation_map)
    saliency_range = float(saliency.max() - saliency.min())
    if saliency_range <= 0.0:
        return float("nan")
    saliency = (saliency - saliency.min()) / saliency_range
    selected = fixation > float(fixation.max() - fixation.std())
    thresholds = np.sort(saliency[selected])[::-1]
    num_fixations = int(thresholds.size)
    num_pixels = int(saliency.size)
    if num_fixations == 0 or num_fixations >= num_pixels:
        return float("nan")
    true_positive = np.zeros(num_fixations + 2, dtype=np.float64)
    false_positive = np.zeros(num_fixations + 2, dtype=np.float64)
    true_positive[-1] = false_positive[-1] = 1.0
    for index, threshold in enumerate(thresholds):
        above = int(np.sum(saliency >= threshold))
        true_positive[index + 1] = index / num_fixations
        false_positive[index + 1] = (above - index) / (num_pixels - num_fixations)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(true_positive, false_positive))


def direction_errors(estimated: dict[str, float], ground_truth: dict[str, float]) -> dict[str, float]:
    azimuth_est = math.radians(estimated["azimuth_degrees"])
    azimuth_gt = math.radians(ground_truth["azimuth_degrees"])
    elevation_est = math.radians(estimated["elevation_degrees"])
    elevation_gt = math.radians(ground_truth["elevation_degrees"])
    azimuth_error = abs(azimuth_est - azimuth_gt)
    azimuth_error = min(azimuth_error, 2.0 * math.pi - azimuth_error)
    elevation_error = abs(elevation_est - elevation_gt)
    haversine = (
        math.sin(elevation_error / 2.0) ** 2
        + math.cos(elevation_gt) * math.cos(elevation_est) * math.sin(azimuth_error / 2.0) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    angular_error = 2.0 * math.atan2(math.sqrt(haversine), math.sqrt(1.0 - haversine))
    return {
        "azimuth_error_degrees": math.degrees(azimuth_error),
        "elevation_error_degrees": math.degrees(elevation_error),
        "angular_error_degrees": math.degrees(angular_error),
    }


def evaluate_foa_pair(
    estimated_audio: Any,
    ground_truth_audio: Any,
    angular_resolution_degrees: float = 2.0,
) -> tuple[dict[str, Any], Any, Any]:
    estimated_map = energy_map(estimated_audio, angular_resolution_degrees)
    ground_truth_map = energy_map(ground_truth_audio, angular_resolution_degrees)
    estimated_direction = dominant_direction(estimated_audio)
    ground_truth_direction = dominant_direction(ground_truth_audio)
    metrics = {
        "spatial_cc": correlation_coefficient(estimated_map, ground_truth_map),
        "auc_judd": auc_judd(estimated_map, ground_truth_map),
        **direction_errors(estimated_direction, ground_truth_direction),
        "estimated_direction": estimated_direction,
        "ground_truth_direction": ground_truth_direction,
    }
    return metrics, estimated_map, ground_truth_map


def save_energy_map(path: str | Path, values: Any, dynamic_range_db: float = 30.0) -> None:
    import numpy as np
    from PIL import Image

    values = np.asarray(values, dtype=np.float32)
    maximum = float(np.max(values))
    db = 20.0 * np.log10(np.maximum(values / max(maximum, 1.0e-12), 1.0e-8))
    floor = max(-float(dynamic_range_db), float(np.min(db)))
    normalized = np.clip((db - floor) / max(-floor, 1.0e-6), 0.0, 1.0)
    # Same perceptually ordered stops as the public dataset audit script.  A
    # piecewise map remains legible for low-dynamic-range real FOA recordings.
    stops = np.asarray(
        [
            (7, 12, 35),
            (46, 20, 89),
            (112, 29, 111),
            (191, 51, 78),
            (244, 115, 43),
            (252, 236, 128),
        ],
        dtype=np.float32,
    )
    positions = normalized * (len(stops) - 1)
    lower = np.floor(positions).astype(np.int16)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (positions - lower)[..., None]
    rgb = (stops[lower] * (1.0 - fraction) + stops[upper] * fraction).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=95)
