from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sonoworld.core.artifacts import ref, save_spatial_configuration
from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.schemas.common import FileRef, read_json
from sonoworld.schemas.spatial import (
    SpatialAudioConfiguration,
    SpatialAudioOptionGroup,
    SpatialAudioSource,
    SpatialPointCloud,
)
from sonoworld.utils.audio_source_utils import normalize_audio_source_type
from sonoworld.utils.file_ref_utils import path_from_ref
from sonoworld.utils.text_utils import clean_text, slug_text


class PointClusterOmniSpatializationStage(Stage):
    """Point, cluster, and omnidirectional spatialization stage.

    This maps localized point sources, clustered area sources, and
    omnidirectional background sources into spatial audio configuration
    artifacts.
    """

    name = "spatial_config"
    backend = "point_cluster_omni"

    def __init__(
        self,
        area_num_points: int = 1000,
        point_num_points: int = 1,
        use_normals: bool = True,
        include_background: bool = False,
        selection_strategy: str = "random",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.area_num_points = int(area_num_points)
        self.point_num_points = int(point_num_points)
        self.use_normals = bool(use_normals)
        self.include_background = bool(include_background)
        self.selection_strategy = selection_strategy
        self.seed = seed

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        stage_cfg = ctx.stage_config(self.name)
        paths.spatial.mkdir(parents=True, exist_ok=True)
        paths.spatial_point_clouds.mkdir(parents=True, exist_ok=True)

        if not paths.segmentation_summary.exists():
            raise FileNotFoundError(f"Missing segmentation summary: {paths.segmentation_summary}")
        if not paths.audio_summary.exists():
            raise FileNotFoundError(f"Missing audio summary: {paths.audio_summary}")

        segmentation = read_json(paths.segmentation_summary)
        audio_summary = read_json(paths.audio_summary)
        depth, points, normals, geometry_inputs = self._load_geometry(ctx, stage_cfg)

        seed = stage_cfg.get("seed", self.seed)
        py_rng = random.Random(None if seed is None else int(seed))
        np_rng = self._numpy_rng(seed)
        audio_items = self._audio_items(audio_summary)
        local_audio = self._local_audio_by_label(audio_items)
        instances = self._instances_by_label(segmentation)

        sources: List[SpatialAudioSource] = []
        option_groups: List[SpatialAudioOptionGroup] = []
        selected_sources: List[str] = []

        if bool(stage_cfg.get("include_background", self.include_background)):
            for audio_item in self._background_audio_items(audio_items):
                source = self._make_background_source(audio_item, len(sources), ctx)
                sources.append(source)
                selected_sources.append(source.source_id)

        for label, label_instances in instances.items():
            candidates = local_audio.get(label, [])
            if not candidates:
                continue

            for instance_index, instance in enumerate(label_instances):
                instance_id = str(instance.get("instance_id", instance_index))
                instance_candidates = self._audio_candidates_for_instance(
                    candidates,
                    instance_id,
                )
                if not instance_candidates:
                    continue

                mask = self._load_instance_mask(instance, ctx, depth.shape)
                source_ids: List[str] = []

                for audio_item in instance_candidates:
                    source_type = normalize_audio_source_type(
                        audio_item.get("source_type"),
                        audio_item.get("grounding_label", label),
                    )
                    if source_type == "background":
                        continue

                    num_points = self._num_points_for_source(source_type, stage_cfg)
                    concentration = mask_concentration(
                        mask,
                        points=points,
                        depth=depth,
                        normals=normals
                        if bool(stage_cfg.get("use_normals", self.use_normals))
                        else None,
                        num=num_points,
                        rng=np_rng,
                    )
                    if concentration is None:
                        continue

                    source = self._make_point_source(
                        audio_item=audio_item,
                        instance=instance,
                        source_type=source_type,
                        concentration=concentration,
                        source_index=len(sources),
                        ctx=ctx,
                    )
                    sources.append(source)
                    source_ids.append(source.source_id)

                if source_ids:
                    selected = self._select_source(source_ids, stage_cfg, py_rng)
                    if selected is not None:
                        selected_sources.append(selected)

                option_groups.append(
                    SpatialAudioOptionGroup(
                        group_id=f"{slug_text(label, default='source')}_{instance_index}",
                        class_label=label,
                        instance_id=str(instance.get("instance_id", instance_index)),
                        source_ids=source_ids,
                        selected_source_id=selected if source_ids else None,
                        metadata={"selection_strategy": self._selection_strategy(stage_cfg)},
                    )
                )

        configuration = SpatialAudioConfiguration(
            sources=sources,
            option_groups=option_groups,
            selected_sources=selected_sources,
            coordinate_system="legacy_yzx",
            sample_rate=int(audio_summary.get("sample_rate", 0) or 0),
            backend=self.backend,
            metadata={
                "area_num_points": int(stage_cfg.get("area_num_points", self.area_num_points)),
                "point_num_points": int(stage_cfg.get("point_num_points", self.point_num_points)),
                "use_normals": bool(stage_cfg.get("use_normals", self.use_normals)),
                "include_background": bool(
                    stage_cfg.get("include_background", self.include_background)
                ),
                "geometry_inputs": geometry_inputs,
            },
        )
        summary_path = save_spatial_configuration(paths, configuration)

        inputs = {
            "segmentation": ref(
                paths.segmentation_summary,
                scene_root,
                role="segmentation_summary",
                media_type="application/json",
            ),
            "audio": ref(
                paths.audio_summary,
                scene_root,
                role="audio_generation_summary",
                media_type="application/json",
            ),
        }
        for key, path in geometry_inputs.items():
            inputs[key] = ref(Path(path), scene_root, role=key, media_type="application/octet-stream")

        return StageResult(
            status="done",
            inputs=inputs,
            outputs={
                "summary": ref(
                    summary_path,
                    scene_root,
                    role="spatial_audio_configuration",
                    media_type="application/json",
                ),
                "choice": ref(
                    paths.legacy_choice,
                    scene_root,
                    role="legacy_choice",
                    media_type="application/json",
                ),
                "options": ref(
                    paths.legacy_options,
                    scene_root,
                    role="legacy_options",
                    media_type="application/json",
                ),
                "meta_grounding": ref(
                    paths.legacy_meta_grounding,
                    scene_root,
                    role="legacy_meta_grounding",
                    media_type="application/json",
                ),
            },
            message="Point, cluster, and omnidirectional spatial configuration complete.",
            metadata={
                "backend": self.backend,
                "num_sources": len(sources),
                "num_option_groups": len(option_groups),
                "num_selected_sources": len(selected_sources),
            },
        )

    def _load_geometry(
        self,
        ctx: StageContext,
        stage_cfg: Dict[str, Any],
    ) -> Tuple[Any, Any, Optional[Any], Dict[str, str]]:
        import numpy as np

        depth_path = self._resolve_path(
            stage_cfg.get("depth_path"),
            ctx.scene_root,
            default=ctx.paths.panorama_depth_npy,
        )
        points_path = self._resolve_path(
            stage_cfg.get("points_path"),
            ctx.scene_root,
            default=ctx.paths.panorama_points_npy,
        )
        normals_path = self._resolve_path(
            stage_cfg.get("normals_path"),
            ctx.scene_root,
            default=ctx.paths.panorama_normals_npy,
        )

        if not depth_path.exists():
            raise FileNotFoundError(f"Missing panorama depth map: {depth_path}")
        if not points_path.exists():
            raise FileNotFoundError(f"Missing panorama points map: {points_path}")

        depth = np.load(depth_path).astype(np.float32)
        points = np.load(points_path).astype(np.float32)
        normals = None
        if normals_path.exists():
            normals = np.load(normals_path).astype(np.float32)
            if normals.shape[:2] != depth.shape[:2]:
                normals = None

        if points.shape[:2] != depth.shape[:2] or points.shape[-1] != 3:
            raise ValueError(
                "Panorama points must have shape HxWx3 aligned with the depth map. "
                f"Got depth={depth.shape}, points={points.shape}."
            )

        geometry_inputs = {
            "depth": str(depth_path),
            "points": str(points_path),
        }
        if normals is not None:
            geometry_inputs["normals"] = str(normals_path)
        return depth, points, normals, geometry_inputs

    def _make_background_source(
        self,
        audio_item: Dict[str, Any],
        source_index: int,
        ctx: StageContext,
    ) -> SpatialAudioSource:
        source_id = f"source_{source_index:03d}"
        return SpatialAudioSource(
            source_id=source_id,
            class_label=str(audio_item.get("class_label") or audio_item.get("grounding_label")),
            grounding_label=str(audio_item.get("grounding_label", "global")),
            source_type="background",
            audio=self._audio_ref(audio_item, ctx),
            prompt_id=optional_int(audio_item.get("prompt_id")),
            peak_db=float(audio_item.get("peak_db", -28.0)),
            metadata={
                "audio_id": audio_item.get("audio_id"),
                "candidate_paths": self._candidate_paths(audio_item),
            },
        )

    def _make_point_source(
        self,
        audio_item: Dict[str, Any],
        instance: Dict[str, Any],
        source_type: str,
        concentration: Dict[str, Any],
        source_index: int,
        ctx: StageContext,
    ) -> SpatialAudioSource:
        source_id = f"source_{source_index:03d}"
        stem = f"{source_index:02d}"
        full_points_path = ctx.paths.spatial_point_clouds / f"{stem}_points.ply"
        down_points_path = ctx.paths.spatial_point_clouds / f"{stem}_points_down.ply"
        weights_path = ctx.paths.spatial_point_clouds / f"{stem}_points_weights.npy"

        self._export_point_cloud(concentration["pcd_points"], full_points_path)
        self._export_point_cloud(concentration["pcd_points_down"], down_points_path)
        self._save_weights(concentration["pcd_weights"], weights_path)

        centroid_pano = concentration["centroid"]
        centroid_legacy = reorder_xyz_to_legacy(centroid_pano).tolist()
        point_cloud = SpatialPointCloud(
            full_points=ref(
                full_points_path,
                ctx.scene_root,
                role="spatial_point_cloud",
                media_type="model/ply",
            ),
            downsampled_points=ref(
                down_points_path,
                ctx.scene_root,
                role="spatial_point_cloud_downsampled",
                media_type="model/ply",
            ),
            weights=ref(
                weights_path,
                ctx.scene_root,
                role="spatial_point_weights",
                media_type="application/octet-stream",
            ),
            num_full_points=int(concentration["pcd_points"].shape[0]),
            num_downsampled_points=int(concentration["pcd_points_down"].shape[0]),
            metadata={"coordinate_system": "legacy_yzx"},
        )

        return SpatialAudioSource(
            source_id=source_id,
            class_label=str(audio_item.get("class_label") or instance.get("class_label")),
            grounding_label=str(audio_item.get("grounding_label") or instance.get("class_label")),
            source_type=source_type,
            audio=self._audio_ref(audio_item, ctx),
            instance_id=str(instance.get("instance_id", "")) or None,
            prompt_id=optional_int(audio_item.get("prompt_id")),
            centroid=centroid_legacy,
            point_cloud=point_cloud,
            peak_db=float(audio_item.get("peak_db", -18.0)),
            metadata={
                "audio_id": audio_item.get("audio_id"),
                "candidate_paths": self._candidate_paths(audio_item),
                "centroid_pano": centroid_pano.tolist(),
            },
        )

    def _audio_items(self, audio_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = audio_summary.get("items", [])
        if not isinstance(items, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            primary_path = path_from_ref(item.get("primary"))
            candidate_paths = self._candidate_paths(item)
            if primary_path is None and candidate_paths:
                primary_path = candidate_paths[0]
            if primary_path is None:
                continue

            grounding_label = clean_text(item.get("grounding_label")) or clean_text(
                item.get("class_label")
            )
            if not grounding_label:
                continue

            source_type = normalize_audio_source_type(item.get("source_type"), grounding_label)
            normalized.append(
                {
                    **item,
                    "primary": primary_path,
                    "class_label": clean_text(item.get("class_label")) or grounding_label,
                    "grounding_label": grounding_label,
                    "source_type": source_type,
                    "peak_db": float(item.get("peak_db", -28.0 if source_type == "background" else -18.0)),
                }
            )
        return normalized

    def _local_audio_by_label(self, audio_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in audio_items:
            label = str(item.get("grounding_label", ""))
            if label.lower() == "global" or item.get("source_type") == "background":
                continue
            grouped.setdefault(label, []).append(item)
        return grouped

    def _audio_candidates_for_instance(
        self,
        candidates: List[Dict[str, Any]],
        instance_id: str,
    ) -> List[Dict[str, Any]]:
        instance_id = clean_text(instance_id)
        exact = [
            item
            for item in candidates
            if clean_text(item.get("instance_id")) == instance_id
        ]
        if exact:
            return exact

        # Backward compatibility for summaries produced before audio items were
        # associated with segmentation instances.
        return [
            item
            for item in candidates
            if not clean_text(item.get("instance_id"))
        ]

    def _background_audio_items(self, audio_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            item
            for item in audio_items
            if str(item.get("grounding_label", "")).lower() == "global"
            or item.get("source_type") == "background"
        ]

    def _instances_by_label(self, segmentation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for instance in segmentation.get("instances", []):
            if not isinstance(instance, dict):
                continue
            label = clean_text(instance.get("class_label"))
            if not label:
                continue
            grouped.setdefault(label, []).append(instance)
        return grouped

    def _load_instance_mask(
        self,
        instance: Dict[str, Any],
        ctx: StageContext,
        target_shape: Tuple[int, int],
    ) -> Any:
        import numpy as np
        from PIL import Image

        mask_path = path_from_ref(instance.get("mask"))
        if mask_path is None:
            raise ValueError(f"Segmentation instance has no mask path: {instance}")

        path = self._resolve_path(mask_path, ctx.scene_root)
        mask = np.asarray(Image.open(path).convert("L")) > 127
        if mask.shape != target_shape:
            resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
            image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask = np.asarray(image.resize((target_shape[1], target_shape[0]), resampling)) > 127
        return mask

    def _audio_ref(self, audio_item: Dict[str, Any], ctx: StageContext) -> FileRef:
        path = str(audio_item["primary"])
        resolved = self._resolve_path(path, ctx.scene_root)
        return ref(resolved, ctx.scene_root, role="audio", media_type="audio/wav")

    def _candidate_paths(self, audio_item: Dict[str, Any]) -> List[str]:
        candidates = audio_item.get("candidates", [])
        if not isinstance(candidates, list):
            return []
        paths: List[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            path = path_from_ref(candidate.get("path"))
            if path is not None:
                paths.append(path)
        return paths

    def _num_points_for_source(self, source_type: str, stage_cfg: Dict[str, Any]) -> int:
        if source_type == "point":
            return max(1, int(stage_cfg.get("point_num_points", self.point_num_points)))
        return max(1, int(stage_cfg.get("area_num_points", self.area_num_points)))

    def _select_source(
        self,
        source_ids: List[str],
        stage_cfg: Dict[str, Any],
        rng: random.Random,
    ) -> Optional[str]:
        if not source_ids:
            return None
        if self._selection_strategy(stage_cfg) == "random":
            return rng.choice(source_ids)
        return source_ids[0]

    def _selection_strategy(self, stage_cfg: Dict[str, Any]) -> str:
        strategy = str(stage_cfg.get("selection_strategy", self.selection_strategy)).strip().lower()
        return "random" if strategy == "random" else "first"

    def _resolve_path(
        self,
        value: Any,
        scene_root: Path,
        default: Optional[Path] = None,
    ) -> Path:
        if value is None or value == "":
            if default is None:
                raise ValueError("No path value was provided.")
            return default

        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return scene_root / path

    def _numpy_rng(self, seed: Any) -> Any:
        import numpy as np

        return np.random.default_rng(None if seed is None else int(seed))

    def _export_point_cloud(self, points: Any, path: Path) -> None:
        import trimesh

        path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=reorder_xyz_to_legacy(points), process=False).export(str(path))

    def _save_weights(self, weights: Any, path: Path) -> None:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, weights)


def mask_concentration(
    pano_mask: Any,
    points: Any,
    depth: Any,
    normals: Optional[Any] = None,
    num: int = 1000,
    rng: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    import numpy as np

    pano_mask = np.asarray(pano_mask).astype(bool)
    points = np.asarray(points, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    if not pano_mask.any():
        return None

    height, width = depth.shape[:2]
    u = np.zeros_like(depth, dtype=np.float32)
    v = np.zeros_like(depth, dtype=np.float32)
    u[:, :] = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
    v[:, :] = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32)[:, None]
    theta = (1.0 - u) * 2.0 * np.pi
    phi = v * np.pi

    weights = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) ** 2
    weights = weights * np.sin(phi)

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        rays = np.stack(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ],
            axis=-1,
        )
        normal_factor = np.abs(np.einsum("ijk,ijk->ij", rays, normals))
        weights = weights / np.clip(normal_factor, a_min=1e-5, a_max=None)

    weights = np.nan_to_num(weights, nan=0.0, posinf=1e5, neginf=0.0)
    weights[weights > 1e5] = 1e5

    pcd_points = points[pano_mask]
    pcd_weights = weights[pano_mask].astype(np.float64)
    if pcd_points.shape[0] == 0:
        return None

    total_weight = float(pcd_weights.sum())
    if total_weight <= 0.0:
        pcd_weights = np.full(pcd_points.shape[0], 1.0 / pcd_points.shape[0], dtype=np.float64)
    else:
        pcd_weights = pcd_weights / total_weight

    centroid = np.sum(pcd_points * pcd_weights[:, None].astype(np.float32), axis=0)
    sample_count = min(max(1, int(num)), pcd_points.shape[0])
    if rng is None:
        rng = np.random.default_rng()
    keep_indices = rng.choice(
        np.arange(pcd_points.shape[0]),
        size=sample_count,
        replace=False,
    )
    pcd_points_down = pcd_points[keep_indices]

    return {
        "centroid": centroid.astype(np.float32),
        "pcd_points": pcd_points.astype(np.float32),
        "pcd_points_down": pcd_points_down.astype(np.float32),
        "pcd_weights": pcd_weights.astype(np.float32),
    }


def optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def reorder_xyz_to_legacy(points: Any) -> Any:
    import numpy as np

    arr = np.asarray(points)
    return arr[..., [1, 2, 0]]


# Backward-compatible aliases for older config class paths.
PCOSpatializationStage = PointClusterOmniSpatializationStage
PointSourceSpatialConfigStage = PointClusterOmniSpatializationStage
