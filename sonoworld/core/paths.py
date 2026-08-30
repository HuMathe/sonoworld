from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScenePaths:
    """Canonical file layout for one scene."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "scene_manifest.json"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def input_image(self) -> Path:
        return self.inputs / "input.png"

    @property
    def input_panorama(self) -> Path:
        return self.inputs / "panorama.png"

    @property
    def known_sources(self) -> Path:
        return self.inputs / "known_sources.json"

    @property
    def outpainting(self) -> Path:
        return self.root / "outpainting"

    @property
    def outpainted_panorama(self) -> Path:
        return self.outpainting / "panorama_outpainted.png"

    @property
    def understanding(self) -> Path:
        return self.root / "understanding"

    @property
    def understanding_metadata(self) -> Path:
        return self.understanding / "metadata.json"

    @property
    def grounding_input(self) -> Path:
        return self.segmentation / "grounding-input.json"

    @property
    def segmentation(self) -> Path:
        return self.root / "segmentation"

    @property
    def segmentation_summary(self) -> Path:
        return self.segmentation / "summary.json"

    @property
    def visual_scene(self) -> Path:
        return self.root / "visual_scene"

    @property
    def visual_scene_summary(self) -> Path:
        return self.visual_scene / "summary.json"

    @property
    def panorama_depth_npy(self) -> Path:
        return self.visual_scene / "panorama_depth.npy"

    @property
    def panorama_depth_exr(self) -> Path:
        return self.visual_scene / "panorama_depth.exr"

    @property
    def panorama_points_npy(self) -> Path:
        return self.visual_scene / "points.npy"

    @property
    def panorama_normals_npy(self) -> Path:
        return self.visual_scene / "normals.npy"

    @property
    def panorama_mask(self) -> Path:
        return self.visual_scene / "mask.png"

    @property
    def marble_request(self) -> Path:
        return self.visual_scene / "marble_request"

    @property
    def marble_representation(self) -> Path:
        return self.visual_scene / "representation" / "marble"

    @property
    def marble_gaussian(self) -> Path:
        return self.marble_representation / "gaussian.ply"

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def audio_assets(self) -> Path:
        return self.audio / "assets"

    @property
    def audio_summary(self) -> Path:
        return self.audio / "summary.json"

    @property
    def spatial(self) -> Path:
        return self.root / "spatial"

    @property
    def spatial_summary(self) -> Path:
        return self.spatial / "summary.json"

    @property
    def spatial_point_clouds(self) -> Path:
        return self.spatial / "point_clouds"

    @property
    def legacy_choice(self) -> Path:
        return self.spatial / "choice.json"

    @property
    def legacy_options(self) -> Path:
        return self.spatial / "options.json"

    @property
    def legacy_meta_grounding(self) -> Path:
        return self.spatial / "meta-grounding.json"

    @property
    def inference(self) -> Path:
        return self.root / "inference"

    def ensure_base_dirs(self) -> None:
        for path in [
            self.inputs,
            self.outpainting,
            self.understanding,
            self.segmentation,
            self.visual_scene,
            self.audio,
            self.audio_assets,
            self.spatial,
            self.spatial_point_clouds,
            self.inference,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def find_panorama(self) -> Path:
        candidates = [
            self.outpainted_panorama,
            self.input_panorama,
            self.root / "panorama.png",
            self.root / "panorama.jpg",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError("No panorama image found in scene root.")

    def find_input_image(self) -> Path:
        candidates = [
            self.input_image,
            self.root / "input.png",
            self.root / "input.jpg",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError("No input image found in scene root.")
