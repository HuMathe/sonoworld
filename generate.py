from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any, Optional

from sonoworld.core.config import (
    load_config,
    get_device,
    get_stage_config,
    is_stage_enabled,
    get_stage_class_path,
)
from sonoworld.core.factory import build_stage_from_config
from sonoworld.core.stage import StageContext
from sonoworld.core.manifest import SceneManifest
from sonoworld.core.paths import ScenePaths
from sonoworld.schemas.common import FileRef, read_json, write_json
from sonoworld.schemas.understanding import normalize_known_sources


GENERATION_ORDER = [
    "outpainting",
    "understanding",
    "visual_scene",
    "segmentation",
    "audio_generation",
    "spatial_config",
]


def run_generation(
    scene_root: Path,
    config: dict,
    input_image: Optional[Path] = None,
    input_panorama: Optional[Path] = None,
    known_sources: Optional[Path] = None,
    resume: bool = False,
    force: bool = False,
) -> None:
    scene_root = Path(scene_root)
    input_image = Path(input_image) if input_image is not None else None
    input_panorama = Path(input_panorama) if input_panorama is not None else None
    known_sources = Path(known_sources) if known_sources is not None else None
    if input_image is not None and input_panorama is not None:
        raise ValueError("Provide either input_image or input_panorama, not both.")

    paths = ScenePaths(scene_root)
    paths.ensure_base_dirs()

    if paths.manifest.exists():
        manifest = SceneManifest.load(paths.manifest)
    else:
        manifest = SceneManifest.create(scene_root)
        manifest.save(paths.manifest)

    input_mode, input_path = _prepare_scene_input(
        paths,
        input_image=input_image,
        input_panorama=input_panorama,
        preferred_mode=manifest.metadata.get("input_mode"),
    )
    known_source_labels = _prepare_known_sources(paths, known_sources)
    input_hash = _file_sha256(input_path)
    known_sources_hash = _file_sha256(paths.known_sources) if paths.known_sources.exists() else None

    old_input_mode = manifest.metadata.get("input_mode")
    old_input_hash = manifest.metadata.get("input_sha256")
    old_known_sources_hash = manifest.metadata.get("known_sources_sha256")
    if resume and not force and manifest.stages:
        if old_input_mode != input_mode or old_input_hash != input_hash:
            _invalidate_stages(manifest, GENERATION_ORDER)
        elif old_known_sources_hash != known_sources_hash:
            _invalidate_stages(
                manifest,
                ["understanding", "segmentation", "audio_generation", "spatial_config"],
            )

    manifest.metadata["input_mode"] = input_mode
    manifest.metadata["input_sha256"] = input_hash
    manifest.metadata.pop("input_image", None)
    manifest.metadata.pop("input_panorama", None)
    manifest.metadata[f"input_{input_mode}"] = FileRef.from_path(
        input_path,
        scene_root=scene_root,
        role=f"input_{input_mode}",
    )
    if paths.known_sources.exists():
        manifest.metadata["known_sources"] = FileRef.from_path(
            paths.known_sources,
            scene_root=scene_root,
            role="known_sources",
            media_type="application/json",
        )
        manifest.metadata["known_source_labels"] = known_source_labels
        manifest.metadata["known_sources_sha256"] = known_sources_hash
    else:
        manifest.metadata.pop("known_sources", None)
        manifest.metadata.pop("known_source_labels", None)
        manifest.metadata.pop("known_sources_sha256", None)
    manifest.save(paths.manifest)

    ctx = StageContext(
        scene_root=scene_root,
        config=config,
        device=get_device(config),
        force=force,
    )

    for stage_name in GENERATION_ORDER:
        stage_cfg = get_stage_config(config, stage_name)

        if stage_name == "outpainting" and input_mode == "panorama":
            passthrough = _prepare_panorama_passthrough(paths, scene_root)
            manifest.mark_skipped(
                stage_name,
                backend="panorama_passthrough",
                message="Skipped because a complete equirectangular panorama was provided.",
                inputs=passthrough["inputs"],
                outputs=passthrough["outputs"],
                metadata=passthrough["metadata"],
            )
            manifest.save(paths.manifest)
            print("[outpainting] skipped, complete panorama provided.")
            continue

        if not is_stage_enabled(config, stage_name, default=True):
            manifest.mark_skipped(
                stage_name,
                backend=None,
                message="Stage is disabled.",
            )
            manifest.save(paths.manifest)
            print(f"[{stage_name}] skipped, disabled.")
            continue

        class_path = get_stage_class_path(config, stage_name)
        if class_path is None:
            raise ValueError(f"Missing class_path for enabled stage: {stage_name}")

        record = manifest.stages.get(stage_name)

        if record is not None and record.status == "done" and resume and not force:
            print(f"[{stage_name}] skipped, already done.")
            continue

        if record is not None and record.status == "waiting" and resume:
            print(f"[{stage_name}] resuming from waiting state.")

        stage = build_stage_from_config(stage_name, stage_cfg)

        try:
            manifest.mark_running(stage_name, backend=class_path)
            manifest.save(paths.manifest)

            result = stage.run(ctx)

            if result.status == "waiting":
                manifest.mark_waiting(
                    stage_name,
                    backend=class_path,
                    message=result.message,
                    inputs=result.inputs,
                    outputs=result.outputs,
                    metadata=result.metadata,
                )
                manifest.save(paths.manifest)

                print(f"[{stage_name}] waiting.")
                print(result.message or "")
                return

            if result.status == "done":
                manifest.mark_done(
                    stage_name,
                    backend=class_path,
                    message=result.message,
                    inputs=result.inputs,
                    outputs=result.outputs,
                    metadata=result.metadata,
                )
                manifest.save(paths.manifest)

                print(f"[{stage_name}] done.")
                continue

            raise RuntimeError(
                result.message or f"Stage returned unsupported status: {result.status}"
            )

        except Exception as exc:
            manifest.mark_failed(stage_name, exc=exc, backend=class_path)
            manifest.save(paths.manifest)
            raise

    print("Generation finished.")


def _prepare_scene_input(
    paths: ScenePaths,
    input_image: Optional[Path],
    input_panorama: Optional[Path],
    preferred_mode: Any = None,
) -> tuple[str, Path]:
    if input_panorama is not None:
        _store_image(input_panorama, paths.input_panorama, require_panorama=True)
        return "panorama", paths.input_panorama
    if input_image is not None:
        _store_image(input_image, paths.input_image, require_panorama=False)
        return "image", paths.input_image
    if preferred_mode == "panorama" and paths.input_panorama.exists():
        _validate_image(paths.input_panorama, require_panorama=True)
        return "panorama", paths.input_panorama
    if preferred_mode == "image" and paths.input_image.exists():
        _validate_image(paths.input_image, require_panorama=False)
        return "image", paths.input_image
    if paths.input_panorama.exists():
        _validate_image(paths.input_panorama, require_panorama=True)
        return "panorama", paths.input_panorama
    if paths.input_image.exists():
        _validate_image(paths.input_image, require_panorama=False)
        return "image", paths.input_image
    raise FileNotFoundError(
        "No scene input found. Provide --input_image or --input_panorama, or place "
        "the file at inputs/input.png or inputs/panorama.png."
    )


def _store_image(source: Path, destination: Path, require_panorama: bool) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    image = _validate_image(source, require_panorama=require_panorama)
    if source != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)


def _validate_image(path: Path, require_panorama: bool) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"Invalid image dimensions for {path}: {image.size}")
    if require_panorama:
        aspect_ratio = image.width / float(image.height)
        if not 1.9 <= aspect_ratio <= 2.1:
            raise ValueError(
                "--input_panorama expects an approximately 2:1 equirectangular image; "
                f"got {image.width}x{image.height} ({aspect_ratio:.3f}:1)."
            )
    return image


def _prepare_known_sources(paths: ScenePaths, source: Optional[Path]) -> list[str]:
    if source is None and not paths.known_sources.exists():
        return []
    source_path = source.expanduser().resolve() if source is not None else paths.known_sources
    if not source_path.is_file():
        raise FileNotFoundError(f"Known-sources JSON does not exist: {source_path}")
    labels = normalize_known_sources(read_json(source_path))
    write_json(paths.known_sources, {"known_sources": labels})
    return labels


def _prepare_panorama_passthrough(
    paths: ScenePaths,
    scene_root: Path,
) -> dict[str, Any]:
    paths.outpainting.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.input_panorama, paths.outpainted_panorama)
    summary = {
        "backend": "panorama_passthrough",
        "input_kind": "panorama",
        "source": str(paths.input_panorama),
        "output": str(paths.outpainted_panorama),
        "skipped_inference": True,
        "reason": "complete_equirectangular_panorama_input",
    }
    stage_json = paths.outpainting / "stage.json"
    write_json(stage_json, summary)
    return {
        "inputs": {
            "panorama": FileRef.from_path(
                paths.input_panorama,
                scene_root=scene_root,
                role="input_panorama",
                media_type="image/png",
            )
        },
        "outputs": {
            "panorama": FileRef.from_path(
                paths.outpainted_panorama,
                scene_root=scene_root,
                role="panorama",
                media_type="image/png",
            ),
            "summary": FileRef.from_path(
                stage_json,
                scene_root=scene_root,
                role="outpainting_summary",
                media_type="application/json",
            ),
        },
        "metadata": summary,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalidate_stages(manifest: SceneManifest, stage_names: list[str]) -> None:
    for stage_name in stage_names:
        if stage_name in manifest.stages:
            manifest.mark_pending(stage_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Perspective input image; defaults to <scene_root>/inputs/input.png.",
    )
    inputs.add_argument(
        "--input_panorama",
        type=str,
        default=None,
        help="Complete 2:1 equirectangular panorama; skips outpainting.",
    )
    parser.add_argument(
        "--known_sources",
        "--known_sources_path",
        "--know_sources",
        dest="known_sources",
        type=str,
        default=None,
        help='SonoScene360 JSON in the form {"known_sources": ["fountain", ...]}.',
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    run_generation(
        scene_root=Path(args.scene_root),
        input_image=Path(args.input_image) if args.input_image else None,
        input_panorama=Path(args.input_panorama) if args.input_panorama else None,
        known_sources=Path(args.known_sources) if args.known_sources else None,
        config=config,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
