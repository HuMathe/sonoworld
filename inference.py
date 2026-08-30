#!/usr/bin/env python3
r"""Render generated SonoWorld scenes at public SonoScene360 mic poses.

Examples:
  python inference.py --dataset-root /path/to/SonoScene360 \
      --scene-id fountain-multi --scene-root outputs/fountain-multi

  python inference.py --dataset-root /path/to/SonoScene360 \
      --scenes-root outputs --all-scenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sonoworld.stages.inference.default_renderer import (
    SonoScene360Dataset,
    SonoScene360InferenceRenderer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of the public SonoScene360 checkout")
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--scene-root", type=Path, help="One generated scene directory")
    roots.add_argument("--scenes-root", type=Path, help="Directory containing <scene-id>/ generated scenes")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene-id", action="append", help="Dataset scene ID; repeat for multiple scenes")
    selection.add_argument("--all-scenes", action="store_true", help="Render every public dataset scene")
    parser.add_argument("--mic-id", action="append", help="Only render this mic ID; repeat as needed")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sonoscene360_inference"))
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--device", default="cuda", help="Device used only for optional Gaussian-view rasterization")
    parser.add_argument("--visualization-width", type=int, default=2048)
    parser.add_argument("--skip-generated-view", action="store_true", help="Skip Marble Gaussian view rasterizations")
    parser.add_argument("--require-generated-view", action="store_true", help="Fail if Gaussian-view rasterization is unavailable")
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Do not add the generated global ambience (internal evaluation included it)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate mic outputs even if a manifest exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the scene/mic plan without rendering")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be positive")
    if args.sample_rate <= 0 or args.visualization_width < 256:
        raise SystemExit("--sample-rate must be positive and --visualization-width must be at least 256")
    if args.scene_root is not None and args.all_scenes:
        raise SystemExit("--scene-root can only be used with one --scene-id")
    if args.scene_root is not None and (not args.scene_id or len(args.scene_id) != 1):
        raise SystemExit("--scene-root requires exactly one --scene-id")

    dataset = SonoScene360Dataset.open(args.dataset_root)
    scene_ids = dataset.scene_ids if args.all_scenes else list(dict.fromkeys(args.scene_id or []))
    unknown = [scene_id for scene_id in scene_ids if scene_id not in dataset.scene_ids]
    if unknown:
        raise SystemExit(f"Unknown SonoScene360 scene(s): {', '.join(unknown)}")

    plan = []
    for scene_id in scene_ids:
        scene_root = args.scene_root if args.scene_root is not None else args.scenes_root / scene_id
        mic_ids = args.mic_id or dataset.mic_ids(scene_id)
        unknown_mics = [mic_id for mic_id in mic_ids if mic_id not in dataset.mic_ids(scene_id)]
        if unknown_mics:
            raise SystemExit(f"Unknown mic(s) for {scene_id}: {', '.join(unknown_mics)}")
        plan.append({"scene_id": scene_id, "scene_root": str(scene_root.resolve()), "mic_ids": mic_ids})
    if args.dry_run:
        print(json.dumps({"dataset_root": str(dataset.root), "output_root": str(args.output_root.resolve()), "plan": plan}, indent=2))
        return 0

    renderer = SonoScene360InferenceRenderer(
        dataset=dataset,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        device=args.device,
        render_generated_view=not args.skip_generated_view,
        require_generated_view=args.require_generated_view,
        visualization_width=args.visualization_width,
        include_background=not args.exclude_background,
    )
    results = []
    errors = []
    for item in plan:
        scene_id = item["scene_id"]
        print(f"[inference] {scene_id}: {len(item['mic_ids'])} microphone(s)", flush=True)
        try:
            results.extend(
                renderer.render_scene(
                    scene_id,
                    item["scene_root"],
                    args.output_root,
                    mic_ids=item["mic_ids"],
                    force=args.force,
                )
            )
        except Exception as exc:
            error = {"scene_id": scene_id, "type": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            print(f"[inference] ERROR {scene_id}: {error['type']}: {error['message']}", file=sys.stderr)
            if args.fail_fast:
                raise

    summary = {
        "schema_version": 1,
        "status": "done" if not errors else "partial",
        "dataset_root": str(dataset.root),
        "output_root": str(args.output_root.resolve()),
        "num_microphones": len(results),
        "results": [f"{item['scene_id']}/{item['mic_id']}/manifest.json" for item in results],
        "errors": errors,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
