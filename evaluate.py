#!/usr/bin/env python3
"""Evaluate rendered FOA predictions against public SonoScene360 recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonoworld.evaluation.sonoscene360 import SonoScene360Evaluator
from sonoworld.stages.inference.default_renderer import SonoScene360Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True, help="Output root produced by inference.py")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sonoscene360_evaluation"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene-id", action="append", help="Dataset scene ID; repeat as needed")
    selection.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--mic-id", action="append", help="Restrict to mic ID; repeat as needed")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--estimated-clip-seconds", type=float, default=5.0)
    parser.add_argument(
        "--ground-truth-clip-seconds",
        type=float,
        default=8.0,
        help="Internal evaluation used 8 s GT clips and 5 s predictions",
    )
    parser.add_argument("--angular-resolution", type=float, default=2.0)
    parser.add_argument(
        "--with-clap",
        action="store_true",
        help="Also compute CLAP audio/text/QA metrics (loads LAION-CLAP checkpoint)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0 or args.angular_resolution <= 0:
        raise SystemExit("--sample-rate and --angular-resolution must be positive")
    if args.estimated_clip_seconds <= 0 or args.ground_truth_clip_seconds <= 0:
        raise SystemExit("Clip durations must be positive")

    dataset = SonoScene360Dataset.open(args.dataset_root)
    scene_ids = dataset.scene_ids if args.all_scenes else list(dict.fromkeys(args.scene_id or []))
    unknown = [scene for scene in scene_ids if scene not in dataset.scene_ids]
    if unknown:
        raise SystemExit(f"Unknown SonoScene360 scene(s): {', '.join(unknown)}")
    plan = []
    for scene_id in scene_ids:
        mics = args.mic_id or dataset.mic_ids(scene_id)
        plan.extend(
            {
                "scene_id": scene_id,
                "mic_id": mic_id,
                "prediction": str((args.predictions_root / scene_id / mic_id / "foa.wav").resolve()),
                "ground_truth": [str(path) for path in dataset.recordings(scene_id, mic_id)],
            }
            for mic_id in mics
        )
    if args.dry_run:
        print(json.dumps({"dataset_root": str(dataset.root), "plan": plan}, indent=2))
        return 0

    clap_metrics = None
    if args.with_clap:
        from sonoworld.evaluation.clap import CLAPMetrics

        print("[evaluation] loading LAION-CLAP checkpoint", flush=True)
        clap_metrics = CLAPMetrics()
    evaluator = SonoScene360Evaluator(
        dataset=dataset,
        predictions_root=args.predictions_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        estimated_clip_seconds=args.estimated_clip_seconds,
        ground_truth_clip_seconds=args.ground_truth_clip_seconds,
        angular_resolution_degrees=args.angular_resolution,
        clap_metrics=clap_metrics,
    )
    summary = evaluator.evaluate(scene_ids, mic_ids=args.mic_id, fail_fast=args.fail_fast)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
