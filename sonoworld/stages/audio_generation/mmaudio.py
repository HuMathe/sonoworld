from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sonoworld.core.artifacts import ref, save_audio_summary
from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.schemas.audio import AudioCandidate, AudioGenerationSummary, AudioItem
from sonoworld.schemas.common import FileRef, read_json
from sonoworld.utils.audio_source_utils import normalize_audio_source_type, normalize_peak_db
from sonoworld.utils.text_utils import clean_text, slug_text


DEFAULT_MODEL_VARIANT = "large_44k_v2"
DEFAULT_MODEL_PATH = ".cache/weights/mmaudio"


class MMAudioGenerationStage(Stage):
    """MMAudio text-to-audio generation stage.

    Heavy audio dependencies are imported only while the stage is running so
    configs that disable this stage can still import the module in leaner envs.
    """

    name = "audio_generation"
    backend = "mmaudio"

    def __init__(
        self,
        model_name: str = "mmaudio",
        model_variant: str = DEFAULT_MODEL_VARIANT,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seconds_total: float = 8.0,
        full_precision: bool = True,
        inference_mode: str = "euler",
        model_path: str = DEFAULT_MODEL_PATH,
        num_candidates: int = 3,
        sample_rate: Optional[int] = None,
        negative_prompt: str = "Low quality.",
        apply_peak_gain: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.model_variant = model_variant
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.seconds_total = float(seconds_total)
        self.full_precision = bool(full_precision)
        self.inference_mode = inference_mode
        self.model_path = model_path
        self.num_candidates = int(num_candidates)
        self.sample_rate = sample_rate
        self.negative_prompt = negative_prompt
        self.apply_peak_gain = bool(apply_peak_gain)
        self._model: Any = None
        self._model_key: Optional[Tuple[Any, ...]] = None

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        stage_cfg = ctx.stage_config(self.name)
        paths.audio.mkdir(parents=True, exist_ok=True)
        paths.audio_assets.mkdir(parents=True, exist_ok=True)

        if not paths.understanding_metadata.exists():
            raise FileNotFoundError(
                f"Missing understanding metadata: {paths.understanding_metadata}"
            )
        if not paths.segmentation_summary.exists():
            raise FileNotFoundError(
                f"Missing segmentation summary: {paths.segmentation_summary}"
            )

        understanding_payload = read_json(paths.understanding_metadata)
        if isinstance(understanding_payload, list):
            understanding = {"items": understanding_payload, "negative_prompt": self.negative_prompt}
        elif isinstance(understanding_payload, dict):
            understanding = understanding_payload
        else:
            raise TypeError(
                "Understanding metadata must be a JSON object or a legacy item list."
            )
        prompt_items = self._understanding_audio_items(understanding)
        segmentation = read_json(paths.segmentation_summary)
        instances_by_label = self._segmentation_instances_by_label(segmentation)
        generation_targets = self._generation_targets(prompt_items, instances_by_label)
        num_candidates = max(1, int(stage_cfg.get("num_candidates", self.num_candidates)))
        seconds_total = float(stage_cfg.get("seconds_total", self.seconds_total))
        negative_prompt = str(
            stage_cfg.get(
                "negative_prompt",
                understanding.get("negative_prompt", self.negative_prompt),
            )
        )

        if not prompt_items:
            summary = AudioGenerationSummary(
                items=[],
                sample_rate=int(stage_cfg.get("sample_rate", self.sample_rate or 44100)),
                default_duration_sec=seconds_total,
                backend=self.backend,
                model=str(stage_cfg.get("model_variant", self.model_variant)),
                metadata={"message": "No audio prompts were enabled."},
            )
            summary_path = save_audio_summary(paths, summary)
            return self._make_stage_result(ctx, summary_path, summary, prompt_count=0)

        if not generation_targets:
            summary = AudioGenerationSummary(
                items=[],
                sample_rate=int(stage_cfg.get("sample_rate", self.sample_rate or 44100)),
                default_duration_sec=seconds_total,
                backend=self.backend,
                model=str(stage_cfg.get("model_variant", self.model_variant)),
                metadata={"message": "No segmented audio instances were found."},
            )
            summary_path = save_audio_summary(paths, summary)
            return self._make_stage_result(
                ctx,
                summary_path,
                summary,
                prompt_count=len(prompt_items),
            )

        model = self._get_model(ctx, stage_cfg, seconds_total=seconds_total)
        model_sample_rate = self._model_sample_rate(model)
        target_sample_rate = int(stage_cfg.get("sample_rate", self.sample_rate or model_sample_rate))

        items: List[AudioItem] = []
        for target in generation_targets:
            item = self._generate_item(
                model=model,
                prompt_item=target["prompt_item"],
                prompt_id=target["prompt_id"],
                instance=target["instance"],
                instance_index=target["instance_index"],
                num_candidates=num_candidates,
                model_sample_rate=model_sample_rate,
                target_sample_rate=target_sample_rate,
                seconds_total=seconds_total,
                negative_prompt=negative_prompt,
                ctx=ctx,
            )
            items.append(item)

        summary = AudioGenerationSummary(
            items=items,
            sample_rate=target_sample_rate,
            default_duration_sec=seconds_total,
            backend=self.backend,
            model=str(stage_cfg.get("model_variant", self.model_variant)),
            metadata={
                "num_candidates": num_candidates,
                "model_sample_rate": model_sample_rate,
                "apply_peak_gain": bool(stage_cfg.get("apply_peak_gain", self.apply_peak_gain)),
            },
        )
        summary_path = save_audio_summary(paths, summary)
        return self._make_stage_result(ctx, summary_path, summary, prompt_count=len(prompt_items))

    def _generate_item(
        self,
        model: Any,
        prompt_item: Dict[str, Any],
        prompt_id: int,
        instance: Optional[Dict[str, Any]],
        instance_index: Optional[int],
        num_candidates: int,
        model_sample_rate: int,
        target_sample_rate: int,
        seconds_total: float,
        negative_prompt: str,
        ctx: StageContext,
    ) -> AudioItem:
        import torch
        import torchaudio

        paths = ctx.paths
        stage_cfg = ctx.stage_config(self.name)
        label = prompt_item["label"]
        grounding_label = prompt_item["grounding_label"]
        source_type = prompt_item["source_type"]
        peak_db = float(prompt_item["peak_db"])
        diffusion_prompt = prompt_item["diffusion_prompt"]
        apply_peak_gain = bool(stage_cfg.get("apply_peak_gain", self.apply_peak_gain))
        instance_id = clean_text(instance.get("instance_id")) if instance is not None else ""

        slug = slug_text(grounding_label, default="audio")
        if instance_id:
            instance_slug = slug_text(instance_id, default=f"instance_{instance_index or 0}")
            file_stem = f"{prompt_id}_{slug}_{instance_slug}"
            audio_id = f"audio_{prompt_id:02d}_{slug}_{instance_slug}"
        else:
            file_stem = f"{prompt_id}_{slug}"
            audio_id = f"audio_{prompt_id:02d}_{slug}"
        candidates: List[AudioCandidate] = []

        for candidate_index in range(num_candidates):
            with torch.no_grad():
                audio = model.generate_t2a(diffusion_prompt)

            audio = self._prepare_audio_tensor(audio, torch)
            if apply_peak_gain:
                audio = audio * (10 ** (peak_db / 20.0))
            audio = audio.clamp(-1.0, 1.0)

            if target_sample_rate != model_sample_rate:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=model_sample_rate,
                    new_freq=target_sample_rate,
                )

            audio_path = paths.audio_assets / f"{file_stem}_{candidate_index}.wav"
            torchaudio.save(audio_path, audio.cpu(), sample_rate=target_sample_rate)
            duration_sec = float(audio.shape[-1]) / float(target_sample_rate)

            candidates.append(
                AudioCandidate(
                    path=ref(audio_path, ctx.scene_root, role="audio", media_type="audio/wav"),
                    candidate_index=candidate_index,
                    sample_rate=target_sample_rate,
                    duration_sec=duration_sec,
                    metadata={"model_sample_rate": model_sample_rate},
                )
            )

        primary = candidates[0].path if candidates else None
        return AudioItem(
            audio_id=audio_id,
            class_label=label,
            grounding_label=grounding_label,
            diffusion_prompt=diffusion_prompt,
            source_type=source_type,
            peak_db=peak_db,
            primary=primary,
            candidates=candidates,
            prompt_id=prompt_id,
            instance_id=instance_id or None,
            negative_prompt=negative_prompt,
            backend=self.backend,
            model=str(stage_cfg.get("model_variant", self.model_variant)),
            metadata={
                "duration_requested_sec": seconds_total,
                "apply_peak_gain": apply_peak_gain,
                "source": prompt_item.get("source", "understanding"),
                "instance_index": instance_index,
            },
        )

    def _get_model(
        self,
        ctx: StageContext,
        stage_cfg: Dict[str, Any],
        seconds_total: float,
    ) -> Any:
        device = str(stage_cfg.get("device", ctx.device))
        model_variant = str(stage_cfg.get("model_variant", self.model_variant))
        model_path = str(stage_cfg.get("model_path", self.model_path))
        steps = int(stage_cfg.get("steps", self.steps))
        guidance_scale = float(stage_cfg.get("guidance_scale", self.guidance_scale))
        full_precision = bool(stage_cfg.get("full_precision", self.full_precision))
        inference_mode = str(stage_cfg.get("inference_mode", self.inference_mode))
        model_name = str(stage_cfg.get("model_name", self.model_name))
        model_key = (
            model_name,
            model_variant,
            steps,
            guidance_scale,
            seconds_total,
            full_precision,
            inference_mode,
            model_path,
            device,
        )
        if self._model is not None and self._model_key == model_key:
            return self._model

        from sonoworld.models.audio_diffusion.mmaudio import MMAudioDiffusion

        model = MMAudioDiffusion(
            model_name=model_name,
            variant=model_variant,
            steps=steps,
            guidance_scale=guidance_scale,
            seconds_total=seconds_total,
            full_precision=full_precision,
            inference_mode=inference_mode,
            model_path=model_path,
            device=device,
        )
        if hasattr(model, "eval"):
            model = model.eval()
        if hasattr(model, "to"):
            model = model.to(device)

        self._model = model
        self._model_key = model_key
        return model

    def _model_sample_rate(self, model: Any) -> int:
        if hasattr(model, "get_config"):
            cfg = model.get_config()
            sample_rate = getattr(cfg, "sample_rate", None)
            if sample_rate is not None:
                return int(sample_rate)
        return int(self.sample_rate or 44100)

    def _prepare_audio_tensor(self, audio: Any, torch: Any) -> Any:
        if not hasattr(audio, "detach"):
            audio = torch.as_tensor(audio)

        audio = audio.detach().cpu().to(torch.float32)
        while audio.ndim > 2:
            audio = audio[0]

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            if audio.shape[0] > 8 and audio.shape[1] <= 8:
                audio = audio.transpose(0, 1)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
        else:
            audio = audio.reshape(1, -1)

        return audio.clamp(-1.0, 1.0)

    def _understanding_audio_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        for obj in data.get("objects", []):
            if not isinstance(obj, dict) or not bool(obj.get("should_generate_audio", True)):
                continue

            grounding_label = clean_text(obj.get("grounding_label")) or clean_text(obj.get("label"))
            if not grounding_label:
                continue

            source_type = normalize_audio_source_type(obj.get("source_type"), grounding_label)
            items.append(
                {
                    "label": clean_text(obj.get("label")) or grounding_label,
                    "grounding_label": grounding_label,
                    "diffusion_prompt": clean_text(obj.get("diffusion_prompt"))
                    or f"{grounding_label} sound",
                    "source_type": source_type,
                    "peak_db": normalize_peak_db(obj.get("peak_db"), source_type),
                    "source": "objects",
                }
            )

        for snd in data.get("global_sounds", []):
            if not isinstance(snd, dict) or not bool(snd.get("should_generate_audio", True)):
                continue

            source_type = "background"
            items.append(
                {
                    "label": clean_text(snd.get("label")) or "global",
                    "grounding_label": "global",
                    "diffusion_prompt": clean_text(snd.get("diffusion_prompt"))
                    or "quiet scene ambience",
                    "source_type": source_type,
                    "peak_db": normalize_peak_db(snd.get("peak_db"), source_type),
                    "source": "global_sounds",
                }
            )

        legacy_items = data.get("items")
        if isinstance(legacy_items, list):
            for item in legacy_items:
                if not isinstance(item, dict):
                    continue
                grounding_label = clean_text(item.get("grounding_label")) or "global"
                source_type = normalize_audio_source_type(item.get("source_type"), grounding_label)
                items.append(
                    {
                        "label": clean_text(item.get("label")) or grounding_label,
                        "grounding_label": grounding_label,
                        "diffusion_prompt": clean_text(item.get("diffusion_prompt"))
                        or f"{grounding_label} sound",
                        "source_type": source_type,
                        "peak_db": normalize_peak_db(item.get("peak_db"), source_type),
                        "source": "items",
                    }
                )

        return items

    def _segmentation_instances_by_label(
        self,
        segmentation: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        instances = segmentation.get("instances", [])
        if not isinstance(instances, list):
            return grouped

        for instance in instances:
            if not isinstance(instance, dict):
                continue
            label = clean_text(instance.get("class_label"))
            instance_id = clean_text(instance.get("instance_id"))
            if not label or not instance_id:
                continue
            grouped.setdefault(label, []).append(instance)
        return grouped

    def _generation_targets(
        self,
        prompt_items: List[Dict[str, Any]],
        instances_by_label: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        for prompt_id, prompt_item in enumerate(prompt_items):
            grounding_label = clean_text(prompt_item.get("grounding_label"))
            source_type = normalize_audio_source_type(
                prompt_item.get("source_type"),
                grounding_label,
            )

            if grounding_label.lower() == "global" or source_type == "background":
                targets.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_item": prompt_item,
                        "instance": None,
                        "instance_index": None,
                    }
                )
                continue

            for instance_index, instance in enumerate(
                instances_by_label.get(grounding_label, [])
            ):
                targets.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_item": prompt_item,
                        "instance": instance,
                        "instance_index": instance_index,
                    }
                )
        return targets

    def _make_stage_result(
        self,
        ctx: StageContext,
        summary_path: Path,
        summary: AudioGenerationSummary,
        prompt_count: int,
    ) -> StageResult:
        outputs: Dict[str, FileRef] = {
            "summary": ref(
                summary_path,
                ctx.scene_root,
                role="audio_generation_summary",
                media_type="application/json",
            ),
        }
        for item in summary.items:
            if item.primary is not None:
                outputs[item.audio_id] = item.primary

        return StageResult(
            status="done",
            inputs={
                "understanding": ref(
                    ctx.paths.understanding_metadata,
                    ctx.scene_root,
                    role="understanding_metadata",
                    media_type="application/json",
                ),
                "segmentation": ref(
                    ctx.paths.segmentation_summary,
                    ctx.scene_root,
                    role="segmentation_summary",
                    media_type="application/json",
                ),
            },
            outputs=outputs,
            message="MMAudio generation complete.",
            metadata={
                "backend": self.backend,
                "model": summary.model,
                "num_prompts": prompt_count,
                "num_audio_items": len(summary.items),
                "sample_rate": summary.sample_rate,
            },
        )
