from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sonoworld.core.artifacts import ref, save_understanding
from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.schemas.common import FileRef, read_json, write_json
from sonoworld.schemas.understanding import (
    GlobalSound,
    SceneUnderstanding,
    SoundObject,
    normalize_known_sources,
)
from sonoworld.utils.audio_source_utils import normalize_audio_source_type, normalize_peak_db
from sonoworld.utils.text_utils import clean_text


DEFAULT_PROMPT_PATH = "configs/prompts/default_uncond.txt"
DEFAULT_MODEL = "gpt-5"


class GPTUnderstandingStage(Stage):
    """OpenAI GPT-backed panorama understanding stage.

    Third-party imports are kept inside the API call so importing this module
    stays cheap in pipelines that do not use GPT.
    """

    name = "understanding"
    backend = "gpt"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_path: str = DEFAULT_PROMPT_PATH,
        max_retries: int = 3,
        grounding_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.prompt_path = prompt_path
        self.max_retries = max(1, int(max_retries))
        self.grounding_threshold = float(grounding_threshold)

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        stage_cfg = ctx.stage_config(self.name)

        model = str(stage_cfg.get("model", self.model))
        prompt_path = self._resolve_prompt_path(
            stage_cfg.get("prompt_path", self.prompt_path),
            scene_root=scene_root,
        )
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {prompt_path}")

        known_sources, known_sources_path = self._load_known_sources(ctx, stage_cfg)
        request_prompt = self._compose_conditioned_prompt(prompt, known_sources)
        paths.understanding.mkdir(parents=True, exist_ok=True)
        request_prompt_path = paths.understanding / "request_prompt.txt"
        request_prompt_path.write_text(request_prompt + "\n", encoding="utf-8")

        panorama = paths.find_panorama()
        response_text = self._request_json(
            model=model,
            prompt=request_prompt,
            image_path=panorama,
            max_retries=int(stage_cfg.get("max_retries", self.max_retries)),
            known_sources=known_sources,
        )
        raw_items = self._parse_response_items(response_text)
        missing_known_sources = self._missing_known_sources(raw_items, known_sources)
        if missing_known_sources:
            raise ValueError(
                "GPT response omitted required major sources after validation: "
                + ", ".join(missing_known_sources)
            )
        understanding = self._to_understanding(
            raw_items,
            model=model,
            prompt_path=prompt_path,
            panorama=panorama,
            response_text=response_text,
            known_sources=known_sources,
            request_prompt_path=request_prompt_path,
        )

        summary_path = save_understanding(paths, understanding)

        grounding_threshold = float(
            stage_cfg.get("grounding_threshold", self.grounding_threshold)
        )
        if grounding_threshold != 0.7:
            write_json(paths.grounding_input, understanding.to_grounding_input(grounding_threshold))

        legacy_path = paths.understanding / "understanding_legacy.json"

        inputs = {
            "panorama": ref(panorama, scene_root, role="panorama", media_type="image"),
            "base_prompt": FileRef.from_path(
                prompt_path,
                scene_root=scene_root,
                role="understanding_base_prompt",
                media_type="text/plain",
            ),
            "request_prompt": ref(
                request_prompt_path,
                scene_root,
                role="understanding_request_prompt",
                media_type="text/plain",
            ),
        }
        if known_sources_path is not None:
            inputs["known_sources"] = ref(
                known_sources_path,
                scene_root,
                role="known_sources",
                media_type="application/json",
            )

        return StageResult(
            status="done",
            inputs=inputs,
            outputs={
                "summary": ref(
                    summary_path,
                    scene_root,
                    role="understanding_metadata",
                    media_type="application/json",
                ),
                "legacy": ref(
                    legacy_path,
                    scene_root,
                    role="understanding_legacy",
                    media_type="application/json",
                ),
                "grounding_input": ref(
                    paths.grounding_input,
                    scene_root,
                    role="grounding_input",
                    media_type="application/json",
                ),
            },
            message="GPT understanding complete.",
            metadata={
                "backend": self.backend,
                "model": model,
                "num_items": len(raw_items),
                "prompt_path": str(prompt_path),
                "request_prompt_path": str(request_prompt_path),
                "known_sources": known_sources,
            },
        )

    def _request_json(
        self,
        model: str,
        prompt: str,
        image_path: Path,
        max_retries: int,
        known_sources: Optional[List[str]] = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "GPTUnderstandingStage requires the OpenAI Python package. "
                "Install `openai` and set OPENAI_API_KEY before running this stage."
            ) from exc

        client = OpenAI()
        image_url = self._image_data_url(image_path)
        last_response = ""
        last_validation_error = ""
        max_attempts = max(1, int(max_retries))

        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if last_validation_error:
                attempt_prompt += (
                    "\n\nCORRECTION REQUIRED\n"
                    "The previous response was rejected because: "
                    f"{last_validation_error}. Return a complete corrected JSON array."
                )
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": attempt_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
            )
            last_response = (result.choices[0].message.content or "").strip()
            try:
                items = self._parse_response_items(last_response)
                missing = self._missing_known_sources(items, known_sources or [])
                if missing:
                    raise ValueError(
                        "missing required major sources: " + ", ".join(missing)
                    )
                return last_response
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_validation_error = str(exc)
                if attempt == max_attempts:
                    break

        raise ValueError(
            "GPT did not return valid understanding JSON after "
            f"{max_attempts} attempt(s). Validation error: {last_validation_error}. "
            f"Last response: {last_response[:500]}"
        )

    def _to_understanding(
        self,
        raw_items: List[Dict[str, Any]],
        model: str,
        prompt_path: Path,
        panorama: Path,
        response_text: str,
        known_sources: Optional[List[str]] = None,
        request_prompt_path: Optional[Path] = None,
    ) -> SceneUnderstanding:
        known_sources = known_sources or []
        objects: List[SoundObject] = []
        global_sounds: List[GlobalSound] = []

        for index, item in enumerate(raw_items):
            grounding_label = (
                clean_text(item.get("grounding_label"))
                or clean_text(item.get("label"))
                or "global"
            )
            source_type = normalize_audio_source_type(item.get("source_type"), grounding_label)
            diffusion_prompt = clean_text(item.get("diffusion_prompt"))
            peak_db = normalize_peak_db(item.get("peak_db"), source_type)

            if grounding_label.lower() == "global" or source_type == "background":
                global_sounds.append(
                    GlobalSound(
                        label="global",
                        diffusion_prompt=diffusion_prompt or "quiet scene ambience",
                        source_type="background",
                        peak_db=peak_db,
                        metadata={"raw_index": index},
                    )
                )
                continue

            label = clean_text(item.get("label")) or grounding_label
            matched_known_sources = [
                source
                for source in known_sources
                if self._item_covers_known_source(item, source)
            ]
            objects.append(
                SoundObject(
                    label=label,
                    grounding_label=grounding_label,
                    diffusion_prompt=diffusion_prompt or f"{grounding_label} sound",
                    source_type=source_type,
                    peak_db=peak_db,
                    metadata={
                        "raw_index": index,
                        "known_sources": matched_known_sources,
                        "known_source_conditioned": bool(matched_known_sources),
                    },
                )
            )

        if not global_sounds:
            global_sounds.append(
                GlobalSound(
                    diffusion_prompt="quiet scene ambience",
                    peak_db=-30.0,
                    metadata={"created_by": self.backend},
                )
            )

        return SceneUnderstanding(
            scene_description="Sound sources inferred from the panorama.",
            objects=objects,
            global_sounds=global_sounds,
            backend=self.backend,
            model=model,
            metadata={
                "prompt_path": str(prompt_path),
                "request_prompt_path": str(request_prompt_path) if request_prompt_path else None,
                "panorama": str(panorama),
                "raw_response": response_text,
                "known_sources": known_sources,
                "known_source_conditioned": bool(known_sources),
            },
        )

    def _load_known_sources(
        self,
        ctx: StageContext,
        stage_cfg: Dict[str, Any],
    ) -> Tuple[List[str], Optional[Path]]:
        labels: List[str] = []
        inline = stage_cfg.get("known_sources")
        if inline is not None:
            labels.extend(normalize_known_sources(inline))

        configured_path = stage_cfg.get("known_sources_path")
        if configured_path:
            path = self._resolve_known_sources_path(configured_path, ctx.scene_root)
            labels.extend(normalize_known_sources(read_json(path)))
            loaded_path = path
        else:
            loaded_path = None
            for candidate in [
                ctx.paths.known_sources,
                ctx.scene_root / "known_sources.json",
            ]:
                if candidate.is_file():
                    labels.extend(normalize_known_sources(read_json(candidate)))
                    loaded_path = candidate
                    break

        deduplicated: List[str] = []
        seen = set()
        for label in labels:
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(label)
        return deduplicated, loaded_path

    def _compose_conditioned_prompt(
        self,
        base_prompt: str,
        known_sources: List[str],
    ) -> str:
        if not known_sources:
            return base_prompt
        payload = json.dumps(known_sources, ensure_ascii=False)
        return (
            base_prompt.rstrip()
            + "\n\nMAJOR SOURCES (HARD REQUIREMENT)\n"
            + f"Major sources JSON: {payload}\n"
            + "Treat these labels as the authoritative major sound sources in the scene, "
            + "even when a source is visually ambiguous. List every major source as a "
            + "prominent non-global item before any optional source. Give it a foreground/strong "
            + "or mid/medium peak_db appropriate to the scene; never demote it to the global "
            + "background bed. For EVERY major source, output at least one non-global "
            + "item. Its diffusion_prompt must explicitly contain the meaningful words from "
            + "the major-source label and describe that sound. Its grounding_label must name "
            + "a visible object or region that directly supports the sound and must share at "
            + "least one meaningful word with the major-source label. If the sounding agent "
            + "is hidden, ground it on its visible host (for example, birds chirping in leaves "
            + "may use grounding_label leaves). Do not merge, omit, or replace a major source. "
            + "You may add other plausible visible sources under the original rules."
        )

    def _missing_known_sources(
        self,
        items: List[Dict[str, Any]],
        known_sources: List[str],
    ) -> List[str]:
        return [
            source
            for source in known_sources
            if not any(self._item_covers_known_source(item, source) for item in items)
        ]

    def _item_covers_known_source(
        self,
        item: Dict[str, Any],
        known_source: str,
    ) -> bool:
        grounding_label = clean_text(item.get("grounding_label"))
        source_type = normalize_audio_source_type(item.get("source_type"), grounding_label)
        if not grounding_label or grounding_label.lower() == "global" or source_type == "background":
            return False
        source_tokens = self._meaningful_tokens(known_source)
        prompt_tokens = self._meaningful_tokens(item.get("diffusion_prompt"))
        grounding_tokens = self._meaningful_tokens(grounding_label)
        return bool(source_tokens) and source_tokens.issubset(prompt_tokens) and bool(
            source_tokens & grounding_tokens
        )

    def _meaningful_tokens(self, value: Any) -> set[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "for",
            "from",
            "in",
            "of",
            "on",
            "the",
            "to",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", clean_text(value).lower())
            if token not in stop_words
        }

    def _parse_response_items(self, response_text: str) -> List[Dict[str, Any]]:
        text = self._strip_code_fence(response_text)
        data = json.loads(text)

        if isinstance(data, dict):
            if isinstance(data.get("items"), list):
                data = data["items"]
            elif isinstance(data.get("sounds"), list):
                data = data["sounds"]
            else:
                data = [data]

        if not isinstance(data, list):
            raise TypeError("GPT response must be a JSON array or an object containing one.")

        items: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Every GPT understanding item must be a JSON object.")
            items.append(item)

        if not items:
            raise ValueError("GPT response contained no understanding items.")

        return items

    def _resolve_prompt_path(self, prompt_path: str | Path, scene_root: Path) -> Path:
        path = Path(prompt_path)
        if path.is_absolute():
            return path

        candidates = [
            scene_root / path,
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[-1]

    def _resolve_known_sources_path(
        self,
        known_sources_path: str | Path,
        scene_root: Path,
    ) -> Path:
        path = Path(known_sources_path).expanduser()
        candidates = [path] if path.is_absolute() else [
            scene_root / path,
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Configured known-sources JSON does not exist: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def _image_data_url(self, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
