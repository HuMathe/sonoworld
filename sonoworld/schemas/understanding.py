from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from sonoworld.utils.text_utils import clean_text

from .common import SerializableDataclass


SourceType = Literal["point", "area", "background"]
GroundingLabel = str


def normalize_known_sources(value: Any) -> List[str]:
    """Validate and normalize the public SonoScene360 known-source format."""
    if isinstance(value, dict):
        value = value.get("known_sources")
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError('Known sources must have the form {"known_sources": ["..."]}.')

    labels: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError("Every known source must be a string.")
        label = clean_text(item)
        if not label:
            raise ValueError("Known-source labels cannot be empty.")
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


@dataclass
class SoundObject(SerializableDataclass):
    """A sound-producing object or region described by the understanding module."""

    label: str
    grounding_label: str
    diffusion_prompt: str

    source_type: SourceType = "area"
    peak_db: float = -6.0

    visual_description: Optional[str] = None
    audio_description: Optional[str] = None

    min_instances: int = 0
    max_instances: Optional[int] = None

    should_segment: bool = True
    should_generate_audio: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalSound(SerializableDataclass):
    """A non-localized ambient sound."""

    label: str = "global"
    diffusion_prompt: str = ""

    source_type: SourceType = "background"
    peak_db: float = -12.0

    audio_description: Optional[str] = None
    should_generate_audio: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneUnderstanding(SerializableDataclass):
    """Unified output schema for GPT, LLaVA, or other scene understanding backends."""

    scene_description: str

    objects: List[SoundObject] = field(default_factory=list)
    global_sounds: List[GlobalSound] = field(default_factory=list)

    negative_prompt: str = "Low quality."
    backend: Optional[str] = None
    model: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def grounding_labels(self) -> List[str]:
        labels = []
        for item in self.objects:
            if item.should_segment and item.grounding_label.strip():
                labels.append(item.grounding_label)
        return sorted(set(labels))

    def audio_items(self) -> List[SoundObject | GlobalSound]:
        items: List[SoundObject | GlobalSound] = []
        items.extend([obj for obj in self.objects if obj.should_generate_audio])
        items.extend([snd for snd in self.global_sounds if snd.should_generate_audio])
        return items

    def to_grounding_input(self, default_threshold: float = 0.7) -> Dict[str, float]:
        return {
            label: default_threshold
            for label in self.grounding_labels()
            if label != "global"
        }

    def to_legacy_understanding_list(self) -> List[Dict[str, Any]]:
        """Export a format compatible with the current experimental audio pipeline."""
        items: List[Dict[str, Any]] = []

        for obj in self.objects:
            if not obj.should_generate_audio:
                continue

            items.append({
                "label": obj.label,
                "grounding_label": obj.grounding_label,
                "diffusion_prompt": obj.diffusion_prompt,
                "source_type": obj.source_type,
                "peak_db": obj.peak_db,
                "visual_description": obj.visual_description,
                "audio_description": obj.audio_description,
            })

        for snd in self.global_sounds:
            if not snd.should_generate_audio:
                continue

            items.append({
                "label": snd.label,
                "grounding_label": "global",
                "diffusion_prompt": snd.diffusion_prompt,
                "source_type": snd.source_type,
                "peak_db": snd.peak_db,
                "audio_description": snd.audio_description,
            })

        return items
