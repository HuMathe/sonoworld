from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .common import SerializableDataclass, FileRef


AudioSourceType = Literal["point", "area", "background"]


@dataclass
class AudioCandidate(SerializableDataclass):
    """One generated audio candidate."""

    path: FileRef
    candidate_index: int

    sample_rate: Optional[int] = None
    duration_sec: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioItem(SerializableDataclass):
    """Generated audio group for one semantic sound description."""

    audio_id: str

    class_label: str
    grounding_label: str

    diffusion_prompt: str
    source_type: AudioSourceType = "area"
    peak_db: float = -6.0

    primary: Optional[FileRef] = None
    candidates: List[AudioCandidate] = field(default_factory=list)

    prompt_id: Optional[int] = None
    instance_id: Optional[str] = None
    negative_prompt: Optional[str] = None

    backend: Optional[str] = None
    model: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def candidate_paths(self) -> List[str]:
        return [candidate.path.path for candidate in self.candidates]


@dataclass
class AudioGenerationSummary(SerializableDataclass):
    """Unified output of the audio generation stage."""

    items: List[AudioItem] = field(default_factory=list)

    sample_rate: int = 48000
    default_duration_sec: Optional[float] = None

    backend: Optional[str] = None
    model: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def items_by_grounding_label(self) -> Dict[str, List[AudioItem]]:
        grouped: Dict[str, List[AudioItem]] = {}
        for item in self.items:
            grouped.setdefault(item.grounding_label, []).append(item)
        return grouped

    def global_items(self) -> List[AudioItem]:
        return [
            item for item in self.items
            if item.grounding_label == "global" or item.source_type == "background"
        ]

    def local_items(self) -> List[AudioItem]:
        return [
            item for item in self.items
            if item.grounding_label != "global" and item.source_type != "background"
        ]
