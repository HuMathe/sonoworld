from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .common import SerializableDataclass, FileRef


@dataclass
class SceneInput(SerializableDataclass):
    """User-provided inputs for a scene."""

    input_image: Optional[FileRef] = None
    panorama: Optional[FileRef] = None
    known_sources: Optional[FileRef] = None
    mask: Optional[FileRef] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneDescription(SerializableDataclass):
    """Static scene-level description."""

    scene_id: str
    input: SceneInput

    title: Optional[str] = None
    description: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
