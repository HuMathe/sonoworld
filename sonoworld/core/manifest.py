from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Literal
import traceback as traceback_lib

from sonoworld.schemas.common import (
    SerializableDataclass,
    FileRef,
    ErrorInfo,
    read_json,
    write_json,
)


StageStatus = Literal[
    "pending",
    "running",
    "waiting",
    "done",
    "failed",
    "skipped",
]


@dataclass
class StageRecord(SerializableDataclass):
    """Persistent status record for one generation stage."""

    name: str
    status: StageStatus = "pending"

    backend: Optional[str] = None
    message: Optional[str] = None

    inputs: Dict[str, FileRef] = field(default_factory=dict)
    outputs: Dict[str, FileRef] = field(default_factory=dict)

    error: Optional[ErrorInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneManifest(SerializableDataclass):
    """Persistent state for a scene generation process."""

    scene_id: str
    root: str

    stages: Dict[str, StageRecord] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, scene_root: Path, scene_id: Optional[str] = None) -> "SceneManifest":
        scene_root = Path(scene_root)
        return cls(
            scene_id=scene_id or scene_root.name,
            root=str(scene_root),
            stages={},
        )

    @classmethod
    def load(cls, path: Path) -> "SceneManifest":
        data = read_json(path)

        stages = {}
        for name, record in data.get("stages", {}).items():
            inputs = {
                key: FileRef(**value)
                for key, value in record.get("inputs", {}).items()
            }
            outputs = {
                key: FileRef(**value)
                for key, value in record.get("outputs", {}).items()
            }

            error_data = record.get("error")
            error = ErrorInfo(**error_data) if error_data else None

            stages[name] = StageRecord(
                name=record["name"],
                status=record.get("status", "pending"),
                backend=record.get("backend"),
                message=record.get("message"),
                inputs=inputs,
                outputs=outputs,
                error=error,
                metadata=record.get("metadata", {}),
            )

        return cls(
            scene_id=data["scene_id"],
            root=data["root"],
            stages=stages,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    def get_stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def mark_pending(self, name: str, backend: Optional[str] = None) -> None:
        record = self.get_stage(name)
        record.status = "pending"
        record.backend = backend
        record.message = None
        record.error = None

    def mark_running(self, name: str, backend: Optional[str] = None) -> None:
        record = self.get_stage(name)
        record.status = "running"
        record.backend = backend
        record.message = None
        record.error = None

    def mark_waiting(
        self,
        name: str,
        backend: Optional[str] = None,
        message: Optional[str] = None,
        inputs: Optional[Dict[str, FileRef]] = None,
        outputs: Optional[Dict[str, FileRef]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self.get_stage(name)
        record.status = "waiting"
        record.backend = backend
        record.message = message
        record.error = None

        if inputs is not None:
            record.inputs = inputs
        if outputs is not None:
            record.outputs = outputs
        if metadata is not None:
            record.metadata.update(metadata)

    def mark_done(
        self,
        name: str,
        backend: Optional[str] = None,
        message: Optional[str] = None,
        inputs: Optional[Dict[str, FileRef]] = None,
        outputs: Optional[Dict[str, FileRef]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self.get_stage(name)
        record.status = "done"
        record.backend = backend
        record.message = message
        record.error = None

        if inputs is not None:
            record.inputs = inputs
        if outputs is not None:
            record.outputs = outputs
        if metadata is not None:
            record.metadata.update(metadata)

    def mark_failed(
        self,
        name: str,
        exc: BaseException,
        backend: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        record = self.get_stage(name)
        record.status = "failed"
        record.backend = backend
        record.message = message or str(exc)
        record.error = ErrorInfo(
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback_lib.format_exc(),
        )

    def mark_skipped(
        self,
        name: str,
        backend: Optional[str] = None,
        message: Optional[str] = None,
        inputs: Optional[Dict[str, FileRef]] = None,
        outputs: Optional[Dict[str, FileRef]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self.get_stage(name)
        record.status = "skipped"
        record.backend = backend
        record.message = message
        record.error = None
        if inputs is not None:
            record.inputs = inputs
        if outputs is not None:
            record.outputs = outputs
        if metadata is not None:
            record.metadata.update(metadata)

    def is_done(self, name: str) -> bool:
        return self.stages.get(name) is not None and self.stages[name].status == "done"

    def is_waiting(self, name: str) -> bool:
        return self.stages.get(name) is not None and self.stages[name].status == "waiting"

    def first_waiting_stage(self) -> Optional[StageRecord]:
        for record in self.stages.values():
            if record.status == "waiting":
                return record
        return None
