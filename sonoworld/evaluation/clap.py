"""Optional CLAP metrics used by the internal SonoScene360 evaluation."""

from __future__ import annotations

import math
from typing import Any, Sequence


def directional_sound(foa: Any, direction: str) -> Any:
    import numpy as np

    w, y, z, x = foa
    mapping = {
        "left": (w + y) / math.sqrt(2.0),
        "right": (w - y) / math.sqrt(2.0),
        "front": (w + x) / math.sqrt(2.0),
        "behind": (w - x) / math.sqrt(2.0),
        "up": (w + z) / math.sqrt(2.0),
        "down": (w - z) / math.sqrt(2.0),
    }
    if direction not in mapping:
        raise ValueError(f"Unknown direction: {direction}")
    return np.asarray(mapping[direction], dtype=np.float32)


class CLAPMetrics:
    """Lazy LAION-CLAP wrapper; construction may download its default checkpoint."""

    def __init__(self) -> None:
        try:
            import laion_clap
        except ImportError as exc:
            raise ImportError(
                "CLAP evaluation requires `laion-clap`; install it and rerun with --with-clap."
            ) from exc
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()

    @staticmethod
    def _cosine(first: Any, second: Any) -> float:
        import numpy as np

        first = np.asarray(first, dtype=np.float64).reshape(-1)
        second = np.asarray(second, dtype=np.float64).reshape(-1)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denominator) if denominator > 0 else float("nan")

    def pair_metrics(
        self,
        estimated: Any,
        ground_truth: Any,
        annotations: Sequence[dict[str, str]],
    ) -> dict[str, float]:
        values = []
        for annotation in annotations:
            direction = str(annotation.get("direction", ""))
            text = str(annotation.get("text_label", "")).strip()
            if not text or direction not in {"left", "right", "front", "behind", "up", "down"}:
                continue
            estimated_embedding = self.model.get_audio_embedding_from_data(
                x=directional_sound(estimated, direction)[None], use_tensor=False
            )
            gt_embedding = self.model.get_audio_embedding_from_data(
                x=directional_sound(ground_truth, direction)[None], use_tensor=False
            )
            text_embedding = self.model.get_text_embedding([text], use_tensor=False)
            audio_similarity = self._cosine(estimated_embedding, gt_embedding)
            text_similarity = self._cosine(estimated_embedding, text_embedding)
            gt_text_similarity = self._cosine(gt_embedding, text_embedding)
            values.append((audio_similarity, text_similarity, text_similarity / gt_text_similarity if gt_text_similarity else 0.0))
        if not values:
            return {"clap_audio_similarity": float("nan"), "clap_text_similarity": float("nan"), "clap_text_ratio": float("nan")}
        import numpy as np

        scores = np.asarray(values, dtype=np.float64)
        return {
            "clap_audio_similarity": float(np.mean(scores[:, 0])),
            "clap_text_similarity": float(np.mean(scores[:, 1])),
            "clap_text_ratio": float(np.mean(scores[:, 2])),
        }

    def qa_accuracy(self, estimated: Any, annotations: Sequence[dict[str, str]]) -> float:
        grouped: dict[str, set[str]] = {}
        for annotation in annotations:
            label = str(annotation.get("text_label", "")).strip()
            direction = str(annotation.get("direction", ""))
            if label and direction in {"left", "right", "front", "behind"}:
                grouped.setdefault(label, set()).add(direction)
        valid = {label: directions for label, directions in grouped.items() if 0 < len(directions) < 4}
        if not valid:
            return 0.0
        directions = ("left", "right", "front", "behind")
        audio_embeddings = {
            direction: self.model.get_audio_embedding_from_data(
                x=directional_sound(estimated, direction)[None], use_tensor=False
            )
            for direction in directions
        }
        text_embeddings = self.model.get_text_embedding(list(valid), use_tensor=False)
        correct = 0
        for label_index, (_, true_directions) in enumerate(valid.items()):
            scores = {
                direction: self._cosine(audio_embeddings[direction], text_embeddings[label_index])
                for direction in directions
            }
            true_scores = [scores[direction] for direction in true_directions]
            false_scores = [scores[direction] for direction in directions if direction not in true_directions]
            if min(true_scores) > max(false_scores):
                correct += 1
        return correct / len(valid)
