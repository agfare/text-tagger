"""Zero-shot classification of text chunks against the tag taxonomy."""

from __future__ import annotations

from typing import Any

from transformers import pipeline

from .tags import HYPOTHESIS_TEMPLATES, TAXONOMY, ChunkTags

# facebook/bart-large-mnli is the de-facto standard for zero-shot classification
# via the HuggingFace pipeline API.  It is ~1.6 GB but runs on CPU.
DEFAULT_MODEL = "facebook/bart-large-mnli"

# Tags that are multi-label (top-1 per category is still used for the primary
# tag, but scores for all candidates are stored).
_MULTI_LABEL_CATEGORIES = {"system", "location"}


class Tagger:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1) -> None:
        """
        Args:
            model_name: HuggingFace model ID for zero-shot classification.
            device: -1 for CPU, 0+ for GPU index.
        """
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )

    def tag_chunk(self, text: str, chunk_id: str, source_file: str) -> ChunkTags:
        chunk = ChunkTags(
            chunk_id=chunk_id,
            source_file=source_file,
            text_preview=text[:200],
        )

        for category, candidates in TAXONOMY.items():
            template = HYPOTHESIS_TEMPLATES[category]
            multi_label = category in _MULTI_LABEL_CATEGORIES

            result: dict[str, Any] = self._pipe(
                text,
                candidate_labels=candidates,
                hypothesis_template=template,
                multi_label=multi_label,
            )

            scores = dict(zip(result["labels"], result["scores"]))
            chunk.scores[category] = scores
            # Primary tag = highest-scoring candidate
            setattr(chunk, category, result["labels"][0])

        return chunk

    def tag_chunks(self, chunks: list[str], source_file: str) -> list[ChunkTags]:
        return [
            self.tag_chunk(text, chunk_id=f"{source_file}:{i}", source_file=source_file)
            for i, text in enumerate(chunks)
        ]
