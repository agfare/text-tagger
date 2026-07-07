"""Rule-based recommendation engine driven by chunk tags."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tagger.tags import COLD_START_DEFAULTS, ChunkTags


@dataclass
class UserState:
    """Minimal representation of a user's current context."""
    system: Optional[str] = None          # set from last-opened model
    chunk_type: str = COLD_START_DEFAULTS["chunk_type"]
    complexity: str = COLD_START_DEFAULTS["complexity"]
    audience: str = COLD_START_DEFAULTS["audience"]
    history: list[str] = field(default_factory=list)  # seen chunk_ids


# ── Progression rules ────────────────────────────────────────────────────────

_CHUNK_TYPE_PROGRESSION = ["overview", "diagnosis", "treatment", "faq"]
_COMPLEXITY_PROGRESSION = ["general", "introductory", "intermediate", "advanced"]

def _next_in(seq: list[str], current: str) -> str:
    try:
        idx = seq.index(current)
        return seq[min(idx + 1, len(seq) - 1)]
    except ValueError:
        return seq[0]


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_chunk(chunk: ChunkTags, state: UserState) -> float:
    """Higher is better.  Simple additive match on tag dimensions."""
    score = 0.0

    if chunk.chunk_id in state.history:
        return -1.0  # already seen

    # Must-match: chunk_type (highest weight)
    if chunk.chunk_type == state.chunk_type:
        score += 3.0

    # Should-match: system
    if state.system and chunk.system == state.system:
        score += 2.0

    # Should-match: complexity
    if chunk.complexity == state.complexity:
        score += 1.5

    # Should-match: audience
    if chunk.audience == state.audience:
        score += 1.5

    return score


# ── Engine ────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    def __init__(self, tagged_dir: Path) -> None:
        self._chunks: list[ChunkTags] = []
        self._load_tagged_dir(tagged_dir)

    def _load_tagged_dir(self, tagged_dir: Path) -> None:
        for json_path in sorted(tagged_dir.rglob("*.json")):
            with open(json_path, encoding="utf-8") as fh:
                records = json.load(fh)
            for rec in records:
                tags = rec.get("tags", {})
                ct = ChunkTags(
                    chunk_id=rec["chunk_id"],
                    source_file=rec["source_file"],
                    text_preview=rec.get("text_preview", ""),
                    **{k: v for k, v in tags.items() if v is not None},
                )
                self._chunks.append(ct)

    def recommend(
        self,
        state: UserState,
        top_k: int = 5,
    ) -> list[ChunkTags]:
        """Return up to top_k chunks ranked for the current user state."""
        scored = [
            (chunk, _score_chunk(chunk, state))
            for chunk in self._chunks
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored[:top_k] if score >= 0]

    def advance_state(self, state: UserState, viewed_chunk_id: str) -> UserState:
        """Update state after a user views a chunk (progress complexity/chunk_type)."""
        state.history.append(viewed_chunk_id)

        # Find the viewed chunk to decide progression
        chunk = next((c for c in self._chunks if c.chunk_id == viewed_chunk_id), None)
        if chunk is None:
            return state

        # After viewing an overview, progress to diagnosis
        if chunk.chunk_type in _CHUNK_TYPE_PROGRESSION:
            state.chunk_type = _next_in(_CHUNK_TYPE_PROGRESSION, chunk.chunk_type)

        # After viewing advanced content, keep complexity
        if chunk.complexity in _COMPLEXITY_PROGRESSION:
            state.complexity = _next_in(_COMPLEXITY_PROGRESSION, chunk.complexity)

        return state

    def cold_start(self, last_opened_system: Optional[str] = None) -> UserState:
        state = UserState()
        if last_opened_system:
            state.system = last_opened_system
        return state
