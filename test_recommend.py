"""Unit tests for the recommendation engine (no model required)."""

import json
import tempfile
from pathlib import Path

import pytest

from recommend.engine import RecommendationEngine, UserState
from tagger.tags import ChunkTags


def _make_tagged_dir(chunks: list[ChunkTags]) -> Path:
    tmp = Path(tempfile.mkdtemp())
    records = [c.to_dict() for c in chunks]
    (tmp / "test.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp


def _chunk(chunk_id, chunk_type, system, complexity="general", audience="general") -> ChunkTags:
    return ChunkTags(
        chunk_id=chunk_id,
        source_file="test.docx",
        text_preview="...",
        chunk_type=chunk_type,
        system=system,
        complexity=complexity,
        audience=audience,
    )


CHUNKS = [
    _chunk("a", "overview", "cardiovascular"),
    _chunk("b", "diagnosis", "cardiovascular"),
    _chunk("c", "treatment", "cardiovascular"),
    _chunk("d", "overview", "neurological"),
    _chunk("e", "overview", "cardiovascular", complexity="advanced", audience="doctor"),
]


def test_cold_start_recommends_overview():
    engine = RecommendationEngine(_make_tagged_dir(CHUNKS))
    state = engine.cold_start(last_opened_system="cardiovascular")
    recs = engine.recommend(state)
    assert recs[0].chunk_type == "overview"


def test_system_filter_ranks_matching_system_first():
    engine = RecommendationEngine(_make_tagged_dir(CHUNKS))
    state = engine.cold_start(last_opened_system="cardiovascular")
    recs = engine.recommend(state)
    systems = [r.system for r in recs]
    assert systems[0] == "cardiovascular"


def test_advance_state_progresses_chunk_type():
    engine = RecommendationEngine(_make_tagged_dir(CHUNKS))
    state = engine.cold_start("cardiovascular")
    state = engine.advance_state(state, "a")  # viewed overview
    assert state.chunk_type == "diagnosis"


def test_seen_chunks_excluded():
    engine = RecommendationEngine(_make_tagged_dir(CHUNKS))
    state = UserState(system="cardiovascular", chunk_type="overview")
    state.history = ["a", "d", "e"]
    recs = engine.recommend(state)
    ids = [r.chunk_id for r in recs]
    assert "a" not in ids
    assert "d" not in ids


def test_no_crash_on_empty_history():
    engine = RecommendationEngine(_make_tagged_dir(CHUNKS))
    state = engine.cold_start()
    recs = engine.recommend(state, top_k=10)
    assert isinstance(recs, list)
