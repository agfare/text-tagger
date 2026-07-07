"""End-to-end tagging pipeline: docx → tagged JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .classifier import DEFAULT_MODEL, Tagger
from .docx_reader import read_docx_chunks
from .tags import ChunkTags


def tag_file(
    docx_path: Path,
    output_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    device: int = -1,
) -> list[ChunkTags]:
    """Tag all chunks in a single docx file and optionally write JSON.

    Returns the list of ChunkTags (also written to output_path when given).
    """
    tagger = Tagger(model_name=model_name, device=device)
    chunks = read_docx_chunks(docx_path)
    tagged = tagger.tag_chunks(chunks, source_file=docx_path.name)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([t.to_dict() for t in tagged], fh, indent=2, ensure_ascii=False)

    return tagged


def tag_directory(
    input_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    device: int = -1,
    glob: str = "**/*.docx",
) -> dict[str, list[ChunkTags]]:
    """Tag every .docx under input_dir, writing one JSON per file to output_dir."""
    tagger = Tagger(model_name=model_name, device=device)
    results: dict[str, list[ChunkTags]] = {}

    docx_files = sorted(input_dir.glob(glob))
    if not docx_files:
        raise FileNotFoundError(f"No .docx files found under {input_dir}")

    for docx_path in docx_files:
        rel = docx_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = read_docx_chunks(docx_path)
        tagged = tagger.tag_chunks(chunks, source_file=str(rel))
        results[str(rel)] = tagged

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([t.to_dict() for t in tagged], fh, indent=2, ensure_ascii=False)

        print(f"  Tagged {len(tagged)} chunks  →  {out_path}")

    return results
