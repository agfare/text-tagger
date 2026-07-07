"""Extract and chunk text from .docx files."""

import re
from pathlib import Path
from typing import Iterator

from docx import Document


# Paragraphs shorter than this (chars) are treated as headings / noise.
MIN_PARAGRAPH_CHARS = 60
# Merge consecutive paragraphs until a chunk reaches this size.
TARGET_CHUNK_CHARS = 800


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_chunks(docx_path: Path, target_chars: int = TARGET_CHUNK_CHARS) -> Iterator[str]:
    """Yield text chunks from a docx file.

    Short paragraphs (headings, captions) are merged into the following body
    text so that each chunk carries enough context for classification.
    """
    doc = Document(str(docx_path))
    buffer: list[str] = []
    buffer_len = 0

    for para in doc.paragraphs:
        text = _clean(para.text)
        if not text:
            continue
        buffer.append(text)
        buffer_len += len(text)

        if buffer_len >= target_chars:
            yield " ".join(buffer)
            buffer = []
            buffer_len = 0

    if buffer:
        yield " ".join(buffer)


def read_docx_chunks(docx_path: Path) -> list[str]:
    return list(iter_chunks(docx_path))
