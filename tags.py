"""Tag taxonomy for the DLW recommendation system."""

from dataclasses import dataclass, field
from typing import Optional

TAXONOMY: dict[str, list[str]] = {
    "type": ["model", "article"],
    "domain": ["pathology", "part"],
    "system": [
        "cardiovascular",
        "neurological",
        "respiratory",
        "digestive",
        "musculoskeletal",
        "endocrine",
        "immune",
        "reproductive",
        "urinary",
        "integumentary",
    ],
    "location": [
        "brain",
        "back",
        "lungs",
        "heart",
        "liver",
        "kidney",
        "spine",
        "chest",
        "abdomen",
        "neck",
        "shoulder",
        "knee",
        "hip",
        "pelvis",
        "head",
    ],
    "chunk_type": ["overview", "diagnosis", "treatment", "faq"],
    "complexity": ["general", "introductory", "intermediate", "advanced"],
    "audience": ["student", "doctor", "patient", "general"],
}

# Natural-language hypothesis templates fed to the zero-shot classifier.
# The {label} placeholder is replaced with each candidate tag value.
HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "type": "This text is a {}.",
    "domain": "This text is about {}.",
    "system": "This text is about the {} system.",
    "location": "This text is about the {}.",
    "chunk_type": "This text is an {} section.",
    "complexity": "This text is written at a {} level.",
    "audience": "This text is intended for a {}.",
}

COLD_START_DEFAULTS: dict[str, str] = {
    "chunk_type": "overview",
    "complexity": "general",
    "audience": "general",
    # "system" is set dynamically from the last-opened model
}


@dataclass
class ChunkTags:
    chunk_id: str
    source_file: str
    text_preview: str  # first 200 chars for reference
    type: Optional[str] = None
    domain: Optional[str] = None
    system: Optional[str] = None
    location: Optional[str] = None
    chunk_type: Optional[str] = None
    complexity: Optional[str] = None
    audience: Optional[str] = None
    scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "text_preview": self.text_preview,
            "tags": {
                "type": self.type,
                "domain": self.domain,
                "system": self.system,
                "location": self.location,
                "chunk_type": self.chunk_type,
                "complexity": self.complexity,
                "audience": self.audience,
            },
            "scores": self.scores,
        }
