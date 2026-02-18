from __future__ import annotations

# Canonical schema for transcript -> notes transformation.
# Keep this stable to preserve consistent formatting across many transcripts.
NOTES_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_concepts": {"type": "array", "items": {"type": "string"}},
        "drills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "goal": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "goal", "steps"],
            },
        },
        "common_mistakes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mistake": {"type": "string"},
                    "correction": {"type": "string"},
                },
                "required": ["mistake", "correction"],
            },
        },
        "patterns": {"type": "array", "items": {"type": "string"}},
        "quotes": {"type": "array", "items": {"type": "string"}},
        "logistics": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "summary",
        "key_concepts",
        "drills",
        "common_mistakes",
        "patterns",
        "quotes",
        "logistics",
    ],
}
