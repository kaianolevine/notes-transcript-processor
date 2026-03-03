from __future__ import annotations

# Canonical schema for transcript -> notes transformation.
# We validate types when present but require nothing and allow extra fields
# so different models (and richer outputs) don't get rejected.
NOTES_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_concepts": {"type": "array", "items": {"type": "string"}},
        "drills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "name": {"type": "string"},
                    "goal": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "common_mistakes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "mistake": {"type": "string"},
                    "correction": {"type": "string"},
                },
            },
        },
        "patterns": {"type": "array", "items": {"type": "string"}},
        "quotes": {"type": "array", "items": {"type": "string"}},
        "logistics": {"type": "array", "items": {"type": "string"}},
    },
}
