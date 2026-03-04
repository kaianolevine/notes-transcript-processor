from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTES SCHEMA
#
# Design principles:
#
#   1. CONTENT-DRIVEN — every section is optional. The model should only
#      populate a section when content genuinely exists in the transcript.
#      An absent section means "not present in this session", not "forgotten".
#
#   2. KNOWN SECTIONS — the properties below are the current recognised
#      vocabulary. Adding a new section here promotes it from "suggested" to
#      "canonical" and it will be rendered with its own heading.
#
#   3. FEEDBACK LOOP — `suggested_new_sections` lets the model flag content
#      that doesn't fit any known section. Review these periodically; if a
#      suggestion appears across many transcripts, promote it to a real section
#      by adding it here and updating render.py.
#
#   4. STABILITY — additionalProperties: True on the root object means new
#      model-invented keys won't cause validation failures. We surface those
#      in the "Additional" catch-all during rendering.
# ---------------------------------------------------------------------------

NOTES_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        # -- Always expected --------------------------------------------------
        "title": {
            "type": "string",
            "description": "Short descriptive title for this session.",
        },
        "date": {
            "type": "string",
            "description": "ISO-8601 date of the session (YYYY-MM-DD) if determinable from the transcript or filename.",
        },
        "session_type": {
            "type": "string",
            "description": (
                "One of: 'private_lesson', 'group_class', 'workshop', "
                "'coaching_session', or 'other'. Infer from context."
            ),
        },
        "participants": {
            "type": "array",
            "description": "Named or role-labelled participants. Include at minimum role (e.g. Instructor, Student) and speaker label.",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "label": {"type": "string"},
                    "role": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        },
        "summary": {
            "type": "string",
            "description": "2-4 sentence plain-English summary of what this session covered and any notable outcomes.",
        },
        # -- Core content sections (populate only if content exists) ----------
        "key_concepts": {
            "type": "array",
            "description": (
                "High-level principles or ideas discussed. Each item is a "
                "concise statement of the concept."
            ),
            "items": {"type": "string"},
        },
        "vocabulary_terms": {
            "type": "array",
            "description": (
                "Dance-specific or instructor-specific terms that were defined "
                "or meaningfully used. Capture the term and its working "
                "definition as used in this session."
            ),
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                },
            },
        },
        "drills": {
            "type": "array",
            "description": "Specific practice exercises given. Only include if the transcript describes an actual drill with intent and method.",
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
            "description": "Errors observed or discussed, paired with the correction.",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "mistake": {"type": "string"},
                    "correction": {"type": "string"},
                },
            },
        },
        "patterns_and_sequences": {
            "type": "array",
            "description": "Named patterns, move sequences, or combinations taught or referenced. Each item may be a string or an object with name and description.",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                ],
            },
        },
        "student_observations": {
            "type": "array",
            "description": (
                "Instructor observations about the specific student's current "
                "dancing -- what is improving, what still needs work. "
                "Only present in private or coaching sessions."
            ),
            "items": {"type": "string"},
        },
        "action_items": {
            "type": "array",
            "description": "Concrete takeaways or homework assigned to the student for next practice.",
            "items": {"type": "string"},
        },
        "competition_notes": {
            "type": "array",
            "description": "Strategy, judging insight, or competition-specific advice discussed.",
            "items": {"type": "string"},
        },
        "quotes": {
            "type": "array",
            "description": "Memorable or particularly clear instructor quotes. Use the speaker's actual words.",
            "items": {"type": "string"},
        },
        "references": {
            "type": "array",
            "description": "Named instructors, dancers, systems, or resources cited. Each item may be a string (e.g. 'Robert Royston shared center system') or an object with name, type, context for richer references.",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "context": {"type": "string"},
                        },
                    },
                ],
            },
        },
        "off_topic_notes": {
            "type": "array",
            "description": (
                "Significant non-dance tangents worth preserving "
                "(personal updates, logistics, scheduling, etc.)."
            ),
            "items": {"type": "string"},
        },
        # -- Feedback loop ----------------------------------------------------
        "suggested_new_sections": {
            "type": "array",
            "description": (
                "If you find content that does not fit any of the known sections "
                "above and seems like it could be a useful recurring category, "
                "flag it here. Do NOT invent a new top-level key -- put it here "
                "for human review instead."
            ),
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "suggested_name": {
                        "type": "string",
                        "description": "snake_case name for the proposed section.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this content does not fit existing sections and why it might recur.",
                    },
                    "sample_content": {
                        "type": "string",
                        "description": "A brief example of the content from this transcript.",
                    },
                },
            },
        },
    },
}
