from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# CANONICAL RENDER ORDER
#
# Sections are rendered in this order when present. Any key returned by the
# model that is not in this list falls through to the "Additional" catch-all
# at the bottom — which acts as a safety net for model-invented keys.
#
# When you promote a suggested_new_section to a real section:
#   1. Add the key + render function/entry to _SECTION_RENDERERS below.
#   2. Add the key to _KNOWN_KEYS so it doesn't double-render in Additional.
#   3. Add the key to schema.py properties.
#   4. Add the key to prompt.py _KNOWN_SECTIONS.
# ---------------------------------------------------------------------------

# Keys handled by explicit render logic (will not appear in Additional).
_KNOWN_KEYS = frozenset(
    {
        "title",
        "date",
        "session_type",
        "participants",
        "summary",
        "key_concepts",
        "vocabulary_terms",
        "drills",
        "common_mistakes",
        "patterns_and_sequences",
        "student_observations",
        "action_items",
        "competition_notes",
        "quotes",
        "references",
        "off_topic_notes",
        "suggested_new_sections",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2(title: str) -> list[str]:
    return [f"## {title}", ""]


def _h3(title: str) -> list[str]:
    return [f"### {title}"]


def _bullet(text: str) -> str:
    return f"- {text}"


def _skip_if_empty(val: Any) -> bool:
    """Return True if the value should cause the whole section to be skipped."""
    if val is None:
        return True
    if isinstance(val, (list, dict)) and not val:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def _generic_value(val: Any) -> list[str]:
    """Fallback renderer for unexpected shapes (used in Additional catch-all)."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        out: list[str] = []
        for item in val:
            if isinstance(item, str):
                out.append(_bullet(item))
            elif isinstance(item, dict):
                out.extend(f"  - **{k}**: {v}" for k, v in item.items() if v)
            else:
                out.append(_bullet(str(item)))
        return out
    if isinstance(val, dict):
        return [f"- **{k}**: {v}" for k, v in val.items() if v]
    return [str(val)]


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_metadata(notes: dict[str, Any]) -> list[str]:
    """Render date / session_type / participants as a tidy metadata block."""
    lines: list[str] = []

    date = (notes.get("date") or "").strip()
    session_type = (notes.get("session_type") or "").strip().replace("_", " ").title()

    if date or session_type:
        lines.append("")
        if date:
            lines.append(f"**Date:** {date}  ")
        if session_type:
            lines.append(f"**Session type:** {session_type}  ")

    participants = notes.get("participants") or []
    if participants:
        lines.append("")
        lines.append("**Participants:**")
        for p in participants:
            if not isinstance(p, dict):
                continue
            name = (p.get("name") or "").strip()
            role = (p.get("role") or "").strip()
            label = (p.get("label") or "").strip()
            parts = []
            if name:
                parts.append(name)
            if role:
                parts.append(role)
            if label and label not in parts:
                parts.append(f"({label})")
            if parts:
                lines.append(f"- {' — '.join(parts)}")

    return lines


def _render_key_concepts(items: list) -> list[str]:
    lines = [*_h2("Key Concepts")]
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(_bullet(item.strip()))
    return lines


def _render_vocabulary_terms(items: list) -> list[str]:
    lines = [*_h2("Vocabulary Terms")]
    for item in items:
        if not isinstance(item, dict):
            continue
        term = (item.get("term") or "").strip()
        defn = (item.get("definition") or "").strip()
        if term:
            lines.append(f"- **{term}**" + (f": {defn}" if defn else ""))
    return lines


def _render_drills(items: list) -> list[str]:
    lines = [*_h2("Drills / Exercises")]
    for i, d in enumerate(items, start=1):
        if not isinstance(d, dict):
            continue
        name = (d.get("name") or "").strip()
        goal = (d.get("goal") or "").strip()
        header = f"{i}. **{name}**" if name else f"{i}."
        if goal:
            header += f" — {goal}"
        lines.append(header)
        for step in d.get("steps") or []:
            if isinstance(step, str) and step.strip():
                lines.append(f"   - {step.strip()}")
    return lines


def _render_common_mistakes(items: list) -> list[str]:
    lines = [*_h2("Common Mistakes & Corrections")]
    for m in items:
        if not isinstance(m, dict):
            continue
        mistake = (m.get("mistake") or "").strip()
        correction = (m.get("correction") or "").strip()
        if mistake:
            lines.append(
                f"- **{mistake}**" + (f" → {correction}" if correction else "")
            )
    return lines


def _render_simple_list(heading: str, items: list) -> list[str]:
    lines = [*_h2(heading)]
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(_bullet(item.strip()))
    return lines


def _render_patterns_and_sequences(items: list) -> list[str]:
    """Items may be strings or objects with name/description."""
    lines = [*_h2("Patterns & Sequences")]
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(_bullet(item.strip()))
        elif isinstance(item, dict):
            name = (item.get("name") or "").strip()
            desc = (item.get("description") or "").strip()
            if name:
                lines.append(_bullet(f"**{name}**" + (f": {desc}" if desc else "")))
    return lines


def _render_references(items: list) -> list[str]:
    """References can be strings or objects with name/type/context."""
    lines = [*_h2("References")]
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(_bullet(item.strip()))
        elif isinstance(item, dict):
            name = (item.get("name") or "").strip()
            typ = (item.get("type") or "").strip()
            ctx = (item.get("context") or "").strip()
            if name:
                line = f"**{name}**"
                if typ:
                    line += f" ({typ})"
                if ctx:
                    line += f": {ctx}"
                lines.append(_bullet(line))
    return lines


def _render_off_topic_notes(items: list) -> list[str]:
    """Off-topic items may be plain strings or objects with topic/summary."""
    lines = [*_h2("Off-Topic Notes")]
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(_bullet(item.strip()))
        elif isinstance(item, dict):
            topic = (item.get("topic") or "").strip()
            summary = (item.get("summary") or "").strip()
            if topic:
                lines.append(
                    _bullet(f"**{topic}**" + (f": {summary}" if summary else ""))
                )
            elif summary:
                lines.append(_bullet(summary))
    return lines


def _render_student_observations(items: list) -> list[str]:
    """Student observations may be plain strings or objects with observation."""
    lines = [*_h2("Student Observations")]
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(_bullet(item.strip()))
        elif isinstance(item, dict):
            obs = (item.get("observation") or "").strip()
            if obs:
                lines.append(_bullet(obs))
    return lines


def _render_quotes(items: list) -> list[str]:
    """Quotes may be plain strings or objects with speaker/quote/context."""
    lines = [*_h2("Memorable Quotes")]
    for q in items:
        if isinstance(q, str) and q.strip():
            lines.append(f'- "{q.strip()}"')
        elif isinstance(q, dict):
            quote = (q.get("quote") or "").strip()
            if not quote:
                continue
            speaker = (q.get("speaker") or "").strip()
            if speaker:
                lines.append(f'- **{speaker}:** "{quote}"')
            else:
                lines.append(f'- "{quote}"')
    return lines


def _render_suggested_new_sections(items: list) -> list[str]:
    lines = [
        *_h2("⚑ Suggested New Sections"),
        "> The model flagged the following content as potentially recurring but "
        "not fitting any existing section. Review periodically and promote to "
        "the schema if warranted.",
        "",
    ]
    for s in items:
        if not isinstance(s, dict):
            continue
        name = (s.get("suggested_name") or "").strip()
        why = (s.get("rationale") or "").strip()
        sample = (s.get("sample_content") or "").strip()
        if name:
            lines.append(f"### `{name}`")
        if why:
            lines.append(f"**Why flagged:** {why}")
        if sample:
            lines.append(f"**Example:** {sample}")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Ordered render plan
# Each entry: (key, renderer_fn)
# Renderer receives the value for that key and returns list[str].
# ---------------------------------------------------------------------------
_SECTION_RENDERERS: list[tuple[str, Any]] = [
    ("key_concepts", _render_key_concepts),
    ("vocabulary_terms", _render_vocabulary_terms),
    ("drills", _render_drills),
    ("common_mistakes", _render_common_mistakes),
    (
        "patterns_and_sequences",
        _render_patterns_and_sequences,
    ),
    ("student_observations", _render_student_observations),
    ("action_items", lambda v: _render_simple_list("Action Items", v)),
    ("competition_notes", lambda v: _render_simple_list("Competition Notes", v)),
    ("quotes", _render_quotes),
    ("references", _render_references),
    ("off_topic_notes", _render_off_topic_notes),
    ("suggested_new_sections", _render_suggested_new_sections),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_markdown(notes: dict[str, Any]) -> str:
    """Render a notes JSON dict to a Markdown string.

    Sections are rendered in canonical order and only when the value is
    non-empty. Unknown keys from the model are collected into an Additional
    section at the end as a safety net.
    """
    lines: list[str] = []

    # Title
    title = (notes.get("title") or "Notes").strip()
    lines.append(f"# {title}")

    # Metadata block (date, session_type, participants)
    lines.extend(_render_metadata(notes))
    lines.append("")

    # Summary
    summary = (notes.get("summary") or "").strip()
    if summary:
        lines.extend(_h2("Summary"))
        lines.append(summary)
        lines.append("")

    # Canonical sections in order
    for key, renderer in _SECTION_RENDERERS:
        val = notes.get(key)
        if _skip_if_empty(val):
            continue
        lines.append("")
        lines.extend(renderer(val))
        lines.append("")

    # Additional catch-all — keys the model invented that aren't in _KNOWN_KEYS
    extra = {
        k: v for k, v in notes.items() if k not in _KNOWN_KEYS and not _skip_if_empty(v)
    }
    if extra:
        lines.append("")
        lines.extend(_h2("Additional"))
        for key, val in extra.items():
            label = key.replace("_", " ").title()
            lines.extend(_h3(label))
            lines.extend(_generic_value(val))
            lines.append("")

    # Trim trailing blank lines, add final newline
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")

    return "\n".join(lines)
