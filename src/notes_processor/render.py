from __future__ import annotations

from typing import Any

# Keys we render in fixed order/sections; any other keys go into "Additional".
_CANONICAL_KEYS = frozenset(
    {
        "title",
        "summary",
        "key_concepts",
        "drills",
        "common_mistakes",
        "patterns",
        "quotes",
        "logistics",
    }
)


def _format_value(val: Any) -> list[str]:
    """Turn a value into markdown lines (bullets or single line)."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        out: list[str] = []
        for item in val:
            if isinstance(item, str):
                out.append(f"- {item}")
            elif isinstance(item, dict):
                parts = [f"  - **{k}**: {v}" for k, v in item.items() if v]
                out.extend(parts)
            else:
                out.append(f"- {item}")
        return out
    if isinstance(val, dict):
        return [f"- **{k}**: {v}" for k, v in val.items() if v]
    return [str(val)]


def render_markdown(notes: dict[str, Any]) -> str:
    """Render Markdown from notes JSON. Uses canonical sections first, then any extra keys."""

    def _lines(items: list[str]) -> str:
        return "\n".join(items)

    lines: list[str] = []
    lines.append(f"# {notes.get('title') or 'Notes'}")
    lines.append("")
    lines.append("## Summary")
    lines.append((notes.get("summary") or "").strip())
    lines.append("")
    lines.append("## Key Concepts")
    for x in notes.get("key_concepts", []):
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Drills / Exercises")
    for i, d in enumerate(notes.get("drills", []), start=1):
        name = d.get("name", "").strip()
        goal = d.get("goal", "").strip()
        lines.append(f"{i}. **{name}** — {goal}".rstrip())
        for s in d.get("steps", []):
            lines.append(f"   - {s}")
    lines.append("")
    lines.append("## Common Mistakes & Corrections")
    for m in notes.get("common_mistakes", []):
        lines.append(f"- **{m.get('mistake', '')}** → {m.get('correction', '')}")
    lines.append("")
    lines.append("## Patterns / Sequences")
    for p in notes.get("patterns", []):
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Memorable Quotes")
    for q in notes.get("quotes", []):
        lines.append(f'- "{q}"')
    lines.append("")
    lines.append("## Announcements / Logistics")
    for logistics_item in notes.get("logistics", []):
        lines.append(f"- {logistics_item}")

    # Any additional keys from the model (e.g. dance_style, key_reminders, sections)
    extra = {k: notes[k] for k in notes if k not in _CANONICAL_KEYS}
    if extra:
        lines.append("")
        lines.append("## Additional information")
        for key, val in extra.items():
            if val is None:
                continue
            label = key.replace("_", " ").title()
            lines.append(f"### {label}")
            for line in _format_value(val):
                lines.append(line)
            lines.append("")
    lines.append("")
    return _lines(lines)
