from __future__ import annotations

from typing import Any


def render_markdown(notes: dict[str, Any]) -> str:
    """Deterministically render Markdown from schema-validated notes JSON."""

    def _lines(items: list[str]) -> str:
        return "\n".join(items)

    lines: list[str] = []
    lines.append(f"# {notes['title']}")
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
    lines.append("")
    return _lines(lines)
