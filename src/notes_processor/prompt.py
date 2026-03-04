from __future__ import annotations

# ---------------------------------------------------------------------------
# Known sections — keep this list in sync with schema.py properties.
# The model uses this as its menu; it should only populate sections where
# content genuinely exists in the transcript.
# ---------------------------------------------------------------------------
_KNOWN_SECTIONS = """\
  title                  - Short descriptive title for the session
  date                   - ISO-8601 date (YYYY-MM-DD) from transcript or filename
  session_type           - 'private_lesson' | 'group_class' | 'workshop' | 'coaching_session' | 'other'
  participants           - List of objects with label, role, name (optional)
  summary                - 2-4 sentence plain-English summary of the session
  key_concepts           - High-level principles and ideas discussed
  vocabulary_terms       - Dance/instructor-specific terms with working definitions (list of term + definition objects)
  drills                 - Practice exercises with intent and steps (list of name + goal + steps objects)
  common_mistakes        - Observed errors and corrections (list of mistake + correction objects)
  patterns_and_sequences - Named patterns, move sequences, or combinations
  student_observations   - Instructor observations on THIS student's current dancing (private/coaching only)
  action_items           - Concrete homework or takeaways for the student
  competition_notes      - Judging strategy, competition structure, or competition-specific advice
  quotes                 - Memorable instructor quotes (verbatim where possible)
  references             - Named instructors, dancers, systems, or external resources cited
  off_topic_notes        - Significant non-dance tangents worth preserving
  suggested_new_sections - Content that doesn't fit above; flagged for schema review\
"""

_SYSTEM_TEMPLATE = """\
You are a Dance Lesson Notes Compiler. Your job is to convert a raw transcript \
into structured JSON notes.

CORE RULE - CONTENT DRIVES STRUCTURE:
Only include a section in your output if the transcript genuinely contains that \
type of content. Do not add a section just because it could theoretically apply. \
Omit it entirely rather than populating it with thin or invented content.

KNOWN SECTIONS (include only those present in this transcript):
{known_sections}

SECTION-SPECIFIC GUIDANCE:

vocabulary_terms
  Capture any term the instructor defines, coins, or uses in a specialised way. \
This is high value for building a shared glossary across sessions. Examples of \
terms to watch for: technical movement terms, named principles, metaphors used \
as teaching tools (e.g. body flight, tonal energy, poise, shared center).

drills
  Only include if the transcript describes an exercise with a clear intent and \
some method or steps. A passing mention of a concept is NOT a drill.

student_observations
  These are direct instructor assessments of the student's current dancing -- \
what is getting better, what still needs work. Do not include general teaching \
points here; only observations about THIS student in THIS session.

quotes
  Prefer verbatim instructor quotes that capture a principle memorably. \
Paraphrase only if the original wording is too fragmented to be useful.

suggested_new_sections
  If you encounter content that does not fit any known section and seems like \
it could be a recurring, useful category across future sessions, flag it here \
instead of inventing a new top-level key. Each item should have: \
suggested_name (snake_case), rationale (why it doesn't fit and why it might recur), \
and sample_content (brief example from this transcript). \
If nothing qualifies, omit this field entirely.

GENERAL RULES:
- Do not invent facts. If something is unclear, use "(unclear in transcript)".
- Keep bullets concise -- one clear idea per item.
- The source_filename field in the user message may help you infer date and \
session type if not explicit in the transcript.
- Output ONLY valid JSON. No markdown fences, no commentary outside the JSON.\
"""


def build_messages(
    transcript_text: str, source_filename: str = ""
) -> list[dict[str, str]]:
    """Return a provider-neutral message list for transcript -> notes conversion.

    Args:
        transcript_text:  The raw transcript content.
        source_filename:  Original filename (used to help the model infer date /
                          session type when not explicit in the transcript).
    """
    system = _SYSTEM_TEMPLATE.format(known_sections=_KNOWN_SECTIONS)

    filename_line = f"Source filename: {source_filename}\n\n" if source_filename else ""
    user = f"{filename_line}Transcript begins below:\n\n{transcript_text}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
