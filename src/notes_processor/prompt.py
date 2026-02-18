from __future__ import annotations


def build_messages(transcript_text: str) -> list[dict[str, str]]:
    """Return provider-neutral message list.

    We keep this simple (system + user) and lean on schema validation for determinism.
    """

    system = (
        "You are a Lecture Notes Compiler. Convert the provided transcript into structured notes. "
        "Output must be JSON that matches the provided JSON Schema. Do not invent facts. "
        "If something is unclear, label it '(unclear in transcript)'. Keep bullets concise."
    )

    user = f"""Transcript begins below:\n\n{transcript_text}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
