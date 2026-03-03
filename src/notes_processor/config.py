from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    incoming_folder_id: str
    output_folder_id: str
    processed_folder_id: str

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    # One or more providers to run for each transcript, in order.
    # e.g. "openai", "anthropic" (or "claude").
    llm_providers: Tuple[str, ...] = ("openai",)

    max_files_per_run: int = 25
    min_transcript_chars: int = 200


def load_config_from_env() -> Config:
    def req(name: str) -> str:
        v = os.getenv(name)
        if not v:
            raise RuntimeError(f"Missing required env var: {name}")
        return v

    incoming = req("INCOMING_FOLDER_ID")
    out = req("OUTPUT_FOLDER_ID")
    processed = req("PROCESSED_FOLDER_ID")

    providers_env = os.getenv("LLM_PROVIDERS")
    if providers_env:
        providers: Tuple[str, ...] = tuple(
            p.strip() for p in providers_env.split(",") if p.strip()
        )
    else:
        providers = (os.getenv("LLM_PROVIDER", "openai"),)

    return Config(
        incoming_folder_id=incoming,
        output_folder_id=out,
        processed_folder_id=processed,
        llm_provider=providers[0],
        llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        llm_providers=providers,
        max_files_per_run=int(os.getenv("MAX_FILES_PER_RUN", "25")),
        min_transcript_chars=int(os.getenv("MIN_TRANSCRIPT_CHARS", "200")),
    )
