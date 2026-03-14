from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    input_folder_id: str
    output_folder_id: str
    processed_folder_id: str

    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    # One or more providers to run for each transcript, in order.
    # e.g. "openai", "anthropic" (or "claude").
    llm_providers: Tuple[str, ...] = ("anthropic",)

    max_files_per_run: int = 25
    min_transcript_chars: int = 200
    max_output_tokens: int = 8192


def load_config_from_env() -> Config:
    def req(name: str) -> str:
        v = os.getenv(name)
        if not v:
            raise RuntimeError(f"Missing required env var: {name}")
        return v

    incoming = req("INPUT_FOLDER_ID")
    out = req("OUTPUT_FOLDER_ID")
    processed = req("PROCESSED_FOLDER_ID")

    providers_env = os.getenv("LLM_PROVIDERS")
    if providers_env:
        providers = tuple(p.strip() for p in providers_env.split(",") if p.strip())
    else:
        # LLM_PROVIDER can also be comma-separated for multiple providers
        provider_env = os.getenv("LLM_PROVIDER", "anthropic")
        providers = tuple(p.strip() for p in provider_env.split(",") if p.strip()) or (
            "anthropic",
        )

    return Config(
        input_folder_id=incoming,
        output_folder_id=out,
        processed_folder_id=processed,
        llm_provider=providers[0],
        llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
        llm_providers=providers,
        max_files_per_run=int(os.getenv("MAX_FILES_PER_RUN", "25")),
        min_transcript_chars=int(os.getenv("MIN_TRANSCRIPT_CHARS", "200")),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "8192")),
    )
