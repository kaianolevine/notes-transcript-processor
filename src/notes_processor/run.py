from __future__ import annotations

import json
import re

from jsonschema import validate
from kaiano import logger as log
from kaiano.google import GoogleAPI
from kaiano.llm import LLMMessage, build_llm

from .config import load_config_from_env
from .metrics import MetricsLogger, RunMetrics, Timer, estimate_tokens, new_run_id
from .prompt import build_messages
from .render import render_markdown
from .schema import NOTES_SCHEMA

LOG = log.get_logger()

DOC_MIME = "application/vnd.google-apps.document"


def _field(obj: object, name: str, default=None):
    """Read a field from either a dict-like object or an attribute-based object."""

    if isinstance(obj, dict):
        return obj.get(name, default)

    # Try exact attribute name
    if hasattr(obj, name):
        return getattr(obj, name)

    # Try common google drive variants
    # mimeType -> mime_type
    if name == "mimeType" and hasattr(obj, "mime_type"):
        return getattr(obj, "mime_type")

    return default


def _is_insufficient_quota(err: Exception) -> bool:
    msg = str(err)
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


def _safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
    return name[:180] if len(name) > 180 else name


def run() -> None:
    cfg = load_config_from_env()
    g = GoogleAPI.from_env()
    llm = build_llm(provider=cfg.llm_provider, model=cfg.llm_model)

    run_id = new_run_id()
    metrics_logger = MetricsLogger()

    LOG.info(
        "Scanning incoming folder", extra={"incoming_folder_id": cfg.incoming_folder_id}
    )

    files = []
    for f in g.drive.get_files_in_folder(cfg.incoming_folder_id, include_folders=False):
        if _field(f, "mimeType") == DOC_MIME:
            files.append(f)

    if not files:
        LOG.info("No files found")
        return

    processed_count = 0

    for f in files:
        file_timer = Timer()
        if processed_count >= cfg.max_files_per_run:
            LOG.info("Reached max files per run", extra={"max": cfg.max_files_per_run})
            break

        file_id = _field(f, "id")
        name = _field(f, "name", file_id)

        # Defaults so failure metrics can still be emitted
        transcript = ""
        char_count_input = 0
        estimated_input_tokens = 0
        char_count_output = None
        estimated_output_tokens = None

        LOG.info("Processing transcript", extra={"file": name, "id": file_id})

        try:
            transcript = g.drive.export_google_doc_as_text(file_id).strip()
            char_count_input = len(transcript)
            estimated_input_tokens = estimate_tokens(transcript)

            if len(transcript) < cfg.min_transcript_chars:
                LOG.warning(
                    "Transcript too short; skipping",
                    extra={"file": name, "chars": len(transcript)},
                )
                continue

            # Build kaiano.llm messages
            msg_dicts = build_messages(transcript)
            messages = [
                LLMMessage(role=m["role"], content=m["content"]) for m in msg_dicts
            ]

            result = llm.generate_json(
                messages=messages, json_schema=NOTES_SCHEMA, schema_name="notes"
            )
            notes = result.data
            validate(instance=notes, schema=NOTES_SCHEMA)

            md = render_markdown(notes)
            char_count_output = len(md)
            estimated_output_tokens = estimate_tokens(md)

            base = _safe_name(name)
            json_name = f"{base} - notes.json"
            md_name = f"{base} - notes.md"

            g.drive.upload_bytes(
                parent_id=cfg.output_folder_id,
                filename=json_name,
                content=json.dumps(notes, indent=2).encode("utf-8"),
                mime_type="application/json",
            )
            g.drive.upload_bytes(
                parent_id=cfg.output_folder_id,
                filename=md_name,
                content=md.encode("utf-8"),
                mime_type="text/markdown",
            )

            # Move original into processed folder (auditable + simple)
            g.drive.move_file(file_id, new_parent_id=cfg.processed_folder_id)

            processed_count += 1
            LOG.info("Done", extra={"file": name})

            metrics = RunMetrics(
                run_id=run_id,
                stage="file_complete",
                file_id=file_id,
                file_name=name,
                char_count_input=char_count_input,
                estimated_input_tokens=estimated_input_tokens,
                model=cfg.llm_model,
                provider=cfg.llm_provider,
                duration_s=file_timer.elapsed(),
                success=True,
                error=None,
                estimated_output_tokens=estimated_output_tokens,
                char_count_output=char_count_output,
                estimated_cost_usd=None,
            )
            try:
                metrics_logger.emit(metrics)
            except Exception:
                # Metrics must never break the pipeline.
                pass

        except Exception as e:
            LOG.exception(
                "Failed processing transcript", extra={"file": name, "id": file_id}
            )

            metrics = RunMetrics(
                run_id=run_id,
                stage="file_complete",
                file_id=file_id,
                file_name=name,
                char_count_input=char_count_input,
                estimated_input_tokens=estimated_input_tokens,
                model=cfg.llm_model,
                provider=cfg.llm_provider,
                duration_s=file_timer.elapsed(),
                success=False,
                error=str(e),
                estimated_output_tokens=estimated_output_tokens,
                char_count_output=char_count_output,
                estimated_cost_usd=None,
            )
            try:
                metrics_logger.emit(metrics)
            except Exception:
                pass

            if _is_insufficient_quota(e):
                LOG.error(
                    "OpenAI quota exhausted; stopping run early. Configure billing/credits for the API key, then re-run.",
                    extra={"provider": cfg.llm_provider, "model": cfg.llm_model},
                )
                break

            # Continue to next file rather than failing the whole run
            continue

    LOG.info("Run complete", extra={"processed": processed_count})
