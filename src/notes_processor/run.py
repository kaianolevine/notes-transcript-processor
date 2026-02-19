from __future__ import annotations

import json
import re

from googleapiclient.errors import HttpError
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
TXT_MIME = "text/plain"
FOLDER_MIME = "application/vnd.google-apps.folder"


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


def _read_transcript_text(g: GoogleAPI, file_id: str, mime_type: str) -> str:
    """Read transcript text from either a Google Doc or a plain .txt file."""

    if mime_type == DOC_MIME:
        return g.drive.export_google_doc_as_text(file_id)

    if mime_type == TXT_MIME:
        # kaiano.google Drive helpers have varied over time; try a few common shapes.
        for meth in (
            "download_bytes",
            "download_file_bytes",
            "download_file_as_bytes",
            "get_file_bytes",
        ):
            if hasattr(g.drive, meth):
                b = getattr(g.drive, meth)(file_id)
                if isinstance(b, (bytes, bytearray)):
                    return bytes(b).decode("utf-8", errors="replace")

        # Last-resort: some clients expose a `.files.get(...).execute()` style.
        if hasattr(g.drive, "service"):
            try:
                req = g.drive.service.files().get_media(fileId=file_id)
                data = req.execute()
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data).decode("utf-8", errors="replace")
            except Exception:
                pass

        raise TypeError(
            "Drive client does not expose a supported bytes download method for text/plain files"
        )

    raise ValueError(f"Unsupported mime type for transcript: {mime_type}")


def _extract_llm_json(result: object) -> dict:
    """Extract parsed JSON from kaiano.llm result objects across versions.

    Supports:
      - dict returned directly
      - result.data (older)
      - result.json / result.parsed / result.output (newer)
      - result.content / result.text containing JSON
    """

    if isinstance(result, dict):
        return result

    # Common attribute names across versions
    for attr in ("data", "json", "parsed", "output", "output_json", "parsed_json"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, dict):
                return val
    # Some versions keep the raw JSON text
    for attr in ("content", "text", "raw", "response", "output_text"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, str) and val.strip():
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

    raise TypeError(
        f"Unsupported LLM result type/shape: {type(result).__name__}. Expected dict or an object with parsed JSON on .data/.json/.parsed/.output or JSON text on .content/.text"
    )


def _safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
    return name[:180] if len(name) > 180 else name


def _get_drive_file_metadata(g: GoogleAPI, file_id: str) -> dict:
    """Best-effort fetch of Drive file metadata across kaiano.google versions."""

    # Try common facade methods first
    for meth in (
        "get_file",
        "get_file_metadata",
        "get_metadata",
        "get_file_by_id",
    ):
        if hasattr(g.drive, meth):
            meta = getattr(g.drive, meth)(file_id)
            if isinstance(meta, dict):
                return meta
            # Some wrappers return objects
            if meta is not None:
                return {
                    "id": _field(meta, "id"),
                    "name": _field(meta, "name"),
                    "mimeType": _field(meta, "mimeType"),
                }

    # Raw service fallback
    if hasattr(g.drive, "service"):
        meta = (
            g.drive.service.files()
            .get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True)
            .execute()
        )
        if isinstance(meta, dict):
            return meta

    return {"id": file_id}


def _assert_drive_folder_access(g: GoogleAPI, folder_id: str, label: str) -> None:
    """Fail fast with a clear error if a configured folder id is missing/inaccessible."""

    try:
        meta = _get_drive_file_metadata(g, folder_id)
        mt = meta.get("mimeType")
        if mt and mt != FOLDER_MIME:
            raise ValueError(
                f"{label} is not a folder (mimeType={mt}): {folder_id} ({meta.get('name')})"
            )
    except HttpError as e:
        # Typical case when the service account/user cannot see the folder or the ID is wrong.
        if (
            getattr(e, "resp", None) is not None
            and getattr(e.resp, "status", None) == 404
        ):
            raise ValueError(
                f"{label} folder not found or not shared with this credential: {folder_id}. "
                "Double-check the ID and ensure the folder is shared with the service account (or the OAuth user) running this job."
            ) from e
        raise
    except Exception as e:
        # Preserve the original error but add context.
        raise RuntimeError(
            f"Failed to verify {label} folder access for id={folder_id}: {e}"
        ) from e


def _find_child_folder_id(g: GoogleAPI, parent_id: str, folder_name: str) -> str | None:
    """Return the Drive folder id for `folder_name` under `parent_id`, if it exists."""

    for item in g.drive.get_files_in_folder(parent_id, include_folders=True):
        if (
            _field(item, "mimeType") == FOLDER_MIME
            and _field(item, "name") == folder_name
        ):
            return _field(item, "id")
    return None


def _create_child_folder(g: GoogleAPI, parent_id: str, folder_name: str) -> str:
    """Create a Drive folder under `parent_id` and return its id.

    The kaiano.google Drive facade has changed over time; try a few common method names.
    """

    for meth in (
        "create_folder",
        "create_drive_folder",
        "create_folder_in_parent",
        "mkdir",
    ):
        if hasattr(g.drive, meth):
            created = getattr(g.drive, meth)(parent_id=parent_id, name=folder_name)
            # created may be an id string, dict, or object
            if isinstance(created, str):
                return created
            if isinstance(created, dict):
                fid = created.get("id")
                if fid:
                    return fid
            fid = _field(created, "id")
            if fid:
                return fid

    # Last-resort: raw google drive service
    if hasattr(g.drive, "service"):
        body = {"name": folder_name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
        try:
            created = (
                g.drive.service.files()
                .create(body=body, fields="id", supportsAllDrives=True)
                .execute()
            )
            if isinstance(created, dict) and created.get("id"):
                return created["id"]
        except HttpError as e:
            # Drive returns 404 here when the *parent* folder id is wrong or not accessible.
            if (
                getattr(e, "resp", None) is not None
                and getattr(e.resp, "status", None) == 404
            ):
                raise ValueError(
                    f"Cannot create folder '{folder_name}' under parent '{parent_id}' (not found / not shared). "
                    "Verify OUTPUT_FOLDER_ID / PROCESSED_FOLDER_ID and ensure they are shared with the credential running this job."
                ) from e
            raise

    raise TypeError(
        "Drive client does not expose a supported folder creation method (expected create_folder/create_drive_folder/etc)"
    )


def _ensure_child_folder(g: GoogleAPI, parent_id: str, folder_name: str) -> str:
    """Get or create a named child folder and return its id."""

    existing = _find_child_folder_id(g, parent_id, folder_name)
    if existing:
        return existing
    return _create_child_folder(g, parent_id, folder_name)


def _ensure_path(g: GoogleAPI, root_id: str, parts: list[str]) -> str:
    """Ensure the folder path root/parts exists and return the deepest folder id."""

    cur = root_id
    for part in parts:
        # Skip empty path parts defensively
        if not part:
            continue
        cur = _ensure_child_folder(g, cur, part)
    return cur


def _iter_transcript_files_recursive(
    g: GoogleAPI, folder_id: str, rel_parts: list[str] | None = None
):
    """Yield (file_obj, rel_parts) for supported transcript files in folder and subfolders."""

    rel_parts = rel_parts or []

    for item in g.drive.get_files_in_folder(folder_id, include_folders=True):
        mt = _field(item, "mimeType")
        if mt == FOLDER_MIME:
            child_id = _field(item, "id")
            child_name = _field(item, "name")
            if child_id and child_name:
                yield from _iter_transcript_files_recursive(
                    g, child_id, rel_parts + [child_name]
                )
            continue

        if mt in (DOC_MIME, TXT_MIME):
            yield item, rel_parts


def run() -> None:
    cfg = load_config_from_env()
    g = GoogleAPI.from_env()
    llm = build_llm(provider=cfg.llm_provider, model=cfg.llm_model)

    # Fail fast if any configured folder IDs are wrong or not shared with this credential.
    _assert_drive_folder_access(g, cfg.incoming_folder_id, "Incoming")
    _assert_drive_folder_access(g, cfg.output_folder_id, "Output")
    _assert_drive_folder_access(g, cfg.processed_folder_id, "Processed")

    run_id = new_run_id()
    metrics_logger = MetricsLogger()

    LOG.info(
        "Scanning incoming folder", extra={"incoming_folder_id": cfg.incoming_folder_id}
    )

    files: list[tuple[object, list[str]]] = list(
        _iter_transcript_files_recursive(g, cfg.incoming_folder_id)
    )

    if not files:
        LOG.info("No files found")
        return

    processed_count = 0

    for f, rel_parts in files:
        file_timer = Timer()
        if processed_count >= cfg.max_files_per_run:
            LOG.info("Reached max files per run", extra={"max": cfg.max_files_per_run})
            break

        file_id = _field(f, "id")
        name = _field(f, "name", file_id)
        mime_type = _field(f, "mimeType")

        # Preserve incoming folder structure in output/processed folders
        output_parent_id = _ensure_path(g, cfg.output_folder_id, rel_parts)
        processed_parent_id = _ensure_path(g, cfg.processed_folder_id, rel_parts)

        # Defaults so failure metrics can still be emitted
        transcript = ""
        char_count_input = 0
        estimated_input_tokens = 0
        char_count_output = None
        estimated_output_tokens = None

        LOG.info("Processing transcript", extra={"file": name, "id": file_id})

        try:
            transcript = _read_transcript_text(g, file_id, mime_type).strip()
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
            notes = _extract_llm_json(result)
            validate(instance=notes, schema=NOTES_SCHEMA)

            md = render_markdown(notes)
            char_count_output = len(md)
            estimated_output_tokens = estimate_tokens(md)

            base = _safe_name(name)
            json_name = f"{base} - notes.json"
            md_name = f"{base} - notes.md"

            g.drive.upload_bytes(
                parent_id=output_parent_id,
                filename=json_name,
                content=json.dumps(notes, indent=2).encode("utf-8"),
                mime_type="application/json",
            )
            g.drive.upload_bytes(
                parent_id=output_parent_id,
                filename=md_name,
                content=md.encode("utf-8"),
                mime_type="text/markdown",
            )

            # Move original into processed folder (auditable + simple)
            g.drive.move_file(file_id, new_parent_id=processed_parent_id)

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
