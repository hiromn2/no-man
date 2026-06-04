"""Session lifecycle management for No-Man analysis runs.

A session encapsulates a single analyst interaction: one decision type, one
local context, zero or more worker-proposed speculative edges, and the chains
and reports generated during the run.

Session state is persisted as JSON under sessions/{session_id}/.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


# ── exceptions ────────────────────────────────────────────────────────────────


class SessionNotFoundError(Exception):
    """Raised when a session directory or session.json does not exist."""


class WorkerEdgeValidationError(Exception):
    """Raised when a worker-submitted edge is missing required fields."""


# ── required fields for worker-submitted edges ────────────────────────────────

_REQUIRED_WORKER_EDGE_FIELDS: tuple[str, ...] = (
    "from",
    "to",
    "sign",
    "proposed_by",
)

_VALID_SIGNS = {"+", "-", "ambiguous"}


# ── session schema ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── public API ────────────────────────────────────────────────────────────────


def create_session(
    decision_type_id: str,
    local_context: dict | None = None,
    language: str = "ja",
    sessions_root: Union[str, Path] = "sessions",
) -> dict:
    """Create and return a new session dict.

    The session is not persisted until save_session is called. This function
    only constructs the in-memory dict with all required fields initialized.

    Args:
        decision_type_id: One of D01–D12 from decision_types.json.
        local_context: Optional dict of local context variables (regime, region, etc.).
        language: Output language for reports — "ja" (default) or "en".
        sessions_root: Root directory for session storage (default "sessions").

    Returns:
        A new session dict with status "open".
    """
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "decision_type_id": decision_type_id,
        "local_context": local_context or {},
        "language": language,
        "status": "open",
        "created_at": _now_iso(),
        "closed_at": None,
        "worker_edges": [],
        "chains": [],
        "report_ids": [],
        "sessions_root": str(sessions_root),
    }


def add_worker_edge(session: dict, edge: dict) -> dict:
    """Validate and append a worker-proposed speculative edge to the session.

    The edge is validated for required fields and a valid sign value before
    being added. An auto-generated id is assigned if the edge lacks one.
    Status is forced to "speculative" regardless of what the caller submitted.

    Args:
        session: The session dict returned by create_session (or load_session).
        edge: Edge dict submitted by a worker. Must contain at minimum:
              "from", "to", "sign", "proposed_by".

    Returns:
        A new session dict with the edge appended to worker_edges. Does not
        mutate the input session.

    Raises:
        WorkerEdgeValidationError: If any required field is missing or sign is invalid.
    """
    _validate_worker_edge(edge)

    enriched = dict(edge)
    enriched["status"] = "speculative"
    if "id" not in enriched or not enriched["id"]:
        enriched["id"] = f"WE-{uuid.uuid4().hex[:8].upper()}"
    if "added_at" not in enriched:
        enriched["added_at"] = _now_iso()

    result = dict(session)
    result["worker_edges"] = list(session.get("worker_edges", [])) + [enriched]
    return result


def close_session(session: dict) -> dict:
    """Mark a session as closed and record the close timestamp.

    Returns a new dict. Does not mutate the input session.

    Raises:
        ValueError: If the session is already closed.
    """
    if session.get("status") == "closed":
        raise ValueError(
            f"Session '{session.get('session_id')}' is already closed."
        )
    result = dict(session)
    result["status"] = "closed"
    result["closed_at"] = _now_iso()
    return result


def save_session(session: dict) -> Path:
    """Persist a session dict to sessions/{session_id}/session.json.

    Creates the session directory if it does not exist.

    Args:
        session: The session dict to persist.

    Returns:
        The Path of the written session.json file.
    """
    sessions_root = Path(session.get("sessions_root", "sessions"))
    session_dir = sessions_root / session["session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)

    session_path = session_dir / "session.json"
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

    return session_path


def load_session(
    session_id: str,
    sessions_root: Union[str, Path] = "sessions",
) -> dict:
    """Load a session dict from sessions/{session_id}/session.json.

    Args:
        session_id: UUID string identifying the session.
        sessions_root: Root directory where sessions are stored.

    Returns:
        The session dict.

    Raises:
        SessionNotFoundError: If the session directory or session.json does not exist.
    """
    session_dir = Path(sessions_root) / session_id
    session_path = session_dir / "session.json"

    if not session_path.exists():
        raise SessionNotFoundError(
            f"Session '{session_id}' not found at '{session_path}'. "
            "Ensure the session was saved before loading."
        )

    with open(session_path, encoding="utf-8") as f:
        return json.load(f)


def attach_report(session: dict, report_id: str) -> dict:
    """Record a report ID as belonging to this session.

    Returns a new dict. Does not mutate the input session.
    """
    result = dict(session)
    result["report_ids"] = list(session.get("report_ids", [])) + [report_id]
    return result


def attach_chains(session: dict, chains: list[dict]) -> dict:
    """Store the adverse chains generated for this session.

    Replaces any existing chains (a session's chains are set once, after
    traversal). Returns a new dict. Does not mutate the input session.
    """
    result = dict(session)
    result["chains"] = list(chains)
    return result


def attach_executive_summary(
    session: dict,
    chain_reports: list[dict],
    decision_type: dict,
) -> dict:
    """Call generate_executive_summary and store the result in the session.

    Call this after all per-chain reports have been assembled via assemble_report.
    The resulting executive_summary_ja is intended to appear at the TOP of the
    full session output, before the chain list. featured_chain_id identifies
    which chain was used as the basis for the summary.

    Returns a new dict. Does not mutate the input session.
    """
    from report_generator import generate_executive_summary  # local import — avoids circular

    language = session.get("language", "ja")
    summary = generate_executive_summary(chain_reports, decision_type, language=language)

    result = dict(session)
    result["executive_summary_ja"] = summary.get("executive_summary_ja")
    result["featured_chain_id"] = summary.get("featured_chain_id")
    if summary.get("interpretive_error"):
        result["executive_summary_error"] = summary.get("error_message")
    return result


# ── private helpers ───────────────────────────────────────────────────────────


def _validate_worker_edge(edge: dict) -> None:
    """Raise WorkerEdgeValidationError if any required field is missing or invalid."""
    for field in _REQUIRED_WORKER_EDGE_FIELDS:
        if field not in edge or edge[field] is None or edge[field] == "":
            raise WorkerEdgeValidationError(
                f"Worker edge is missing required field '{field}'. "
                f"Every worker-proposed edge must supply: {list(_REQUIRED_WORKER_EDGE_FIELDS)}."
            )

    sign = edge.get("sign")
    if sign not in _VALID_SIGNS:
        raise WorkerEdgeValidationError(
            f"Worker edge has invalid sign '{sign}'. "
            f"Sign must be one of {sorted(_VALID_SIGNS)}."
        )
