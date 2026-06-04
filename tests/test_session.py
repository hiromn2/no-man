"""Tests for src/session.py — session lifecycle management.

Covers:
  - create_session: all required fields present, defaults correct
  - add_worker_edge: validates required fields, raises on missing, enriches edge
  - close_session: sets status and closed_at, raises if already closed
  - save_session / load_session: roundtrip, correct path, SessionNotFoundError
  - attach_report: appends report_id without mutation
  - attach_chains: stores chains without mutation
  - No mutation: all functions return new dicts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session import (  # noqa: E402
    SessionNotFoundError,
    WorkerEdgeValidationError,
    add_worker_edge,
    attach_chains,
    attach_report,
    close_session,
    create_session,
    load_session,
    save_session,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_session(tmp_path: Path, **overrides) -> dict:
    s = create_session(
        decision_type_id="D02",
        local_context={"sector": "real_estate", "prefecture": "愛知県"},
        language="ja",
        sessions_root=tmp_path,
    )
    s.update(overrides)
    return s


def _valid_worker_edge(**overrides) -> dict:
    base = {
        "from": "bank_loan_volume",
        "to": "regional_sme_credit_access",
        "sign": "+",
        "proposed_by": "worker_tanaka",
    }
    base.update(overrides)
    return base


# ── create_session ────────────────────────────────────────────────────────────


def test_create_session_has_session_id(tmp_path):
    s = _make_session(tmp_path)
    assert "session_id" in s
    assert isinstance(s["session_id"], str)
    assert len(s["session_id"]) > 0


def test_create_session_session_id_is_uuid(tmp_path):
    import uuid
    s = _make_session(tmp_path)
    uuid.UUID(s["session_id"])  # raises ValueError if not a valid UUID


def test_create_session_status_is_open(tmp_path):
    s = _make_session(tmp_path)
    assert s["status"] == "open"


def test_create_session_closed_at_is_none(tmp_path):
    s = _make_session(tmp_path)
    assert s["closed_at"] is None


def test_create_session_created_at_is_string(tmp_path):
    s = _make_session(tmp_path)
    assert isinstance(s["created_at"], str)
    assert len(s["created_at"]) > 0


def test_create_session_worker_edges_empty(tmp_path):
    s = _make_session(tmp_path)
    assert s["worker_edges"] == []


def test_create_session_chains_empty(tmp_path):
    s = _make_session(tmp_path)
    assert s["chains"] == []


def test_create_session_report_ids_empty(tmp_path):
    s = _make_session(tmp_path)
    assert s["report_ids"] == []


def test_create_session_stores_decision_type_id(tmp_path):
    s = _make_session(tmp_path)
    assert s["decision_type_id"] == "D02"


def test_create_session_stores_local_context(tmp_path):
    s = _make_session(tmp_path)
    assert s["local_context"]["prefecture"] == "愛知県"


def test_create_session_default_local_context_empty(tmp_path):
    s = create_session("D01", sessions_root=tmp_path)
    assert s["local_context"] == {}


def test_create_session_stores_language(tmp_path):
    s = _make_session(tmp_path)
    assert s["language"] == "ja"


def test_create_session_default_language_is_ja(tmp_path):
    s = create_session("D01", sessions_root=tmp_path)
    assert s["language"] == "ja"


def test_create_session_language_en(tmp_path):
    s = create_session("D01", language="en", sessions_root=tmp_path)
    assert s["language"] == "en"


def test_create_session_custom_local_context(tmp_path):
    s = create_session(
        "D01",
        local_context={"regime": "interest_rate_normalization", "region": "tohoku"},
        sessions_root=tmp_path,
    )
    assert s["local_context"]["regime"] == "interest_rate_normalization"
    assert s["local_context"]["region"] == "tohoku"


def test_create_session_unique_ids(tmp_path):
    s1 = _make_session(tmp_path)
    s2 = _make_session(tmp_path)
    assert s1["session_id"] != s2["session_id"]


# ── add_worker_edge ───────────────────────────────────────────────────────────


def test_add_worker_edge_appends_to_worker_edges(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    result = add_worker_edge(s, edge)
    assert len(result["worker_edges"]) == 1


def test_add_worker_edge_forces_status_speculative(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge(status="tested")
    result = add_worker_edge(s, edge)
    assert result["worker_edges"][0]["status"] == "speculative"


def test_add_worker_edge_assigns_id_when_missing(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    result = add_worker_edge(s, edge)
    assert "id" in result["worker_edges"][0]
    assert len(result["worker_edges"][0]["id"]) > 0


def test_add_worker_edge_preserves_existing_id(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge(id="WE-CUSTOM-001")
    result = add_worker_edge(s, edge)
    assert result["worker_edges"][0]["id"] == "WE-CUSTOM-001"


def test_add_worker_edge_adds_added_at(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    result = add_worker_edge(s, edge)
    assert "added_at" in result["worker_edges"][0]


def test_add_worker_edge_multiple_edges(tmp_path):
    s = _make_session(tmp_path)
    s = add_worker_edge(s, _valid_worker_edge(sign="+", proposed_by="worker_a"))
    s = add_worker_edge(s, _valid_worker_edge(sign="-", proposed_by="worker_b"))
    assert len(s["worker_edges"]) == 2


def test_add_worker_edge_raises_on_missing_from(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    del edge["from"]
    with pytest.raises(WorkerEdgeValidationError, match="from"):
        add_worker_edge(s, edge)


def test_add_worker_edge_raises_on_missing_to(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    del edge["to"]
    with pytest.raises(WorkerEdgeValidationError, match="to"):
        add_worker_edge(s, edge)


def test_add_worker_edge_raises_on_missing_sign(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    del edge["sign"]
    with pytest.raises(WorkerEdgeValidationError, match="sign"):
        add_worker_edge(s, edge)


def test_add_worker_edge_raises_on_missing_proposed_by(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge()
    del edge["proposed_by"]
    with pytest.raises(WorkerEdgeValidationError, match="proposed_by"):
        add_worker_edge(s, edge)


def test_add_worker_edge_raises_on_invalid_sign(tmp_path):
    s = _make_session(tmp_path)
    edge = _valid_worker_edge(sign="?")
    with pytest.raises(WorkerEdgeValidationError, match="sign"):
        add_worker_edge(s, edge)


def test_add_worker_edge_accepts_all_valid_signs(tmp_path):
    for sign in ("+", "-", "ambiguous"):
        s = _make_session(tmp_path)
        edge = _valid_worker_edge(sign=sign)
        result = add_worker_edge(s, edge)
        assert result["worker_edges"][0]["sign"] == sign


def test_add_worker_edge_does_not_mutate_session(tmp_path):
    s = _make_session(tmp_path)
    original_edges = list(s["worker_edges"])
    add_worker_edge(s, _valid_worker_edge())
    assert s["worker_edges"] == original_edges


# ── close_session ─────────────────────────────────────────────────────────────


def test_close_session_sets_status_closed(tmp_path):
    s = _make_session(tmp_path)
    result = close_session(s)
    assert result["status"] == "closed"


def test_close_session_sets_closed_at(tmp_path):
    s = _make_session(tmp_path)
    result = close_session(s)
    assert result["closed_at"] is not None
    assert isinstance(result["closed_at"], str)
    assert len(result["closed_at"]) > 0


def test_close_session_raises_if_already_closed(tmp_path):
    s = _make_session(tmp_path)
    s = close_session(s)
    with pytest.raises(ValueError, match="already closed"):
        close_session(s)


def test_close_session_does_not_mutate_input(tmp_path):
    s = _make_session(tmp_path)
    original_status = s["status"]
    close_session(s)
    assert s["status"] == original_status
    assert s["closed_at"] is None


def test_close_session_preserves_other_fields(tmp_path):
    s = _make_session(tmp_path)
    s = add_worker_edge(s, _valid_worker_edge())
    result = close_session(s)
    assert len(result["worker_edges"]) == 1
    assert result["decision_type_id"] == s["decision_type_id"]
    assert result["session_id"] == s["session_id"]


# ── save_session / load_session roundtrip ─────────────────────────────────────


def test_save_creates_session_json(tmp_path):
    s = _make_session(tmp_path)
    path = save_session(s)
    assert path.exists()
    assert path.name == "session.json"


def test_save_creates_directory_per_session(tmp_path):
    s = _make_session(tmp_path)
    path = save_session(s)
    assert path.parent.name == s["session_id"]


def test_load_roundtrip_preserves_all_fields(tmp_path):
    s = _make_session(tmp_path)
    s = add_worker_edge(s, _valid_worker_edge())
    save_session(s)
    loaded = load_session(s["session_id"], sessions_root=tmp_path)
    assert loaded["session_id"] == s["session_id"]
    assert loaded["decision_type_id"] == s["decision_type_id"]
    assert loaded["language"] == s["language"]
    assert loaded["status"] == s["status"]
    assert len(loaded["worker_edges"]) == 1


def test_load_roundtrip_worker_edge_fields_intact(tmp_path):
    s = _make_session(tmp_path)
    s = add_worker_edge(s, _valid_worker_edge(proposed_by="worker_suzuki"))
    save_session(s)
    loaded = load_session(s["session_id"], sessions_root=tmp_path)
    edge = loaded["worker_edges"][0]
    assert edge["proposed_by"] == "worker_suzuki"
    assert edge["status"] == "speculative"


def test_load_roundtrip_closed_session(tmp_path):
    s = _make_session(tmp_path)
    s = close_session(s)
    save_session(s)
    loaded = load_session(s["session_id"], sessions_root=tmp_path)
    assert loaded["status"] == "closed"
    assert loaded["closed_at"] is not None


def test_save_roundtrip_produces_valid_json(tmp_path):
    s = _make_session(tmp_path)
    path = save_session(s)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["session_id"] == s["session_id"]


def test_load_raises_session_not_found_error(tmp_path):
    with pytest.raises(SessionNotFoundError):
        load_session("nonexistent-uuid-0000", sessions_root=tmp_path)


def test_load_raises_session_not_found_error_message_contains_id(tmp_path):
    bad_id = "00000000-dead-beef-0000-000000000000"
    with pytest.raises(SessionNotFoundError, match=bad_id):
        load_session(bad_id, sessions_root=tmp_path)


def test_save_creates_parent_directories(tmp_path):
    deep_root = tmp_path / "a" / "b" / "sessions"
    s = create_session("D01", sessions_root=deep_root)
    path = save_session(s)
    assert path.exists()


# ── attach_report / attach_chains ─────────────────────────────────────────────


def test_attach_report_appends_id(tmp_path):
    s = _make_session(tmp_path)
    result = attach_report(s, "report-001")
    assert "report-001" in result["report_ids"]


def test_attach_report_multiple(tmp_path):
    s = _make_session(tmp_path)
    s = attach_report(s, "report-001")
    s = attach_report(s, "report-002")
    assert len(s["report_ids"]) == 2


def test_attach_report_does_not_mutate_session(tmp_path):
    s = _make_session(tmp_path)
    original = list(s["report_ids"])
    attach_report(s, "report-x")
    assert s["report_ids"] == original


def test_attach_chains_stores_chains(tmp_path):
    s = _make_session(tmp_path)
    chains = [{"chain_id": "C1"}, {"chain_id": "C2"}]
    result = attach_chains(s, chains)
    assert len(result["chains"]) == 2


def test_attach_chains_does_not_mutate_session(tmp_path):
    s = _make_session(tmp_path)
    original = list(s["chains"])
    attach_chains(s, [{"chain_id": "C1"}])
    assert s["chains"] == original


def test_attach_chains_replaces_existing(tmp_path):
    s = _make_session(tmp_path)
    s = attach_chains(s, [{"chain_id": "C1"}])
    s = attach_chains(s, [{"chain_id": "C2"}, {"chain_id": "C3"}])
    assert len(s["chains"]) == 2
    assert s["chains"][0]["chain_id"] == "C2"
