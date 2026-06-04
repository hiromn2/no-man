"""Tests for src/feedback.py — governance loop functions.

Covers all five public functions:
  propose_edge, run_adversarial_review, submit_for_human_review,
  human_approve, flag_edge

Also covers the graph_io additions required by this module:
  "proposed" status, governance field validators, cross-graph uniqueness check.

LLM and Semantic Scholar calls are mocked throughout.
"""

from __future__ import annotations

import json
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feedback import (  # noqa: E402
    propose_edge,
    run_adversarial_review,
    submit_for_human_review,
    human_approve,
    flag_edge,
)
from graph_io import (  # noqa: E402
    GraphIntegrityError,
    validate,
    check_cross_graph_uniqueness,
)


# ── shared fixtures ───────────────────────────────────────────────────────────


NODES_CONTENT = {
    "nodes": [
        {
            "id": "node_a",
            "name_en": "Node A",
            "name_ja": "ノードA",
            "category": "financial",
            "welfare_relevant": False,
            "description_en": "A",
            "description_ja": "A",
        },
        {
            "id": "node_b",
            "name_en": "Node B",
            "name_ja": "ノードB",
            "category": "real_economy",
            "welfare_relevant": True,
            "welfare_direction": "negative",
            "description_en": "B",
            "description_ja": "B",
        },
        {
            "id": "node_c",
            "name_en": "Node C",
            "name_ja": "ノードC",
            "category": "macro",
            "welfare_relevant": False,
            "description_en": "C",
            "description_ja": "C",
        },
    ]
}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_proposed_edge(
    edge_id: str = "WE-TEST01",
    from_node: str = "node_a",
    to_node: str = "node_b",
    sign: str = "-",
    proposed_by: str = "test_worker",
) -> dict:
    return {
        "id": edge_id,
        "from": from_node,
        "to": to_node,
        "sign": sign,
        "proposed_by": proposed_by,
        "status": "speculative",  # session sets "speculative"; propose_edge will override
        "confidence": "low",
        "confidence_tier": 1,
        "sources": [],
        "monotone": True,
        "monotone_note": None,
        "regime": "normal",
        "regime_condition": None,
        "regime_flip": None,
        "research_gap": False,
        "research_gap_note": None,
        "international_evidence": [],
        "local_weight_note": None,
        "name_en": f"Node {from_node} → Node {to_node}",
        "name_ja": f"ノード{from_node} → ノード{to_node}",
        "confirmed_by": [],
        "rejected_by": [],
        "cycle_risk": False,
        "cycle_risk_note": None,
        "last_updated": "2026-05-31",
    }


def _make_speculative_edge_with_result(
    edge_id: str = "WE-SPEC01",
    from_node: str = "node_a",
    to_node: str = "node_b",
    sources: list | None = None,
) -> dict:
    """Return a fully-formed speculative edge ready for human_approve()."""
    edge = _make_proposed_edge(edge_id=edge_id, from_node=from_node, to_node=to_node)
    edge["status"] = "speculative"
    edge["sources"] = sources if sources is not None else ["Test Source 2024"]
    edge["adversarial_result"] = {
        "counterclaim": {"from": "node_b", "to": "node_a", "sign": "+", "basis": "counter"},
        "supporting_papers": [],
        "search_exhausted": True,
        "reviewed_at": "2026-05-30T00:00:00+00:00",
        "method": "llm_adversarial_review",
    }
    return edge


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """Create a temp knowledge directory with nodes.json and empty graphs."""
    _write_json(tmp_path / "nodes.json", NODES_CONTENT)
    _write_json(tmp_path / "speculative_graph.json", {"edges": []})
    _write_json(tmp_path / "seed_graph.json", {"edges": []})
    return tmp_path


def _make_session(worker_edges: list[dict]) -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "decision_type_id": "D03",
        "worker_edges": worker_edges,
        "chains": [],
        "report_ids": [],
        "status": "open",
    }


# ── propose_edge ──────────────────────────────────────────────────────────────


def test_propose_edge_writes_proposed_status(graph_dir):
    edge = _make_proposed_edge()
    session = _make_session([edge])

    propose_edge(edge, session, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    assert len(data["edges"]) == 1
    assert data["edges"][0]["status"] == "proposed"


def test_propose_edge_assigns_default_confidence_and_tier(graph_dir):
    edge = _make_proposed_edge()
    edge.pop("confidence", None)
    edge.pop("confidence_tier", None)
    session = _make_session([edge])

    propose_edge(edge, session, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    written = data["edges"][0]
    assert written["confidence"] == "low"
    assert written["confidence_tier"] == 1


def test_propose_edge_raises_if_not_in_session(graph_dir):
    edge = _make_proposed_edge(edge_id="WE-A")
    other = _make_proposed_edge(edge_id="WE-B")  # different ID in session
    session = _make_session([other])

    with pytest.raises(ValueError, match="not found in session"):
        propose_edge(edge, session, graph_dir / "speculative_graph.json")


def test_propose_edge_raises_if_id_already_in_graph(graph_dir):
    edge = _make_proposed_edge(edge_id="WE-DUP")
    session = _make_session([edge])

    propose_edge(edge, session, graph_dir / "speculative_graph.json")

    with pytest.raises(GraphIntegrityError, match="already exists"):
        propose_edge(edge, session, graph_dir / "speculative_graph.json")


def test_propose_edge_raises_if_id_in_seed_graph(graph_dir):
    """Cross-graph uniqueness: reject proposal if ID already in seed_graph.json."""
    edge = _make_proposed_edge(edge_id="E-CONFLICT")
    session = _make_session([edge])

    # Pre-populate seed with the same ID
    seed_edge = _make_speculative_edge_with_result(edge_id="E-CONFLICT")
    seed_edge["status"] = "tested"
    _write_json(graph_dir / "seed_graph.json", {"edges": [seed_edge]})

    with pytest.raises(GraphIntegrityError, match="E-CONFLICT"):
        propose_edge(edge, session, graph_dir / "speculative_graph.json")


def test_propose_edge_raises_on_unknown_node(graph_dir):
    edge = _make_proposed_edge(from_node="nonexistent_node_xyz")
    session = _make_session([edge])

    with pytest.raises(GraphIntegrityError, match="Invariant 1"):
        propose_edge(edge, session, graph_dir / "speculative_graph.json")


def test_propose_edge_uses_save_graph_not_raw_dump(graph_dir):
    """After propose_edge the file must wrap edges under 'edges' key (save_graph format)."""
    edge = _make_proposed_edge()
    session = _make_session([edge])

    propose_edge(edge, session, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    assert "edges" in data
    assert isinstance(data["edges"], list)


# ── run_adversarial_review ────────────────────────────────────────────────────


def _mock_anthropic_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _setup_proposed_edge_in_graph(graph_dir: Path) -> dict:
    """Write a proposed edge to speculative_graph.json and return it."""
    edge = _make_proposed_edge()
    session = _make_session([edge])
    propose_edge(edge, session, graph_dir / "speculative_graph.json")
    # Re-read to get the saved version
    data = _read_json(graph_dir / "speculative_graph.json")
    return data["edges"][0]


_COUNTERCLAIM_JSON = '{"from": "node_b", "to": "node_a", "sign": "+", "basis": "Reverse channel dominates"}'


def test_run_adversarial_review_writes_adversarial_result(graph_dir):
    _setup_proposed_edge_in_graph(graph_dir)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(_COUNTERCLAIM_JSON)

    with patch("feedback._get_anthropic_client", return_value=mock_client):
        with patch("literature.search_edge_support", return_value=[]):
            result = run_adversarial_review(
                "WE-TEST01",
                graph_dir / "speculative_graph.json",
                graph_dir / "nodes.json",
            )

    assert "counterclaim" in result
    assert result["method"] == "llm_adversarial_review"
    assert "reviewed_at" in result

    # Verify written to disk
    data = _read_json(graph_dir / "speculative_graph.json")
    assert data["edges"][0]["adversarial_result"] is not None


def test_run_adversarial_review_search_exhausted_when_no_papers(graph_dir):
    _setup_proposed_edge_in_graph(graph_dir)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(_COUNTERCLAIM_JSON)

    with patch("feedback._get_anthropic_client", return_value=mock_client):
        with patch("literature.search_edge_support", return_value=[]):
            result = run_adversarial_review(
                "WE-TEST01",
                graph_dir / "speculative_graph.json",
                graph_dir / "nodes.json",
            )

    assert result["search_exhausted"] is True
    assert result["supporting_papers"] == []


def test_run_adversarial_review_finds_supporting_paper(graph_dir):
    _setup_proposed_edge_in_graph(graph_dir)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(_COUNTERCLAIM_JSON)

    fake_paper = {
        "title": "Counter Evidence 2024",
        "year": 2024,
        "doi": "10.xxxx/yyy",
        "source_type": "peer_reviewed_journal_article",
        "is_accepted_source_type": True,
    }

    with patch("feedback._get_anthropic_client", return_value=mock_client):
        with patch("literature.search_edge_support", return_value=[fake_paper]):
            result = run_adversarial_review(
                "WE-TEST01",
                graph_dir / "speculative_graph.json",
                graph_dir / "nodes.json",
            )

    assert result["search_exhausted"] is False
    assert len(result["supporting_papers"]) == 1
    assert result["supporting_papers"][0]["title"] == "Counter Evidence 2024"


def test_run_adversarial_review_requires_proposed_status(graph_dir):
    # Write an edge directly as "speculative" (not "proposed")
    spec_edge = _make_speculative_edge_with_result()
    _write_json(graph_dir / "speculative_graph.json", {"edges": [spec_edge]})

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(_COUNTERCLAIM_JSON)

    with patch("feedback._get_anthropic_client", return_value=mock_client):
        with pytest.raises(ValueError, match="expected 'proposed'"):
            run_adversarial_review(
                spec_edge["id"],
                graph_dir / "speculative_graph.json",
                graph_dir / "nodes.json",
            )


def test_run_adversarial_review_edge_not_found(graph_dir):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(_COUNTERCLAIM_JSON)

    with patch("feedback._get_anthropic_client", return_value=mock_client):
        with pytest.raises(ValueError, match="not found"):
            run_adversarial_review(
                "NONEXISTENT",
                graph_dir / "speculative_graph.json",
                graph_dir / "nodes.json",
            )


def test_run_adversarial_review_handles_llm_json_error(graph_dir):
    """When LLM returns non-JSON, adversarial_result still written with llm_error."""
    _setup_proposed_edge_in_graph(graph_dir)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        "This is not JSON at all."
    )

    with patch("feedback._get_anthropic_client", return_value=mock_client):
        with patch("literature.search_edge_support", return_value=[]):
            result = run_adversarial_review(
                "WE-TEST01",
                graph_dir / "speculative_graph.json",
                graph_dir / "nodes.json",
            )

    assert "llm_error" in result
    assert result["search_exhausted"] is True  # no counterclaim → no search


# ── submit_for_human_review ───────────────────────────────────────────────────


def _setup_edge_with_adversarial_result(graph_dir: Path) -> None:
    """Write a proposed edge that already has adversarial_result."""
    edge = _make_proposed_edge()
    edge["adversarial_result"] = {
        "counterclaim": {"from": "node_b", "to": "node_a", "sign": "+", "basis": "test"},
        "supporting_papers": [],
        "search_exhausted": True,
        "reviewed_at": _now_iso(),
        "method": "llm_adversarial_review",
    }
    edge["status"] = "proposed"
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_submit_for_human_review_changes_status_to_speculative(graph_dir):
    _setup_edge_with_adversarial_result(graph_dir)

    submit_for_human_review("WE-TEST01", graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    assert data["edges"][0]["status"] == "speculative"


def test_submit_for_human_review_creates_and_appends_review_queue(graph_dir):
    _setup_edge_with_adversarial_result(graph_dir)
    queue_path = graph_dir / "review_queue.json"
    assert not queue_path.exists()

    submit_for_human_review("WE-TEST01", graph_dir / "speculative_graph.json")

    assert queue_path.exists()
    q = _read_json(queue_path)
    assert "WE-TEST01" in q["queue"]


def test_submit_for_human_review_idempotent_queue_append(graph_dir):
    """Calling twice should not duplicate the entry in the queue."""
    _setup_edge_with_adversarial_result(graph_dir)

    submit_for_human_review("WE-TEST01", graph_dir / "speculative_graph.json")

    # Status is now "speculative"; re-calling raises ValueError (wrong status),
    # so queue should still have exactly one entry
    with pytest.raises(ValueError, match="expected 'proposed'"):
        submit_for_human_review("WE-TEST01", graph_dir / "speculative_graph.json")

    q = _read_json(graph_dir / "review_queue.json")
    assert q["queue"].count("WE-TEST01") == 1


def test_submit_for_human_review_requires_proposed_status(graph_dir):
    spec_edge = _make_speculative_edge_with_result()
    spec_edge["status"] = "speculative"
    _write_json(graph_dir / "speculative_graph.json", {"edges": [spec_edge]})

    with pytest.raises(ValueError, match="expected 'proposed'"):
        submit_for_human_review(spec_edge["id"], graph_dir / "speculative_graph.json")


def test_submit_for_human_review_requires_adversarial_result(graph_dir):
    """Edge without adversarial_result must be rejected."""
    edge = _make_proposed_edge()
    edge["status"] = "proposed"
    edge.pop("adversarial_result", None)
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})

    with pytest.raises(ValueError, match="adversarial_result"):
        submit_for_human_review(edge["id"], graph_dir / "speculative_graph.json")


def test_submit_for_human_review_edge_not_found(graph_dir):
    with pytest.raises(ValueError, match="not found"):
        submit_for_human_review("NONEXISTENT", graph_dir / "speculative_graph.json")


# ── human_approve ─────────────────────────────────────────────────────────────


def _setup_for_approval(graph_dir: Path) -> str:
    """Write speculative edge ready for approval. Returns edge_id."""
    edge = _make_speculative_edge_with_result()
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})
    # Pre-populate review queue
    _write_json(graph_dir / "review_queue.json", {
        "queue": [edge["id"]], "created_at": "2026-05-31T00:00:00+00:00"
    })
    return edge["id"]


def test_human_approve_writes_edge_to_seed(graph_dir):
    edge_id = _setup_for_approval(graph_dir)

    human_approve(
        edge_id, "reviewer_tanaka",
        graph_dir / "speculative_graph.json",
        graph_dir / "seed_graph.json",
        graph_dir / "nodes.json",
    )

    seed_data = _read_json(graph_dir / "seed_graph.json")
    assert any(e["id"] == edge_id for e in seed_data["edges"])


def test_human_approve_sets_tested_status(graph_dir):
    edge_id = _setup_for_approval(graph_dir)

    human_approve(
        edge_id, "reviewer_tanaka",
        graph_dir / "speculative_graph.json",
        graph_dir / "seed_graph.json",
        graph_dir / "nodes.json",
    )

    seed_data = _read_json(graph_dir / "seed_graph.json")
    approved = next(e for e in seed_data["edges"] if e["id"] == edge_id)
    assert approved["status"] == "tested"


def test_human_approve_sets_verified_fields(graph_dir):
    edge_id = _setup_for_approval(graph_dir)

    human_approve(
        edge_id, "reviewer_tanaka",
        graph_dir / "speculative_graph.json",
        graph_dir / "seed_graph.json",
        graph_dir / "nodes.json",
    )

    seed_data = _read_json(graph_dir / "seed_graph.json")
    approved = next(e for e in seed_data["edges"] if e["id"] == edge_id)
    assert approved["verified_by"] == "reviewer_tanaka"
    assert approved["verified_at"] is not None


def test_human_approve_removes_from_speculative(graph_dir):
    edge_id = _setup_for_approval(graph_dir)

    human_approve(
        edge_id, "reviewer_tanaka",
        graph_dir / "speculative_graph.json",
        graph_dir / "seed_graph.json",
        graph_dir / "nodes.json",
    )

    spec_data = _read_json(graph_dir / "speculative_graph.json")
    assert not any(e["id"] == edge_id for e in spec_data["edges"])


def test_human_approve_removes_from_review_queue(graph_dir):
    edge_id = _setup_for_approval(graph_dir)

    human_approve(
        edge_id, "reviewer_tanaka",
        graph_dir / "speculative_graph.json",
        graph_dir / "seed_graph.json",
        graph_dir / "nodes.json",
    )

    q = _read_json(graph_dir / "review_queue.json")
    assert edge_id not in q["queue"]


def test_human_approve_raises_if_id_already_in_seed(graph_dir):
    """Cross-graph guard: raise if edge already exists in seed graph."""
    edge = _make_speculative_edge_with_result(edge_id="E-ALREADY")
    # Write to BOTH graphs
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})
    seed_edge = dict(edge)
    seed_edge["status"] = "tested"
    _write_json(graph_dir / "seed_graph.json", {"edges": [seed_edge]})

    with pytest.raises(GraphIntegrityError, match="E-ALREADY"):
        human_approve(
            "E-ALREADY", "reviewer_tanaka",
            graph_dir / "speculative_graph.json",
            graph_dir / "seed_graph.json",
            graph_dir / "nodes.json",
        )


def test_human_approve_raises_if_no_sources(graph_dir):
    edge = _make_speculative_edge_with_result()
    edge["sources"] = []
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})

    with pytest.raises(GraphIntegrityError, match="no sources"):
        human_approve(
            edge["id"], "reviewer_tanaka",
            graph_dir / "speculative_graph.json",
            graph_dir / "seed_graph.json",
            graph_dir / "nodes.json",
        )


def test_human_approve_requires_speculative_status(graph_dir):
    edge = _make_proposed_edge()
    edge["status"] = "proposed"
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})

    with pytest.raises(ValueError, match="expected 'speculative'"):
        human_approve(
            edge["id"], "reviewer_tanaka",
            graph_dir / "speculative_graph.json",
            graph_dir / "seed_graph.json",
            graph_dir / "nodes.json",
        )


def test_human_approve_edge_not_found(graph_dir):
    with pytest.raises(ValueError, match="not found"):
        human_approve(
            "NONEXISTENT", "reviewer_tanaka",
            graph_dir / "speculative_graph.json",
            graph_dir / "seed_graph.json",
            graph_dir / "nodes.json",
        )


# ── flag_edge ─────────────────────────────────────────────────────────────────


def _setup_flaggable_edge(graph_dir: Path) -> str:
    """Write a speculative edge to speculative_graph.json for flagging tests."""
    edge = _make_speculative_edge_with_result(edge_id="WE-FLAG01")
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})
    return edge["id"]


def test_flag_edge_appends_dispute_flag(graph_dir):
    edge_id = _setup_flaggable_edge(graph_dir)

    flag_edge(edge_id, "analyst_x", "Magnitude reverses", None,
              graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    edge = next(e for e in data["edges"] if e["id"] == edge_id)
    assert len(edge["dispute_flags"]) == 1
    assert edge["dispute_flags"][0]["flagged_by"] == "analyst_x"
    assert edge["dispute_flags"][0]["reason"] == "Magnitude reverses"
    assert edge["dispute_flags"][0]["paper"] is None
    assert "flagged_at" in edge["dispute_flags"][0]


def test_flag_edge_with_paper_stored(graph_dir):
    edge_id = _setup_flaggable_edge(graph_dir)
    paper = {"title": "Counter 2024", "doi": "10.x/y"}

    flag_edge(edge_id, "analyst_y", "Contradicted here", paper,
              graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    edge = next(e for e in data["edges"] if e["id"] == edge_id)
    assert edge["dispute_flags"][0]["paper"] == paper


def test_flag_edge_no_review_required_below_threshold(graph_dir):
    """2 flags < 3 → review_required must NOT be set."""
    edge_id = _setup_flaggable_edge(graph_dir)

    flag_edge(edge_id, "a", "reason 1", None, graph_dir / "speculative_graph.json")
    flag_edge(edge_id, "b", "reason 2", None, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    edge = next(e for e in data["edges"] if e["id"] == edge_id)
    assert not edge.get("review_required")


def test_flag_edge_no_review_required_when_flags_are_recent(graph_dir):
    """3 flags but all recent (< 90 days) → review_required must NOT be set."""
    edge = _make_speculative_edge_with_result(edge_id="WE-RECENT")
    now_str = datetime.now(timezone.utc).isoformat()
    edge["dispute_flags"] = [
        {"flagged_by": "a", "reason": "x", "paper": None, "flagged_at": now_str},
        {"flagged_by": "b", "reason": "y", "paper": None, "flagged_at": now_str},
    ]
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})

    flag_edge("WE-RECENT", "c", "z", None, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    e = next(x for x in data["edges"] if x["id"] == "WE-RECENT")
    assert not e.get("review_required")


def test_flag_edge_sets_review_required_after_threshold_and_90_days(graph_dir):
    """3rd flag where oldest existing flag is > 90 days → review_required=True."""
    edge = _make_speculative_edge_with_result(edge_id="WE-OLD")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    edge["dispute_flags"] = [
        {"flagged_by": "a", "reason": "old 1", "paper": None, "flagged_at": old_ts},
        {"flagged_by": "b", "reason": "old 2", "paper": None, "flagged_at": old_ts},
    ]
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})

    flag_edge("WE-OLD", "c", "new flag", None, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    e = next(x for x in data["edges"] if x["id"] == "WE-OLD")
    assert e.get("review_required") is True
    assert "accumulated_flags" in e.get("review_reason", "")


def test_flag_edge_no_review_required_if_human_verified(graph_dir):
    """Even with 3 old flags, if verified_by is set, review_required is not triggered."""
    edge = _make_speculative_edge_with_result(edge_id="WE-VERIFIED")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    edge["dispute_flags"] = [
        {"flagged_by": "a", "reason": "x", "paper": None, "flagged_at": old_ts},
        {"flagged_by": "b", "reason": "y", "paper": None, "flagged_at": old_ts},
    ]
    edge["verified_by"] = "reviewer_tanaka"
    _write_json(graph_dir / "speculative_graph.json", {"edges": [edge]})

    flag_edge("WE-VERIFIED", "c", "z", None, graph_dir / "speculative_graph.json")

    data = _read_json(graph_dir / "speculative_graph.json")
    e = next(x for x in data["edges"] if x["id"] == "WE-VERIFIED")
    assert not e.get("review_required")


def test_flag_edge_on_seed_graph(graph_dir):
    """flag_edge works on seed_graph.json too."""
    edge = _make_speculative_edge_with_result(edge_id="E-SEED-FLAG")
    edge["status"] = "tested"
    _write_json(graph_dir / "seed_graph.json", {"edges": [edge]})

    flag_edge("E-SEED-FLAG", "analyst_z", "Seed-level dispute", None,
              graph_dir / "seed_graph.json")

    data = _read_json(graph_dir / "seed_graph.json")
    e = next(x for x in data["edges"] if x["id"] == "E-SEED-FLAG")
    assert len(e["dispute_flags"]) == 1


def test_flag_edge_edge_not_found(graph_dir):
    with pytest.raises(ValueError, match="not found"):
        flag_edge("NOPE", "x", "y", None, graph_dir / "speculative_graph.json")


# ── graph_io additions: proposed status ──────────────────────────────────────


MINIMAL_NODES = {
    "node_a": {"id": "node_a", "name_en": "A", "name_ja": "A", "category": "financial",
               "welfare_relevant": False, "description_en": "A", "description_ja": "A"},
    "node_b": {"id": "node_b", "name_en": "B", "name_ja": "B", "category": "real_economy",
               "welfare_relevant": True, "welfare_direction": "negative",
               "description_en": "B", "description_ja": "B"},
}


def _base_proposed_edge() -> dict:
    return {
        "id": "E-P01", "from": "node_a", "to": "node_b",
        "sign": "-", "status": "proposed", "confidence": "low",
        "proposed_by": "test_worker",
        "monotone": True, "monotone_note": None, "regime": "normal",
        "regime_condition": None, "regime_flip": None, "research_gap": False,
        "research_gap_note": None, "international_evidence": [], "sources": [],
        "local_weight_note": None, "name_en": "A→B", "name_ja": "A→B",
        "confirmed_by": [], "rejected_by": [], "cycle_risk": False,
        "cycle_risk_note": None, "last_updated": "2026-05-31",
    }


def test_proposed_is_valid_status():
    edges = [_base_proposed_edge()]
    validate(edges, MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_proposed_status_requires_proposed_by():
    edge = _base_proposed_edge()
    edge["proposed_by"] = None
    with pytest.raises(GraphIntegrityError, match="Invariant 7"):
        validate([edge], MINIMAL_NODES, graph_type="speculative")


# ── graph_io additions: governance field validators ───────────────────────────


def test_dispute_flags_invalid_type_raises():
    edge = _base_proposed_edge()
    edge["dispute_flags"] = "not a list"
    with pytest.raises(GraphIntegrityError, match="dispute_flags"):
        validate([edge], MINIMAL_NODES, graph_type="speculative")


def test_dispute_flags_as_list_passes():
    edge = _base_proposed_edge()
    edge["dispute_flags"] = [{"flagged_by": "x", "reason": "y", "flagged_at": "t"}]
    validate([edge], MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_dispute_flags_null_passes():
    edge = _base_proposed_edge()
    edge["dispute_flags"] = None
    validate([edge], MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_adversarial_result_invalid_type_raises():
    edge = _base_proposed_edge()
    edge["adversarial_result"] = "should be dict"
    with pytest.raises(GraphIntegrityError, match="adversarial_result"):
        validate([edge], MINIMAL_NODES, graph_type="speculative")


def test_adversarial_result_as_dict_passes():
    edge = _base_proposed_edge()
    edge["adversarial_result"] = {"counterclaim": {}, "search_exhausted": True}
    validate([edge], MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_review_required_invalid_type_raises():
    edge = _base_proposed_edge()
    edge["review_required"] = "yes"  # should be bool
    with pytest.raises(GraphIntegrityError, match="review_required"):
        validate([edge], MINIMAL_NODES, graph_type="speculative")


def test_review_required_bool_passes():
    edge = _base_proposed_edge()
    edge["review_required"] = True
    validate([edge], MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_review_reason_invalid_type_raises():
    edge = _base_proposed_edge()
    edge["review_reason"] = 42
    with pytest.raises(GraphIntegrityError, match="review_reason"):
        validate([edge], MINIMAL_NODES, graph_type="speculative")


def test_review_reason_string_passes():
    edge = _base_proposed_edge()
    edge["review_reason"] = "accumulated_flags"
    validate([edge], MINIMAL_NODES, graph_type="speculative")  # must not raise


# ── graph_io additions: cross-graph uniqueness ────────────────────────────────


def test_check_cross_graph_uniqueness_detects_overlap():
    edges_a = [{"id": "E-001"}, {"id": "E-002"}]
    edges_b = [{"id": "E-002"}, {"id": "E-003"}]
    with pytest.raises(GraphIntegrityError, match="E-002"):
        check_cross_graph_uniqueness(edges_a, edges_b)


def test_check_cross_graph_uniqueness_no_overlap_passes():
    edges_a = [{"id": "E-001"}, {"id": "E-002"}]
    edges_b = [{"id": "E-003"}, {"id": "E-004"}]
    check_cross_graph_uniqueness(edges_a, edges_b)  # must not raise


def test_check_cross_graph_uniqueness_empty_graphs_pass():
    check_cross_graph_uniqueness([], [])  # must not raise


def test_check_cross_graph_uniqueness_one_empty():
    edges_a = [{"id": "E-001"}]
    check_cross_graph_uniqueness(edges_a, [])  # must not raise
    check_cross_graph_uniqueness([], edges_a)  # must not raise
