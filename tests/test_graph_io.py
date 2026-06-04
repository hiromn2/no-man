"""Tests for src/graph_io.py — one test per graph invariant plus happy-path tests.

Invariants under test (from CLAUDE.md section 3):
  1. Every edge from/to node exists in nodes.json
  2. sign ∈ {'+', '-', 'ambiguous'}
  3. status ∈ {'tested', 'speculative', 'rejected'}
  4. confidence ∈ {'high', 'medium', 'low'}
  5. The tested (seed) graph contains no cycles
  6. Every tested edge has at least one source
  7. Every speculative edge has a non-null proposed_by
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_io import (  # noqa: E402
    GraphIntegrityError,
    load_graph,
    load_nodes,
    save_graph,
    validate,
)

# ── fixtures & helpers ────────────────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent / "fixtures"

MINIMAL_NODES: dict[str, dict] = {
    "node_a": {
        "id": "node_a",
        "name_en": "Node A",
        "name_ja": "ノードA",
        "category": "financial",
        "welfare_relevant": False,
        "description_en": "A",
        "description_ja": "A",
    },
    "node_b": {
        "id": "node_b",
        "name_en": "Node B",
        "name_ja": "ノードB",
        "category": "real_economy",
        "welfare_relevant": True,
        "description_en": "B",
        "description_ja": "B",
    },
    "node_c": {
        "id": "node_c",
        "name_en": "Node C",
        "name_ja": "ノードC",
        "category": "macro",
        "welfare_relevant": False,
        "description_en": "C",
        "description_ja": "C",
    },
}


def _make_edge(**overrides) -> dict:
    """Return a minimal valid tested edge, with optional field overrides."""
    base = {
        "id": "E-T01",
        "from": "node_a",
        "to": "node_b",
        "sign": "+",
        "status": "tested",
        "confidence": "medium",
        "monotone": True,
        "monotone_note": None,
        "regime": "normal",
        "regime_condition": None,
        "regime_flip": None,
        "research_gap": False,
        "research_gap_note": None,
        "international_evidence": [],
        "sources": ["Test Source 2024"],
        "local_weight_note": None,
        "name_en": "Node A → Node B",
        "name_ja": "ノードA → ノードB",
        "proposed_by": "literature",
        "confirmed_by": [],
        "rejected_by": [],
        "cycle_risk": False,
        "cycle_risk_note": None,
        "last_updated": "2025-01-15",
    }
    base.update(overrides)
    return base


def _three_edge_chain() -> list[dict]:
    """Return a valid three-edge chain A→B→C with no cycle."""
    return [
        _make_edge(id="E-T01", **{"from": "node_a", "to": "node_b"}),
        _make_edge(id="E-T02", **{"from": "node_b", "to": "node_c"}),
    ]


# ── invariant 1: node references ─────────────────────────────────────────────


def test_inv1_unknown_from_node_raises():
    edges = [_make_edge(**{"from": "nonexistent_node", "to": "node_b"})]
    with pytest.raises(GraphIntegrityError, match="Invariant 1"):
        validate(edges, MINIMAL_NODES)


def test_inv1_unknown_to_node_raises():
    edges = [_make_edge(**{"from": "node_a", "to": "ghost_node"})]
    with pytest.raises(GraphIntegrityError, match="Invariant 1"):
        validate(edges, MINIMAL_NODES)


def test_inv1_both_nodes_known_passes():
    edges = [_make_edge(**{"from": "node_a", "to": "node_b"})]
    validate(edges, MINIMAL_NODES)  # must not raise


# ── invariant 2: sign values ─────────────────────────────────────────────────


def test_inv2_invalid_sign_raises():
    edges = [_make_edge(sign="?")]
    with pytest.raises(GraphIntegrityError, match="Invariant 2"):
        validate(edges, MINIMAL_NODES)


def test_inv2_empty_sign_raises():
    edges = [_make_edge(sign="")]
    with pytest.raises(GraphIntegrityError, match="Invariant 2"):
        validate(edges, MINIMAL_NODES)


def test_inv2_none_sign_raises():
    edges = [_make_edge(sign=None)]
    with pytest.raises(GraphIntegrityError, match="Invariant 2"):
        validate(edges, MINIMAL_NODES)


@pytest.mark.parametrize("sign", ["+", "-", "ambiguous"])
def test_inv2_all_valid_signs_pass(sign):
    edges = [_make_edge(sign=sign)]
    validate(edges, MINIMAL_NODES)  # must not raise


# ── invariant 3: status values ───────────────────────────────────────────────


def test_inv3_invalid_status_raises():
    edges = [_make_edge(status="unknown")]
    with pytest.raises(GraphIntegrityError, match="Invariant 3"):
        validate(edges, MINIMAL_NODES)


@pytest.mark.parametrize("status", ["tested", "speculative", "rejected"])
def test_inv3_all_valid_statuses_pass(status):
    edge = _make_edge(status=status)
    if status == "speculative":
        edge["proposed_by"] = "test_worker"
    validate([edge], MINIMAL_NODES)  # must not raise


# ── invariant 4: confidence values ───────────────────────────────────────────


def test_inv4_invalid_confidence_raises():
    edges = [_make_edge(confidence="very_high")]
    with pytest.raises(GraphIntegrityError, match="Invariant 4"):
        validate(edges, MINIMAL_NODES)


def test_inv4_numeric_confidence_raises():
    edges = [_make_edge(confidence=5)]
    with pytest.raises(GraphIntegrityError, match="Invariant 4"):
        validate(edges, MINIMAL_NODES)


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_inv4_all_valid_confidences_pass(confidence):
    edges = [_make_edge(confidence=confidence)]
    validate(edges, MINIMAL_NODES)  # must not raise


# ── invariant 5: no cycles in seed graph ─────────────────────────────────────


def test_inv5_simple_cycle_raises():
    """A→B→C→A must be rejected."""
    edges = [
        _make_edge(id="E-C1", **{"from": "node_a", "to": "node_b"}),
        _make_edge(id="E-C2", **{"from": "node_b", "to": "node_c"}),
        _make_edge(id="E-C3", **{"from": "node_c", "to": "node_a"}),
    ]
    with pytest.raises(GraphIntegrityError, match="Invariant 5"):
        validate(edges, MINIMAL_NODES, graph_type="seed")


def test_inv5_self_loop_raises():
    """A→A must be rejected."""
    edges = [_make_edge(id="E-S1", **{"from": "node_a", "to": "node_a"})]
    with pytest.raises(GraphIntegrityError, match="Invariant 5"):
        validate(edges, MINIMAL_NODES, graph_type="seed")


def test_inv5_dag_passes():
    """A→B→C is a valid DAG."""
    edges = _three_edge_chain()
    validate(edges, MINIMAL_NODES, graph_type="seed")  # must not raise


def test_inv5_cycle_not_checked_for_speculative_graph():
    """Cycle check is skipped when graph_type='speculative'."""
    edges = [
        _make_edge(id="E-C1", status="speculative", proposed_by="worker_x",
                   **{"from": "node_a", "to": "node_b"}),
        _make_edge(id="E-C2", status="speculative", proposed_by="worker_x",
                   **{"from": "node_b", "to": "node_a"}),
    ]
    validate(edges, MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_inv5_empty_graph_passes():
    """Empty edge list is trivially a DAG."""
    validate([], MINIMAL_NODES, graph_type="seed")  # must not raise


# ── invariant 6: tested edges must have sources ───────────────────────────────


def test_inv6_tested_empty_sources_raises():
    edges = [_make_edge(status="tested", sources=[])]
    with pytest.raises(GraphIntegrityError, match="Invariant 6"):
        validate(edges, MINIMAL_NODES)


def test_inv6_tested_missing_sources_key_raises():
    edge = _make_edge(status="tested")
    del edge["sources"]
    with pytest.raises(GraphIntegrityError, match="Invariant 6"):
        validate([edge], MINIMAL_NODES)


def test_inv6_tested_with_one_source_passes():
    edges = [_make_edge(status="tested", sources=["One Source 2024"])]
    validate(edges, MINIMAL_NODES)  # must not raise


def test_inv6_speculative_with_no_sources_passes():
    """Invariant 6 applies only to tested edges; speculative edges need no sources."""
    edges = [_make_edge(status="speculative", sources=[], proposed_by="worker")]
    validate(edges, MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_inv6_rejected_with_no_sources_passes():
    """Invariant 6 applies only to tested edges; rejected edges need no sources."""
    edges = [_make_edge(status="rejected", sources=[])]
    validate(edges, MINIMAL_NODES)  # must not raise


# ── invariant 7: speculative edges must have proposed_by ─────────────────────


def test_inv7_speculative_null_proposed_by_raises():
    edges = [_make_edge(status="speculative", proposed_by=None, sources=[])]
    with pytest.raises(GraphIntegrityError, match="Invariant 7"):
        validate(edges, MINIMAL_NODES, graph_type="speculative")


def test_inv7_speculative_with_proposed_by_passes():
    edges = [_make_edge(status="speculative", proposed_by="analyst_tanaka", sources=[])]
    validate(edges, MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_inv7_tested_without_proposed_by_passes():
    """Invariant 7 applies only to speculative edges."""
    edge = _make_edge(status="tested")
    edge.pop("proposed_by", None)
    validate([edge], MINIMAL_NODES)  # must not raise


# ── happy-path integration tests ─────────────────────────────────────────────


def test_valid_graph_passes_all_invariants():
    """A well-formed multi-edge graph passes without exception."""
    nodes = deepcopy(MINIMAL_NODES)
    edges = [
        _make_edge(id="E-1", **{"from": "node_a", "to": "node_b"}, sign="+",
                   confidence="high", sources=["Source A 2023"]),
        _make_edge(id="E-2", **{"from": "node_b", "to": "node_c"}, sign="-",
                   confidence="medium", sources=["Source B 2022"]),
    ]
    validate(edges, nodes, graph_type="seed")  # must not raise


def test_mixed_statuses_pass():
    """A graph with tested, speculative, and rejected edges all passes invariants."""
    edges = [
        _make_edge(id="E-1", **{"from": "node_a", "to": "node_b"},
                   status="tested", sources=["Src"]),
        _make_edge(id="E-2", **{"from": "node_b", "to": "node_c"},
                   status="speculative", proposed_by="worker", sources=[]),
        _make_edge(id="E-3", **{"from": "node_a", "to": "node_c"},
                   status="rejected", sources=[]),
    ]
    validate(edges, MINIMAL_NODES, graph_type="speculative")  # must not raise


def test_load_nodes_returns_dict_keyed_by_id():
    result = load_nodes(FIXTURE_DIR / "test_nodes.json")
    assert isinstance(result, dict)
    assert "node_a" in result
    assert "node_b" in result
    assert "node_c" in result
    assert "node_d" in result
    assert result["node_a"]["name_en"] == "Node A"
    assert result["node_b"]["welfare_relevant"] is True


def test_load_graph_returns_edges_and_validates(tmp_path):
    nodes_path = FIXTURE_DIR / "test_nodes.json"
    graph_path = FIXTURE_DIR / "test_graph.json"
    edges = load_graph(graph_path, nodes_path, graph_type="seed")
    assert isinstance(edges, list)
    assert len(edges) == 3
    assert edges[0]["id"] == "E-T01"


def test_save_and_load_roundtrip(tmp_path):
    """save_graph then load_graph produces the same edges."""
    nodes = deepcopy(MINIMAL_NODES)
    original_edges = [
        _make_edge(id="E-1", **{"from": "node_a", "to": "node_b"}, sign="+"),
        _make_edge(id="E-2", **{"from": "node_b", "to": "node_c"}, sign="-"),
    ]

    graph_path = tmp_path / "roundtrip_graph.json"
    nodes_path = tmp_path / "roundtrip_nodes.json"

    nodes_list = list(nodes.values())
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes_list}, f)

    save_graph(original_edges, graph_path)
    loaded_edges = load_graph(graph_path, nodes_path, graph_type="seed")

    assert len(loaded_edges) == len(original_edges)
    for orig, loaded in zip(original_edges, loaded_edges):
        assert orig["id"] == loaded["id"]
        assert orig["sign"] == loaded["sign"]
        assert orig["from"] == loaded["from"]
        assert orig["to"] == loaded["to"]


def test_save_graph_creates_parent_directories(tmp_path):
    """save_graph must create parent directories if they do not exist."""
    edges = [_make_edge()]
    deep_path = tmp_path / "a" / "b" / "c" / "graph.json"
    save_graph(edges, deep_path)
    assert deep_path.exists()


# ── seed graph integration: validate actual seed_graph.json ──────────────────


def test_seed_graph_passes_all_invariants():
    """The actual seed_graph.json and nodes.json must pass all invariants with zero errors."""
    repo_root = Path(__file__).parent.parent
    nodes_path = repo_root / "knowledge" / "nodes.json"
    graph_path = repo_root / "knowledge" / "seed_graph.json"

    if not nodes_path.exists() or not graph_path.exists():
        pytest.skip("knowledge/ files not present — skipping integration test")

    edges = load_graph(graph_path, nodes_path, graph_type="seed")
    assert len(edges) == 44, f"Expected 44 edges, found {len(edges)}"

    nodes = load_nodes(nodes_path)
    assert len(nodes) == 28, f"Expected 28 nodes, found {len(nodes)}"

    for edge in edges:
        assert edge["status"] == "tested", (
            f"Edge {edge['id']} has status '{edge['status']}'; "
            f"all seed graph edges must be 'tested'"
        )
        assert edge["sources"], f"Edge {edge['id']} has no sources"
