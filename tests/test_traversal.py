"""Tests for src/traversal.py.

Covers all functions and every case listed in the task spec:
  - propagate_sign: all sign combinations, ambiguous edge, empty path
  - find_adverse_chains: welfare-relevant filtering, max_length, do-operator,
                         empty-result cases, net_sign computation, shock_sign variation
  - flag_cycle_risks: direct feedback, downstream feedback, no-feedback baseline
  - flag_non_monotonicities: non-monotone, research_gap, international_evidence,
                              combined, and clean-edge cases
  - detect_ambiguity: competing-sign paths, single sign, no-path cases
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traversal import (  # noqa: E402
    detect_ambiguity,
    find_adverse_chains,
    flag_cycle_risks,
    flag_non_monotonicities,
    propagate_sign,
)

# ── shared helpers ────────────────────────────────────────────────────────────


def _node(
    node_id: str,
    category: str = "financial",
    welfare_relevant: bool = False,
    welfare_direction: str | None = None,
) -> dict:
    node = {
        "id": node_id,
        "name_en": f"Node {node_id}",
        "name_ja": f"ノード{node_id}",
        "category": category,
        "welfare_relevant": welfare_relevant,
        "description_en": f"Test node {node_id}",
        "description_ja": f"テストノード{node_id}",
    }
    if welfare_relevant:
        node["welfare_direction"] = welfare_direction if welfare_direction is not None else "negative"
    return node


def _edge(
    edge_id: str,
    from_node: str,
    to_node: str,
    sign: str,
    *,
    monotone: bool = True,
    monotone_note: str | None = None,
    research_gap: bool = False,
    research_gap_note: str | None = None,
    international_evidence: list | None = None,
) -> dict:
    return {
        "id": edge_id,
        "from": from_node,
        "to": to_node,
        "sign": sign,
        "status": "tested",
        "confidence": "medium",
        "monotone": monotone,
        "monotone_note": monotone_note,
        "regime": "normal",
        "regime_condition": None,
        "regime_flip": None,
        "research_gap": research_gap,
        "research_gap_note": research_gap_note,
        "international_evidence": international_evidence or [],
        "sources": ["Test Source 2024"],
        "local_weight_note": None,
        "name_en": f"Node {from_node} → Node {to_node}",
        "name_ja": f"ノード{from_node} → ノード{to_node}",
        "proposed_by": "literature",
        "confirmed_by": [],
        "rejected_by": [],
        "cycle_risk": False,
        "cycle_risk_note": None,
        "last_updated": "2025-01-15",
    }


# Shared test graph used across multiple tests
#
#   upstream → src → mid ──(-)→ welf_a  [welfare]
#              │      └──(+)→ no_welf   [not welfare]
#              ├──(-)→ welf_b           [welfare]
#              └──(+)→ welf_a           [welfare, direct positive]
#   (upstream → src is an incoming edge to src — for do-operator tests)
#
# mid also has an incoming edge from itself back to src: mid → src
# (this creates a cycle in the test graph, exercising the do-operator)

NODES = {
    "src":      _node("src",      "financial",    welfare_relevant=False),
    "mid":      _node("mid",      "institutional", welfare_relevant=False),
    "welf_a":   _node("welf_a",   "real_economy", welfare_relevant=True),
    "welf_b":   _node("welf_b",   "real_economy", welfare_relevant=True),
    "no_welf":  _node("no_welf",  "behavioral",   welfare_relevant=False),
    "upstream": _node("upstream", "macro",        welfare_relevant=False),
    "distant":  _node("distant",  "real_economy", welfare_relevant=True),
}

EDGES = [
    _edge("E-1",  "src",      "mid",     "+"),   # src → mid (+)
    _edge("E-2",  "mid",      "welf_a",  "-"),   # mid → welf_a (-) → src→mid→welf_a net:-
    _edge("E-3",  "src",      "welf_b",  "-"),   # src → welf_b (-) → net:- (adverse)
    _edge("E-4",  "src",      "no_welf", "-"),   # not welfare_relevant → excluded
    _edge("E-5",  "src",      "welf_a",  "+"),   # src → welf_a (+) → net:+ with shock+
    _edge("E-6",  "mid",      "no_welf", "-"),   # not welfare_relevant → excluded
    _edge("E-7",  "upstream", "src",     "+"),   # INCOMING to src (do-operator removes this)
    _edge("E-8",  "mid",      "src",     "-"),   # INCOMING cycle edge to src (do-operator removes)
    _edge("E-9",  "src",      "distant", "+"),   # src → distant (+) → net:+ with shock+ (NOT adverse)
]


# ── propagate_sign ────────────────────────────────────────────────────────────


def test_propagate_sign_empty_path_returns_positive():
    assert propagate_sign([]) == "+"


def test_propagate_sign_single_positive():
    assert propagate_sign(["+"]) == "+"


def test_propagate_sign_single_negative():
    assert propagate_sign(["-"]) == "-"


def test_propagate_sign_two_positives():
    assert propagate_sign(["+", "+"]) == "+"


def test_propagate_sign_positive_negative():
    assert propagate_sign(["+", "-"]) == "-"


def test_propagate_sign_two_negatives():
    assert propagate_sign(["-", "-"]) == "+"


def test_propagate_sign_three_negatives():
    assert propagate_sign(["-", "-", "-"]) == "-"


def test_propagate_sign_four_negatives():
    assert propagate_sign(["-", "-", "-", "-"]) == "+"


def test_propagate_sign_mixed_long_chain():
    # + × - × + × - = + (two negatives → even)
    assert propagate_sign(["+", "-", "+", "-"]) == "+"
    # + × - × - × - = - (three negatives → odd)
    assert propagate_sign(["+", "-", "-", "-"]) == "-"


def test_propagate_sign_ambiguous_in_path():
    assert propagate_sign(["ambiguous"]) == "ambiguous"


def test_propagate_sign_ambiguous_overrides_others():
    assert propagate_sign(["+", "ambiguous", "-"]) == "ambiguous"
    assert propagate_sign(["-", "-", "ambiguous"]) == "ambiguous"


def test_propagate_sign_all_positive_long():
    assert propagate_sign(["+", "+", "+", "+"]) == "+"


# ── find_adverse_chains ───────────────────────────────────────────────────────


def test_find_adverse_chains_returns_only_welfare_nodes():
    """Terminal nodes with welfare_relevant=False must never appear."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        assert NODES[chain["terminal_node"]]["welfare_relevant"] is True, (
            f"Chain terminal '{chain['terminal_node']}' is not welfare-relevant"
        )


def test_find_adverse_chains_returns_only_negative_net_sign():
    """All returned chains must have net_sign matching the terminal node's adverse direction."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        direction = NODES[chain["terminal_node"]]["welfare_direction"]
        expected = "-" if direction == "negative" else "+"
        assert chain["net_sign"] == expected, (
            f"Chain {chain['path']} has net_sign '{chain['net_sign']}', expected '{expected}'"
        )


def test_find_adverse_chains_shock_positive_finds_correct_chains():
    """With shock '+' on src, adverse chains are: src→welf_b and src→mid→welf_a."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    paths_found = [c["path"] for c in chains]

    # src → welf_b: sign propagation = propagate_sign(["+", "-"]) = "-" ✓
    assert ["src", "welf_b"] in paths_found

    # src → mid → welf_a: propagate_sign(["+", "+", "-"]) = "-" ✓
    assert ["src", "mid", "welf_a"] in paths_found

    # src → welf_a with sign "+": propagate_sign(["+", "+"]) = "+" → NOT adverse
    assert ["src", "welf_a"] not in paths_found

    # distant has no incoming adverse path from src with shock +: + × + = +
    assert ["src", "distant"] not in paths_found


def test_find_adverse_chains_shock_negative_finds_correct_chains():
    """With shock '-' on src, adverse chains flip: src→welf_a becomes adverse."""
    chains = find_adverse_chains("src", "-", EDGES, NODES)
    paths_found = [c["path"] for c in chains]

    # src → welf_a with sign "+": propagate_sign(["-", "+"]) = "-" → ADVERSE with shock -
    assert ["src", "welf_a"] in paths_found

    # src → welf_b with sign "-": propagate_sign(["-", "-"]) = "+" → NOT adverse
    assert ["src", "welf_b"] not in paths_found

    # src → mid → welf_a: propagate_sign(["-", "+", "-"]) = "+" → NOT adverse
    assert ["src", "mid", "welf_a"] not in paths_found


def test_find_adverse_chains_respects_max_length_one():
    """max_length=1 returns only length-1 chains."""
    chains = find_adverse_chains("src", "+", EDGES, NODES, max_length=1)
    for chain in chains:
        assert chain["chain_length"] == 1

    # Only length-1 adverse chain: src → welf_b (net "-")
    paths_found = [c["path"] for c in chains]
    assert ["src", "welf_b"] in paths_found
    # Length-2 chain src → mid → welf_a must NOT appear
    assert ["src", "mid", "welf_a"] not in paths_found


def test_find_adverse_chains_respects_max_length_two():
    """max_length=2 returns chains of length 1 and 2, not longer."""
    chains = find_adverse_chains("src", "+", EDGES, NODES, max_length=2)
    for chain in chains:
        assert chain["chain_length"] <= 2


def test_find_adverse_chains_returns_empty_when_no_adverse_path():
    """No adverse chains when decision node has no outgoing paths to welfare nodes."""
    simple_nodes = {
        "x": _node("x", "financial", welfare_relevant=False),
        "y": _node("y", "behavioral", welfare_relevant=False),
    }
    simple_edges = [_edge("E-1", "x", "y", "-")]
    chains = find_adverse_chains("x", "+", simple_edges, simple_nodes)
    assert chains == []


def test_find_adverse_chains_returns_empty_for_disconnected_decision_node():
    """Decision node with no outgoing edges produces empty list."""
    chains = find_adverse_chains("welf_a", "+", EDGES, NODES)
    assert chains == []


def test_find_adverse_chains_returns_empty_for_unknown_decision_node():
    """Unknown decision node (not in graph) produces empty list."""
    chains = find_adverse_chains("nonexistent_node", "+", EDGES, NODES)
    assert chains == []


def test_find_adverse_chains_chain_contains_required_fields():
    """Every returned chain must have all required keys."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    assert chains, "Expected at least one adverse chain"
    required_keys = {"path", "signs", "net_sign", "edge_objects", "terminal_node", "chain_length"}
    for chain in chains:
        assert required_keys <= chain.keys(), (
            f"Chain is missing keys: {required_keys - chain.keys()}"
        )


def test_find_adverse_chains_path_length_matches_chain_length():
    """chain_length must equal len(path) - 1 == len(signs) == len(edge_objects)."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        assert chain["chain_length"] == len(chain["path"]) - 1
        assert chain["chain_length"] == len(chain["signs"])
        assert chain["chain_length"] == len(chain["edge_objects"])


def test_find_adverse_chains_terminal_node_matches_path_end():
    """terminal_node must be the last element of path."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        assert chain["terminal_node"] == chain["path"][-1]


def test_find_adverse_chains_path_starts_at_decision_node():
    """Every chain path must start at the decision node."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        assert chain["path"][0] == "src"


def test_find_adverse_chains_do_operator_removes_incoming_edges():
    """Incoming edges to decision_node are excluded; valid outgoing chains are still found."""
    # E-7 (upstream → src) and E-8 (mid → src) are incoming to src.
    # The do-operator must remove them. Outgoing paths from src must still be found.
    chains = find_adverse_chains("src", "+", EDGES, NODES)

    # Verify adverse chains ARE found (do-operator did not break outgoing traversal)
    assert any(c["path"] == ["src", "welf_b"] for c in chains)
    assert any(c["path"] == ["src", "mid", "welf_a"] for c in chains)

    # Verify upstream never appears as a non-start node (it feeds INTO src, not the other way)
    for chain in chains:
        assert "upstream" not in chain["path"], (
            "'upstream' should not appear in chains starting from 'src'"
        )


def test_find_adverse_chains_do_operator_with_cycle_edge_still_finds_chains():
    """With a cycle edge mid→src present, correct chains are still found exactly once."""
    # E-8: mid → src creates a graph cycle. The do-operator removes it.
    # Without simple-path + do-operator, the cycle could cause double-counting.
    chains = find_adverse_chains("src", "+", EDGES, NODES)

    # src → mid → welf_a should appear exactly once
    matching = [c for c in chains if c["path"] == ["src", "mid", "welf_a"]]
    assert len(matching) == 1, "src→mid→welf_a should appear exactly once"


def test_find_adverse_chains_edge_objects_match_path():
    """edge_objects[i].from == path[i] and edge_objects[i].to == path[i+1]."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        for i, edge in enumerate(chain["edge_objects"]):
            assert edge["from"] == chain["path"][i]
            assert edge["to"] == chain["path"][i + 1]


def test_find_adverse_chains_net_sign_consistent_with_propagate_sign():
    """net_sign must equal propagate_sign([shock_sign] + signs)."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        expected = propagate_sign(["+" ] + chain["signs"])
        assert chain["net_sign"] == expected


def test_find_adverse_chains_simple_path_no_repeated_nodes():
    """No node should appear more than once in any returned chain path."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    for chain in chains:
        assert len(chain["path"]) == len(set(chain["path"])), (
            f"Cycle in path: {chain['path']}"
        )


def test_find_adverse_chains_secondary_nodes_reach_welfare_when_primary_cannot():
    """Chain is found via secondary_node when primary_node has no outgoing edges."""
    sec_nodes = {
        "primary":   _node("primary",   "financial",   welfare_relevant=False),
        "secondary": _node("secondary", "financial",   welfare_relevant=False),
        "harm":      _node("harm",      "real_economy", welfare_relevant=True),
    }
    sec_edges = [
        _edge("E-1", "secondary", "harm", "-"),  # shock+ × - = "-" → adverse
        # primary has no outgoing edges
    ]
    chains = find_adverse_chains(
        "primary", "+", sec_edges, sec_nodes, secondary_nodes=["secondary"]
    )
    adverse = [c for c in chains if c.get("chain_type") == "adverse"]
    assert len(adverse) == 1
    assert adverse[0]["path"] == ["secondary", "harm"]


def test_find_adverse_chains_secondary_nodes_deduplication():
    """Primary and secondary both reach same welfare node — each path appears exactly once."""
    ded_nodes = {
        "primary":   _node("primary",   "financial",   welfare_relevant=False),
        "secondary": _node("secondary", "financial",   welfare_relevant=False),
        "harm":      _node("harm",      "real_economy", welfare_relevant=True),
    }
    ded_edges = [
        _edge("E-1", "primary",   "harm", "-"),  # shock+ × - = "-" → adverse
        _edge("E-2", "secondary", "harm", "-"),  # shock+ × - = "-" → adverse
    ]
    chains = find_adverse_chains(
        "primary", "+", ded_edges, ded_nodes, secondary_nodes=["secondary"]
    )
    adverse = [c for c in chains if c.get("chain_type") == "adverse"]
    paths = [tuple(c["path"]) for c in adverse]
    assert paths.count(("primary", "harm")) == 1
    assert paths.count(("secondary", "harm")) == 1
    assert len(adverse) == 2


# ── flag_cycle_risks ──────────────────────────────────────────────────────────


def _make_chain(path: list[str], signs: list[str]) -> dict:
    """Build a minimal chain dict for flag_cycle_risks and flag_non_monotonicities tests."""
    edge_objects = [
        _edge(f"E-{i}", path[i], path[i + 1], signs[i])
        for i in range(len(signs))
    ]
    return {
        "path": path,
        "signs": signs,
        "net_sign": propagate_sign(signs),
        "edge_objects": edge_objects,
        "terminal_node": path[-1],
        "chain_length": len(signs),
    }


def test_flag_cycle_risks_detects_direct_feedback_from_terminal():
    """Terminal node with edge back to a path node → cycle_risk=True."""
    chain = _make_chain(["src", "mid", "welf_a"], ["+", "-"])
    extra_edges = [
        _edge("E-back", "welf_a", "src", "+"),  # terminal → path node
    ]
    result = flag_cycle_risks(chain, extra_edges)
    assert result["cycle_risk"] is True
    assert "welf_a" in result["cycle_risk_note"]
    assert "src" in result["cycle_risk_note"]


def test_flag_cycle_risks_detects_downstream_feedback():
    """A node downstream of terminal with edge back into chain → cycle_risk=True."""
    # welf_a → extra_node → mid (mid is in chain path)
    chain = _make_chain(["src", "mid", "welf_a"], ["+", "-"])
    extra_edges = [
        _edge("E-d1", "welf_a", "extra", "+"),
        _edge("E-d2", "extra", "mid", "+"),   # downstream node → chain path
    ]
    result = flag_cycle_risks(chain, extra_edges)
    assert result["cycle_risk"] is True


def test_flag_cycle_risks_no_feedback_returns_false():
    """No edges from terminal or downstream back into chain → cycle_risk=False."""
    chain = _make_chain(["src", "mid", "welf_a"], ["+", "-"])
    unrelated_edges = [
        _edge("E-u", "welf_a", "elsewhere", "+"),  # goes somewhere outside path
    ]
    result = flag_cycle_risks(chain, unrelated_edges)
    assert result["cycle_risk"] is False
    assert result["cycle_risk_note"] is None


def test_flag_cycle_risks_empty_graph_returns_false():
    """Empty graph_edges → no feedback possible → cycle_risk=False."""
    chain = _make_chain(["src", "welf_a"], ["-"])
    result = flag_cycle_risks(chain, [])
    assert result["cycle_risk"] is False


def test_flag_cycle_risks_preserves_original_chain_fields():
    """Result must contain all fields from original chain plus cycle_risk fields."""
    chain = _make_chain(["src", "welf_a"], ["-"])
    chain["extra_field"] = "should_be_preserved"
    result = flag_cycle_risks(chain, [])
    assert result["extra_field"] == "should_be_preserved"
    assert "cycle_risk" in result


def test_flag_cycle_risks_does_not_mutate_input():
    """Input chain dict must not be modified."""
    chain = _make_chain(["src", "welf_a"], ["-"])
    original_keys = set(chain.keys())
    flag_cycle_risks(chain, [])
    assert set(chain.keys()) == original_keys


def test_flag_cycle_risks_feedback_only_to_path_node_not_elsewhere():
    """Edges from terminal to nodes NOT in path must not trigger cycle_risk."""
    chain = _make_chain(["src", "mid", "welf_a"], ["+", "-"])
    edges_outside_path = [
        _edge("E-out", "welf_a", "completely_different_node", "+"),
    ]
    result = flag_cycle_risks(chain, edges_outside_path)
    assert result["cycle_risk"] is False


# ── flag_non_monotonicities ───────────────────────────────────────────────────


def _chain_with_edges(edge_list: list[dict]) -> dict:
    """Build a chain dict wrapping custom edge objects."""
    path = [edge_list[0]["from"]] + [e["to"] for e in edge_list]
    signs = [e["sign"] for e in edge_list]
    return {
        "path": path,
        "signs": signs,
        "net_sign": propagate_sign(signs),
        "edge_objects": edge_list,
        "terminal_node": path[-1],
        "chain_length": len(edge_list),
    }


def test_flag_non_monotonicities_flags_non_monotone_edge():
    """Non-monotone edge produces a non_monotone warning."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "-", monotone=False,
              monotone_note="Sign depends on regime"),
    ])
    result = flag_non_monotonicities(chain)
    types = [w["type"] for w in result["warnings"]]
    assert "non_monotone" in types


def test_flag_non_monotonicities_flags_research_gap_edge():
    """Research-gap edge produces a research_gap warning."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "-", research_gap=True,
              research_gap_note="Understudied in Japan"),
    ])
    result = flag_non_monotonicities(chain)
    types = [w["type"] for w in result["warnings"]]
    assert "research_gap" in types


def test_flag_non_monotonicities_flags_international_evidence():
    """Edge with non-empty international_evidence produces an international_evidence warning."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "+",
              international_evidence=["European Banking Study 2020"]),
    ])
    result = flag_non_monotonicities(chain)
    types = [w["type"] for w in result["warnings"]]
    assert "international_evidence" in types


def test_flag_non_monotonicities_both_flags_on_same_edge():
    """An edge that is both non-monotone AND has a research gap produces two warnings."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "-", monotone=False, research_gap=True),
    ])
    result = flag_non_monotonicities(chain)
    assert len(result["warnings"]) == 2
    types = {w["type"] for w in result["warnings"]}
    assert "non_monotone" in types
    assert "research_gap" in types


def test_flag_non_monotonicities_clean_chain_no_warnings():
    """Monotone edge with no research gap produces no warnings."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "+"),
    ])
    result = flag_non_monotonicities(chain)
    assert result["warnings"] == []


def test_flag_non_monotonicities_warning_contains_edge_id():
    """Each warning must include the edge_id field."""
    chain = _chain_with_edges([
        _edge("E-XYZ", "a", "b", "-", monotone=False),
    ])
    result = flag_non_monotonicities(chain)
    assert result["warnings"]
    for warning in result["warnings"]:
        assert "edge_id" in warning
        assert warning["edge_id"] == "E-XYZ"


def test_flag_non_monotonicities_multiple_edges_each_flagged():
    """All problematic edges in chain are flagged independently."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "+", monotone=False),
        _edge("E-2", "b", "c", "-", research_gap=True),
    ])
    result = flag_non_monotonicities(chain)
    ids = [w["edge_id"] for w in result["warnings"]]
    assert "E-1" in ids
    assert "E-2" in ids


def test_flag_non_monotonicities_preserves_existing_warnings():
    """Existing warnings in chain are not discarded."""
    chain = _chain_with_edges([
        _edge("E-1", "a", "b", "-", monotone=False),
    ])
    chain["warnings"] = [{"type": "pre_existing", "message": "keep me"}]
    result = flag_non_monotonicities(chain)
    types = [w["type"] for w in result["warnings"]]
    assert "pre_existing" in types
    assert "non_monotone" in types


def test_flag_non_monotonicities_does_not_mutate_input():
    """Input chain dict must not be modified."""
    chain = _chain_with_edges([_edge("E-1", "a", "b", "-", monotone=False)])
    assert "warnings" not in chain
    flag_non_monotonicities(chain)
    assert "warnings" not in chain


# ── detect_ambiguity ─────────────────────────────────────────────────────────


def _ambig_nodes():
    return {
        "src":    _node("src",    "financial",   welfare_relevant=False),
        "mid":    _node("mid",    "institutional", welfare_relevant=False),
        "target": _node("target", "real_economy", welfare_relevant=True),
        "alt":    _node("alt",    "macro",        welfare_relevant=False),
    }


def test_detect_ambiguity_two_paths_opposite_sign():
    """Two paths from src to target with opposite net_sign → ambiguous=True."""
    edges = [
        _edge("E-1", "src", "target",  "+"),  # direct: + × + = + (positive)
        _edge("E-2", "src", "mid",     "+"),
        _edge("E-3", "mid", "target",  "-"),  # via mid: + × + × - = - (negative)
    ]
    result = detect_ambiguity("src", "+", "target", edges)
    assert result["ambiguous"] is True
    assert "positive_paths" in result
    assert "negative_paths" in result
    assert len(result["positive_paths"]) >= 1
    assert len(result["negative_paths"]) >= 1


def test_detect_ambiguity_single_path_not_ambiguous():
    """Single path to target → ambiguous=False with correct sign."""
    edges = [
        _edge("E-1", "src", "target", "-"),
    ]
    result = detect_ambiguity("src", "+", "target", edges)
    assert result["ambiguous"] is False
    # net_sign = propagate_sign(["+", "-"]) = "-"
    assert result["sign"] == "-"
    assert len(result["paths"]) == 1


def test_detect_ambiguity_multiple_same_sign_not_ambiguous():
    """Two paths both with net_sign "+" → ambiguous=False, sign="+"."""
    edges = [
        _edge("E-1", "src", "target", "+"),          # + × + = +
        _edge("E-2", "src", "alt",    "+"),
        _edge("E-3", "alt", "target", "+"),           # + × + × + = +
    ]
    result = detect_ambiguity("src", "+", "target", edges)
    assert result["ambiguous"] is False
    assert result["sign"] == "+"
    assert len(result["paths"]) == 2


def test_detect_ambiguity_no_paths_returns_false_none():
    """No path from src to target → ambiguous=False, sign=None, paths=[]."""
    edges = [_edge("E-1", "src", "alt", "+")]  # alt != target
    result = detect_ambiguity("src", "+", "target", edges)
    assert result["ambiguous"] is False
    assert result["sign"] is None
    assert result["paths"] == []


def test_detect_ambiguity_unreachable_target():
    """Target completely disconnected → ambiguous=False, sign=None."""
    result = detect_ambiguity("src", "+", "target", [])
    assert result["ambiguous"] is False
    assert result["sign"] is None


def test_detect_ambiguity_decision_equals_target():
    """If decision_node == target_node, no paths (requires at least one edge)."""
    result = detect_ambiguity("src", "+", "src", EDGES)
    assert result["ambiguous"] is False
    assert result["sign"] is None


def test_detect_ambiguity_paths_contain_required_fields():
    """Each path in the result must contain path, edge_signs, and net_sign."""
    edges = [_edge("E-1", "src", "target", "-")]
    result = detect_ambiguity("src", "+", "target", edges)
    for path in result["paths"]:
        assert "path" in path
        assert "edge_signs" in path
        assert "net_sign" in path


def test_detect_ambiguity_negative_shock_flips_paths():
    """Shock sign "-" inverts net signs on all paths."""
    edges = [
        _edge("E-1", "src", "target", "+"),  # shock "-" + "+" → net "-"
        _edge("E-2", "src", "mid",    "+"),
        _edge("E-3", "mid", "target", "-"),  # shock "-" + "+" + "-" → net "+"
    ]
    result = detect_ambiguity("src", "-", "target", edges)
    # First path: propagate_sign(["-", "+"]) = "-"  (negative)
    # Second path: propagate_sign(["-", "+", "-"]) = "+"  (positive)
    assert result["ambiguous"] is True
    assert any(p["net_sign"] == "-" for p in result.get("negative_paths", []))
    assert any(p["net_sign"] == "+" for p in result.get("positive_paths", []))


def test_detect_ambiguity_ambiguous_edge_propagates():
    """An 'ambiguous' edge sign produces net_sign 'ambiguous' for that path."""
    edges = [
        _edge("E-1", "src", "target", "ambiguous"),
        _edge("E-2", "src", "mid",    "+"),
        _edge("E-3", "mid", "target", "-"),
    ]
    result = detect_ambiguity("src", "+", "target", edges)
    # ambiguous path: propagate_sign(["+", "ambiguous"]) = "ambiguous"
    # negative path: propagate_sign(["+", "+", "-"]) = "-"
    # Both present but no opposite +/- conflict (only - and ambiguous)
    # → ambiguous=False because there are no positive_paths to conflict with negative_paths
    all_paths = result.get("paths", result.get("positive_paths", []) + result.get("negative_paths", []))
    net_signs = {p["net_sign"] for p in all_paths}
    assert "ambiguous" in net_signs


# ── integration: traversal on actual seed graph ───────────────────────────────


def test_traversal_on_seed_graph_bank_npl_shock():
    """Integration test: shock to bank_npl_ratio produces adverse chains in seed graph."""
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from graph_io import load_nodes, load_graph

    repo_root = Path(__file__).parent.parent
    nodes_path = repo_root / "knowledge" / "nodes.json"
    graph_path = repo_root / "knowledge" / "seed_graph.json"

    if not nodes_path.exists() or not graph_path.exists():
        pytest.skip("knowledge/ files not present — skipping integration test")

    nodes = load_nodes(nodes_path)
    edges = load_graph(graph_path, nodes_path, graph_type="seed")

    # bank_npl_ratio increases (+) → should produce adverse chains
    chains = find_adverse_chains("bank_npl_ratio", "+", edges, nodes, max_length=4)

    assert chains, "Expected at least one adverse chain from bank_npl_ratio shock"

    for chain in chains:
        direction = nodes[chain["terminal_node"]]["welfare_direction"]
        expected = "-" if direction == "negative" else "+"
        assert chain["net_sign"] == expected
        assert nodes[chain["terminal_node"]]["welfare_relevant"] is True
        assert chain["chain_length"] >= 1
        assert chain["chain_length"] <= 4
        assert chain["path"][0] == "bank_npl_ratio"


# ── feedback loop detection ───────────────────────────────────────────────────

# Graph with a cycle not involving the decision node:
#   loop_a (decision) → loop_b → loop_c → loop_b  (cycle: c → b)
#   loop_a → loop_harm                              (adverse chain: welfare)

LOOP_NODES = {
    "loop_a":    _node("loop_a",    "financial",    welfare_relevant=False),
    "loop_b":    _node("loop_b",    "institutional", welfare_relevant=False),
    "loop_c":    _node("loop_c",    "real_economy", welfare_relevant=False),
    "loop_harm": _node("loop_harm", "real_economy", welfare_relevant=True),
}

LOOP_EDGES = [
    _edge("LE-1", "loop_a", "loop_b",    "+"),   # a → b (+)
    _edge("LE-2", "loop_b", "loop_c",    "-"),   # b → c (-)
    _edge("LE-3", "loop_c", "loop_b",    "+"),   # c → b (+) — revisits b
    _edge("LE-4", "loop_a", "loop_harm", "-"),   # a → harm (-) — adverse chain
]


def test_find_adverse_chains_feedback_loop_chain_type():
    """Chains that revisit a path node must have chain_type='feedback_loop'."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains, "Expected at least one feedback_loop chain"


def test_find_adverse_chains_feedback_loop_terminal_marker_structure():
    """The last element of a feedback_loop chain's path must be the loop marker dict."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains, "Expected at least one feedback_loop chain"
    for chain in loop_chains:
        marker = chain["path"][-1]
        assert isinstance(marker, dict), "Terminal marker must be a dict"
        assert marker["node_id"] == "__feedback_loop__"
        assert "loop_target" in marker
        assert marker["sign"] is None


def test_find_adverse_chains_feedback_loop_terminal_node_field():
    """terminal_node for a feedback_loop chain must be the sentinel string."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains
    for chain in loop_chains:
        assert chain["terminal_node"] == "__feedback_loop__"


def test_find_adverse_chains_feedback_loop_loop_target_field():
    """loop_target must be the node that would have been revisited."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains
    for chain in loop_chains:
        assert chain["loop_target"] == "loop_b"
        assert chain["path"][-1]["loop_target"] == "loop_b"


def test_find_adverse_chains_feedback_loop_stops_before_revisit():
    """String nodes in a feedback_loop chain path must appear at most once."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains
    for chain in loop_chains:
        string_nodes = [n for n in chain["path"] if isinstance(n, str)]
        assert len(string_nodes) == len(set(string_nodes)), (
            f"Repeated string nodes in feedback loop path: {string_nodes}"
        )


def test_find_adverse_chains_feedback_loop_note_ja():
    """Feedback loop chains must carry the Japanese note field."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains
    for chain in loop_chains:
        assert "feedback_loop_note_ja" in chain
        assert "フィードバックループ" in chain["feedback_loop_note_ja"]


def test_find_adverse_chains_adverse_chains_have_chain_type():
    """Adverse chains must have chain_type='adverse'."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    adverse_chains = [c for c in chains if c.get("chain_type") == "adverse"]
    assert adverse_chains, "Expected at least one adverse chain"
    for chain in adverse_chains:
        assert chain["chain_type"] == "adverse"
        # Verify it's a real adverse chain: terminal is welfare_relevant
        assert LOOP_NODES[chain["terminal_node"]]["welfare_relevant"] is True


def test_find_adverse_chains_feedback_loop_does_not_produce_adverse_chain():
    """The loop path itself must not appear as an adverse chain."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    adverse_paths = [c["path"] for c in chains if c.get("chain_type") == "adverse"]
    # The loop path [loop_a, loop_b, loop_c, {marker}] must not be among adverse paths
    for path in adverse_paths:
        assert not any(isinstance(n, dict) for n in path), (
            f"Adverse chain path contains loop marker: {path}"
        )


def test_find_adverse_chains_feedback_loop_path_starts_at_decision_node():
    """Feedback loop chains must start at the decision node."""
    chains = find_adverse_chains("loop_a", "+", LOOP_EDGES, LOOP_NODES)
    for chain in chains:
        assert chain["path"][0] == "loop_a"


def test_find_adverse_chains_no_feedback_loops_in_acyclic_graph():
    """No feedback_loop chains returned when graph has no back edges."""
    chains = find_adverse_chains("src", "+", EDGES, NODES)
    loop_chains = [c for c in chains if c.get("chain_type") == "feedback_loop"]
    assert loop_chains == [], "Acyclic test graph must produce no feedback_loop chains"


def test_propagate_sign_canonical_matches_spec():
    """The exact function body from CLAUDE.md section 5 must be used — verify by exhaustive check."""
    # Mirror the spec exactly
    def spec_propagate_sign(path_signs: list[str]) -> str:
        if "ambiguous" in path_signs:
            return "ambiguous"
        negatives = path_signs.count("-")
        return "-" if negatives % 2 == 1 else "+"

    test_cases = [
        [], ["+"], ["-"], ["+", "+"], ["+", "-"], ["-", "-"],
        ["-", "-", "-"], ["ambiguous"], ["+", "ambiguous", "-"],
        ["+", "+", "-", "-"], ["-", "-", "-", "-", "-"],
    ]
    for case in test_cases:
        assert propagate_sign(case) == spec_propagate_sign(case), (
            f"propagate_sign({case}) diverges from spec"
        )
