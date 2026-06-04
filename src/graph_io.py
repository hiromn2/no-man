"""Load, save, and validate causal graph JSON files for No-Man.

Enforces all seven graph invariants from CLAUDE.md section 3.
Also validates optional fields added in schema v1.2 (confidence_tier, audit fields)
when present. Raises GraphIntegrityError on any violation — never silently continues.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

VALID_SIGNS = {"+", "-", "ambiguous"}
VALID_STATUSES = {"tested", "speculative", "rejected", "proposed"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_CONFIDENCE_TIERS = {1, 2, 3, 4, 5}

# Optional audit fields added in schema v1.2. All are nullable; missing = valid.
_AUDIT_STRING_FIELDS = ("proposed_at", "basis", "verified_by", "verified_at")


class GraphIntegrityError(Exception):
    """Raised when any graph invariant is violated."""


# ── public API ────────────────────────────────────────────────────────────────


def load_nodes(nodes_path: Union[str, Path]) -> dict[str, dict]:
    """Load nodes.json and return a dict keyed by node id.

    Expected format: {"nodes": [...]} where each element has at minimum an "id" field.
    """
    path = Path(nodes_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("nodes", data)
    if isinstance(raw, list):
        return {n["id"]: n for n in raw}
    if isinstance(raw, dict):
        return raw
    raise GraphIntegrityError(
        f"nodes file {path} must contain a list under key 'nodes' or a dict of node objects"
    )


def load_graph(
    graph_path: Union[str, Path],
    nodes_path: Union[str, Path],
    graph_type: str = "seed",
) -> list[dict]:
    """Load a graph JSON file, validate all invariants, and return the edge list.

    Args:
        graph_path: Path to the graph file (seed_graph.json or speculative_graph.json).
        nodes_path: Path to nodes.json for node-reference validation.
        graph_type: "seed" or "speculative". Only the seed graph is checked for cycles
                    (invariant 5). Defaults to "seed" (most conservative).

    Returns:
        List of edge dicts, validated against all applicable invariants.

    Raises:
        GraphIntegrityError: If any invariant is violated. Halts immediately.
    """
    path = Path(graph_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", data)
    if not isinstance(edges, list):
        raise GraphIntegrityError(
            f"Graph file {path} must contain a list of edges under key 'edges'"
        )

    nodes = load_nodes(nodes_path)
    validate(edges, nodes, graph_type=graph_type)
    return edges


def save_graph(edges: list[dict], graph_path: Union[str, Path]) -> None:
    """Serialize edges to a graph JSON file.

    Creates parent directories if they do not exist.
    Does NOT validate before saving — callers should validate first.
    """
    path = Path(graph_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"edges": edges}, f, ensure_ascii=False, indent=2)


def validate(
    edges: list[dict],
    nodes: dict[str, dict],
    graph_type: str = "seed",
) -> None:
    """Validate all graph invariants. Raises GraphIntegrityError on first violation.

    Invariants enforced (from CLAUDE.md section 3):
      1. Every edge 'from' and 'to' node exists in nodes
      2. No edge has sign other than '+', '-', or 'ambiguous'
      3. No edge has status other than 'tested', 'speculative', or 'rejected'
      4. No edge has confidence other than 'high', 'medium', or 'low'
      5. The tested graph (seed) contains no cycles  [seed only]
      6. Every tested edge has at least one entry in sources
      7. Every speculative edge has a non-null proposed_by

    Optional field checks (do not raise on missing; raise on invalid value):
      confidence_tier: if present, must be integer in {1, 2, 3, 4, 5}
      audit fields (proposed_at, basis, verified_by, verified_at): if non-null, must be str
      governance fields (dispute_flags, adversarial_result, review_required, review_reason):
        if present, must match expected types

    Args:
        edges: List of edge dicts to validate.
        nodes: Dict of node_id → node_dict (from load_nodes).
        graph_type: "seed" triggers the DAG cycle check (invariant 5).
    """
    _check_node_references(edges, nodes)
    _check_sign_values(edges)
    _check_status_values(edges)
    _check_confidence_values(edges)
    if graph_type == "seed":
        _check_no_cycles(edges)
    _check_tested_sources(edges)
    _check_speculative_proposed_by(edges)
    _check_confidence_tier_values(edges)
    _check_audit_field_types(edges)
    _check_governance_field_types(edges)


# ── invariant checkers ────────────────────────────────────────────────────────


def _check_node_references(edges: list[dict], nodes: dict[str, dict]) -> None:
    """Invariant 1: every edge from/to node must exist in nodes."""
    for edge in edges:
        edge_id = edge.get("id", "<unknown>")
        for field in ("from", "to"):
            node_id = edge.get(field)
            if node_id not in nodes:
                raise GraphIntegrityError(
                    f"Invariant 1 violated: edge '{edge_id}' references unknown node "
                    f"'{node_id}' in field '{field}'. "
                    f"Add the node to nodes.json before using it in an edge."
                )


def _check_sign_values(edges: list[dict]) -> None:
    """Invariant 2: sign must be one of '+', '-', 'ambiguous'."""
    for edge in edges:
        sign = edge.get("sign")
        if sign not in VALID_SIGNS:
            raise GraphIntegrityError(
                f"Invariant 2 violated: edge '{edge.get('id', '?')}' has invalid sign "
                f"'{sign}'. Must be one of {sorted(VALID_SIGNS)}."
            )


def _check_status_values(edges: list[dict]) -> None:
    """Invariant 3: status must be one of 'tested', 'speculative', 'rejected'."""
    for edge in edges:
        status = edge.get("status")
        if status not in VALID_STATUSES:
            raise GraphIntegrityError(
                f"Invariant 3 violated: edge '{edge.get('id', '?')}' has invalid status "
                f"'{status}'. Must be one of {sorted(VALID_STATUSES)}."
            )


def _check_confidence_values(edges: list[dict]) -> None:
    """Invariant 4: confidence must be one of 'high', 'medium', 'low'."""
    for edge in edges:
        confidence = edge.get("confidence")
        if confidence not in VALID_CONFIDENCES:
            raise GraphIntegrityError(
                f"Invariant 4 violated: edge '{edge.get('id', '?')}' has invalid confidence "
                f"'{confidence}'. Must be one of {sorted(VALID_CONFIDENCES)}."
            )


def _check_no_cycles(edges: list[dict]) -> None:
    """Invariant 5: the graph must be a DAG (no directed cycles).

    Uses Kahn's algorithm (topological sort via BFS on in-degrees).
    If the sort cannot process all nodes, a cycle exists.
    """
    node_ids: set[str] = set()
    for edge in edges:
        node_ids.add(edge["from"])
        node_ids.add(edge["to"])

    if not node_ids:
        return

    adj: dict[str, list[str]] = {n: [] for n in node_ids}
    in_degree: dict[str, int] = {n: 0 for n in node_ids}

    seen_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        src, dst = edge["from"], edge["to"]
        pair = (src, dst)
        if pair not in seen_pairs:
            adj[src].append(dst)
            in_degree[dst] += 1
            seen_pairs.add(pair)

    queue = [n for n in node_ids if in_degree[n] == 0]
    visited_count = 0

    while queue:
        node = queue.pop()
        visited_count += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited_count != len(node_ids):
        cycle_nodes = [n for n in node_ids if in_degree[n] > 0]
        raise GraphIntegrityError(
            f"Invariant 5 violated: the seed graph contains a directed cycle. "
            f"Topological sort processed {visited_count}/{len(node_ids)} nodes. "
            f"Nodes still in cycle (non-zero in-degree): {sorted(cycle_nodes)}. "
            f"The seed graph must be a DAG. Remove or redirect the cycle-forming edge."
        )


def _check_tested_sources(edges: list[dict]) -> None:
    """Invariant 6: every tested edge must have at least one source."""
    for edge in edges:
        if edge.get("status") == "tested":
            sources = edge.get("sources", [])
            if not sources:
                raise GraphIntegrityError(
                    f"Invariant 6 violated: tested edge '{edge.get('id', '?')}' "
                    f"({edge.get('from', '?')} → {edge.get('to', '?')}) has no sources. "
                    f"Every tested edge must cite at least one academic source."
                )


def _check_speculative_proposed_by(edges: list[dict]) -> None:
    """Invariant 7: every speculative or proposed edge must have a non-null proposed_by."""
    for edge in edges:
        if edge.get("status") in ("speculative", "proposed"):
            proposed_by = edge.get("proposed_by")
            if proposed_by is None:
                raise GraphIntegrityError(
                    f"Invariant 7 violated: {edge.get('status')} edge '{edge.get('id', '?')}' "
                    f"({edge.get('from', '?')} → {edge.get('to', '?')}) has null proposed_by. "
                    f"Every speculative and proposed edge must record who proposed it."
                )


def _check_confidence_tier_values(edges: list[dict]) -> None:
    """Optional field: if confidence_tier is present it must be an integer in {1,2,3,4,5}."""
    for edge in edges:
        tier = edge.get("confidence_tier")
        if tier is None:
            continue
        if tier not in VALID_CONFIDENCE_TIERS:
            raise GraphIntegrityError(
                f"confidence_tier invalid on edge '{edge.get('id', '?')}': "
                f"got {tier!r}, expected one of {sorted(VALID_CONFIDENCE_TIERS)}. "
                "See governance.json confidence_tier_schema for the 5-tier definition."
            )


def _check_audit_field_types(edges: list[dict]) -> None:
    """Optional audit fields: if non-null, proposed_at/basis/verified_by/verified_at must be str."""
    for edge in edges:
        edge_id = edge.get("id", "?")
        for field in _AUDIT_STRING_FIELDS:
            value = edge.get(field)
            if value is not None and not isinstance(value, str):
                raise GraphIntegrityError(
                    f"Audit field '{field}' on edge '{edge_id}' must be a string or null; "
                    f"got {type(value).__name__!r} ({value!r})."
                )


def _check_governance_field_types(edges: list[dict]) -> None:
    """Optional governance fields added for the feedback loop (schema v1.3).

    dispute_flags:     if present, must be a list
    adversarial_result: if present and non-null, must be a dict
    review_required:   if present and non-null, must be a bool
    review_reason:     if present and non-null, must be a str
    """
    for edge in edges:
        eid = edge.get("id", "?")

        flags = edge.get("dispute_flags")
        if flags is not None and not isinstance(flags, list):
            raise GraphIntegrityError(
                f"dispute_flags on edge '{eid}' must be a list or null; "
                f"got {type(flags).__name__!r}."
            )

        adv = edge.get("adversarial_result")
        if adv is not None and not isinstance(adv, dict):
            raise GraphIntegrityError(
                f"adversarial_result on edge '{eid}' must be a dict or null; "
                f"got {type(adv).__name__!r}."
            )

        rr = edge.get("review_required")
        if rr is not None and not isinstance(rr, bool):
            raise GraphIntegrityError(
                f"review_required on edge '{eid}' must be bool or null; "
                f"got {type(rr).__name__!r}."
            )

        rreason = edge.get("review_reason")
        if rreason is not None and not isinstance(rreason, str):
            raise GraphIntegrityError(
                f"review_reason on edge '{eid}' must be a string or null; "
                f"got {type(rreason).__name__!r}."
            )


def check_cross_graph_uniqueness(
    edges_a: list[dict],
    edges_b: list[dict],
) -> None:
    """Raise GraphIntegrityError if any edge ID appears in both edge lists.

    Call this when working with both seed_graph.json and speculative_graph.json
    simultaneously to enforce global edge ID uniqueness.
    """
    ids_a = {e.get("id") for e in edges_a if e.get("id")}
    ids_b = {e.get("id") for e in edges_b if e.get("id")}
    overlap = ids_a & ids_b
    if overlap:
        raise GraphIntegrityError(
            f"Cross-graph ID uniqueness violation: edge ID(s) {sorted(overlap)} "
            "appear in both graph files. Each edge ID must be globally unique "
            "across seed_graph.json and speculative_graph.json."
        )
