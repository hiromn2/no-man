"""Core path enumeration, sign propagation, and chain analysis for No-Man.

Implements the causal graph traversal and annotation functions defined in
CLAUDE.md section 5. All functions are pure (no side effects, no mutation
of input arguments).
"""

from __future__ import annotations


# ── sign propagation ──────────────────────────────────────────────────────────


def propagate_sign(path_signs: list[str]) -> str:
    """Multiply signs along a path.

    Returns "+" if there are an even number of negatives (including zero),
    "-" if there are an odd number, "ambiguous" if any element is "ambiguous".

    This is the canonical implementation from CLAUDE.md section 5.
    Do not modify this function.
    """
    if "ambiguous" in path_signs:
        return "ambiguous"
    negatives = path_signs.count("-")
    return "-" if negatives % 2 == 1 else "+"


# ── core traversal ────────────────────────────────────────────────────────────


def is_adverse_terminal(node_id: str, net_sign: str, nodes: dict[str, dict]) -> bool:
    """Return True when node_id is a welfare-relevant terminal hit by an adverse shock.

    Raises ValueError if a welfare_relevant node is missing welfare_direction —
    that is a data error in nodes.json, not a silent failure.
    """
    node = nodes.get(node_id, {})
    if not node.get("welfare_relevant", False):
        return False
    direction = node.get("welfare_direction")
    if direction is None:
        raise ValueError(
            f"Node '{node_id}' has welfare_relevant=True but is missing "
            "welfare_direction. Add \"positive\" or \"negative\" to nodes.json."
        )
    if direction == "negative":
        return net_sign == "-"
    if direction == "positive":
        return net_sign == "+"
    raise ValueError(
        f"Node '{node_id}' has unrecognised welfare_direction '{direction}'. "
        "Expected \"positive\" or \"negative\"."
    )


def find_adverse_chains(
    decision_node: str,
    shock_sign: str,
    graph_edges: list[dict],
    nodes: dict[str, dict],
    max_length: int = 4,
    secondary_shocks: list[tuple[str, str]] | None = None,
    secondary_nodes: list[str] | None = None,
) -> list[dict]:
    """Enumerate all simple directed paths that produce an adverse outcome.

    Applies the do-operator to every directly-shocked node: incoming edges to
    decision_node (and each node in secondary_shocks/secondary_nodes, if provided)
    are removed before traversal. The original graph_edges list is not mutated.

    Args:
      decision_node:    Primary node directly set by the decision.
      shock_sign:       Direction of the shock on decision_node ("+" or "-").
      graph_edges:      Full edge list (seed or combined graph).
      nodes:            Node dict keyed by node id.
      max_length:       Maximum number of edges per chain (default 4).
      secondary_shocks: Optional list of (node_id, shock_sign) pairs for
                        additional nodes the decision directly sets alongside
                        decision_node. Each secondary node also has its incoming
                        edges removed (do-operator) and DFS is started from it
                        using its own shock_sign.
      secondary_nodes:  Optional list of node IDs that the decision also directly
                        sets, all using the same shock_sign as decision_node.
                        Convenience alternative to secondary_shocks; both may be
                        supplied and their entries are combined. Duplicate paths
                        produced across multiple start nodes are deduplicated.

    Returns only chains where:
      - net_sign == "-"  (the intervention adversely affects the terminal)
      - terminal node has welfare_relevant == True in nodes
      - 1 <= chain_length <= max_length

    Each returned chain dict contains:
      path         — list of node IDs from start node to terminal (inclusive)
      signs        — list of edge signs along the path
      net_sign     — propagated sign including the start node's shock_sign
      edge_objects — list of full edge dicts in path order
      terminal_node — last node in path
      chain_length  — number of edges (len(path) - 1)
    """
    # Collect all directly-shocked (start_node, shock) pairs
    all_shocks: list[tuple[str, str]] = [(decision_node, shock_sign)]
    if secondary_shocks:
        all_shocks.extend(secondary_shocks)
    if secondary_nodes:
        all_shocks.extend((n, shock_sign) for n in secondary_nodes)

    # Do-operator: remove incoming edges to every directly-shocked node
    shocked_nodes = {node for node, _ in all_shocks}
    working_edges = [e for e in graph_edges if e["to"] not in shocked_nodes]

    # Build outgoing adjacency once (shared across all DFS starts)
    adj: dict[str, list[dict]] = {}
    for edge in working_edges:
        adj.setdefault(edge["from"], []).append(edge)

    seen_paths: set[tuple[str, ...]] = set()
    results: list[dict] = []

    for start_node, start_shock in all_shocks:
        # Iterative DFS over simple paths (visited set prevents revisiting nodes)
        # Stack entry: (current_node, path_nodes, path_edges)
        stack: list[tuple[str, list[str], list[dict]]] = [
            (start_node, [start_node], [])
        ]

        while stack:
            current, path_nodes, path_edges = stack.pop()

            # Evaluate current path endpoint as a candidate adverse chain
            if path_edges:
                terminal = path_nodes[-1]
                edge_signs = [e["sign"] for e in path_edges]
                net_sign = propagate_sign([start_shock] + edge_signs)

                if is_adverse_terminal(terminal, net_sign, nodes):
                    path_key = tuple(path_nodes)
                    if path_key not in seen_paths:
                        seen_paths.add(path_key)
                        results.append({
                            "chain_type": "adverse",
                            "path": list(path_nodes),
                            "signs": edge_signs,
                            "net_sign": net_sign,
                            "edge_objects": list(path_edges),
                            "terminal_node": terminal,
                            "chain_length": len(path_edges),
                            "min_edge_confidence_tier": min(
                                e.get("confidence_tier", 4) for e in path_edges
                            ),
                        })

            # Continue DFS only if we have room for at least one more edge
            if len(path_edges) < max_length:
                for edge in adj.get(current, []):
                    next_node = edge["to"]
                    if next_node not in path_nodes:  # simple path: no revisit
                        stack.append((
                            next_node,
                            path_nodes + [next_node],
                            path_edges + [edge],
                        ))
                    else:
                        # Feedback loop: next_node would revisit an earlier path node.
                        # Record as a feedback_loop chain rather than silently skipping.
                        loop_path_key = tuple(path_nodes) + ("__feedback_loop__", next_node)
                        if loop_path_key not in seen_paths:
                            seen_paths.add(loop_path_key)
                            loop_signs = [e["sign"] for e in path_edges] + [edge["sign"]]
                            loop_edges = list(path_edges) + [edge]
                            results.append({
                                "chain_type": "feedback_loop",
                                "path": list(path_nodes) + [
                                    {"node_id": "__feedback_loop__", "loop_target": next_node, "sign": None}
                                ],
                                "signs": loop_signs,
                                "net_sign": propagate_sign([start_shock] + loop_signs),
                                "edge_objects": loop_edges,
                                "terminal_node": "__feedback_loop__",
                                "loop_target": next_node,
                                "chain_length": len(path_edges) + 1,
                                "feedback_loop_note_ja": "【フィードバックループ検出: 自己強化または自己修正の可能性 — 定量分析が必要】",
                                "min_edge_confidence_tier": min(
                                    e.get("confidence_tier", 4) for e in loop_edges
                                ),
                            })

    return results


# ── chain annotation ──────────────────────────────────────────────────────────


def flag_cycle_risks(chain: dict, graph_edges: list[dict]) -> dict:
    """Check for potential feedback loops from the terminal node back into the chain.

    Per CLAUDE.md section 5: checks whether the terminal node (or any node
    reachable downstream from it in graph_edges) has an edge pointing back toward
    any node in chain["path"]. If so, adds cycle_risk=True and cycle_risk_note.

    Returns a new dict. Does not mutate the input chain.
    """
    result = dict(chain)
    terminal = chain["terminal_node"]
    path_set = set(chain["path"])

    # BFS from terminal to find all downstream-reachable nodes (including terminal itself)
    adj: dict[str, list[str]] = {}
    for edge in graph_edges:
        adj.setdefault(edge["from"], []).append(edge["to"])

    reachable: set[str] = {terminal}
    queue: list[str] = [terminal]
    while queue:
        node = queue.pop()
        for neighbor in adj.get(node, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Any edge from a reachable node into the chain path signals a potential feedback loop
    feedback_edges = [
        e for e in graph_edges
        if e["from"] in reachable and e["to"] in path_set
    ]

    if feedback_edges:
        sources = sorted({e["from"] for e in feedback_edges})
        targets = sorted({e["to"] for e in feedback_edges})
        result["cycle_risk"] = True
        result["cycle_risk_note"] = (
            f"Terminal node '{terminal}' or downstream node(s) {sources} "
            f"have edge(s) pointing toward {targets}, which are in this causal chain. "
            "A self-reinforcing dynamic may exist that is not captured in this analysis."
        )
    else:
        result.setdefault("cycle_risk", False)
        result.setdefault("cycle_risk_note", None)

    return result


def flag_non_monotonicities(chain: dict) -> dict:
    """Add warnings for non-monotone, research-gap, and international-evidence edges.

    Per CLAUDE.md section 5:
      - monotone: false  → non-monotonicity warning
      - research_gap: true → research gap warning with suggested test
      - international_evidence non-empty → transferability warning

    Appends to chain["warnings"] (creating the list if absent).
    Returns a new dict. Does not mutate the input chain.
    """
    result = dict(chain)
    warnings: list[dict] = list(result.get("warnings", []))

    for edge in chain["edge_objects"]:
        edge_id = edge.get("id", "<unknown>")
        edge_name = edge.get("name_en", "")

        if not edge.get("monotone", True):
            note = edge.get("monotone_note") or (
                "Non-monotone: the sign of this edge may not hold uniformly "
                "across all regimes or magnitudes of change."
            )
            warnings.append({
                "type": "non_monotone",
                "edge_id": edge_id,
                "edge_name_en": edge_name,
                "message": f"Edge '{edge_id}' ({edge_name}) is non-monotone. {note}",
            })

        if edge.get("research_gap", False):
            note = edge.get("research_gap_note") or (
                "Research gap: empirical support for this causal relationship is limited. "
                "Treat this edge with additional caution."
            )
            warnings.append({
                "type": "research_gap",
                "edge_id": edge_id,
                "edge_name_en": edge_name,
                "message": (
                    f"Edge '{edge_id}' ({edge_name}) has a research gap. {note}"
                ),
            })

        intl = edge.get("international_evidence", [])
        if intl:
            warnings.append({
                "type": "international_evidence",
                "edge_id": edge_id,
                "edge_name_en": edge_name,
                "message": (
                    f"Edge '{edge_id}' ({edge_name}) is supported primarily by "
                    f"international evidence: {intl}. "
                    "Transferability to the Japanese regional banking context "
                    "has not been independently verified."
                ),
            })

    result["warnings"] = warnings
    return result


# ── ambiguity detection ───────────────────────────────────────────────────────


def detect_ambiguity(
    decision_node: str,
    shock_sign: str,
    target_node: str,
    graph_edges: list[dict],
) -> dict:
    """Find all simple directed paths from decision_node to target_node and check sign agreement.

    net_sign for each path = propagate_sign([shock_sign] + edge_signs_along_path).

    Returns:
      {"ambiguous": True, "positive_paths": [...], "negative_paths": [...]}
        when paths of both positive and negative net_sign exist.
      {"ambiguous": False, "sign": str | None, "paths": [...]}
        when all found paths agree on sign, or when no paths exist (sign=None).

    Each path entry: {"path": [node_ids], "edge_signs": [str], "net_sign": str}.
    """
    # Build adjacency for DFS
    adj: dict[str, list[dict]] = {}
    for edge in graph_edges:
        adj.setdefault(edge["from"], []).append(edge)

    all_paths: list[dict] = []

    # DFS over all simple paths from decision_node; stop branch when target is reached
    stack: list[tuple[str, list[str], list[dict]]] = [
        (decision_node, [decision_node], [])
    ]

    while stack:
        current, path_nodes, path_edges = stack.pop()

        if current == target_node and path_edges:
            # Record this path; do not extend beyond target
            edge_signs = [e["sign"] for e in path_edges]
            net_sign = propagate_sign([shock_sign] + edge_signs)
            all_paths.append({
                "path": list(path_nodes),
                "edge_signs": edge_signs,
                "net_sign": net_sign,
            })
            continue

        for edge in adj.get(current, []):
            next_node = edge["to"]
            if next_node not in path_nodes:  # simple path: no revisit
                stack.append((
                    next_node,
                    path_nodes + [next_node],
                    path_edges + [edge],
                ))

    if not all_paths:
        return {"ambiguous": False, "sign": None, "paths": []}

    positive_paths = [p for p in all_paths if p["net_sign"] == "+"]
    negative_paths = [p for p in all_paths if p["net_sign"] == "-"]

    if positive_paths and negative_paths:
        return {
            "ambiguous": True,
            "positive_paths": positive_paths,
            "negative_paths": negative_paths,
        }

    # All paths have the same sign (could be "+", "-", or "ambiguous")
    signs_present = {p["net_sign"] for p in all_paths}
    consensus = signs_present.pop() if len(signs_present) == 1 else "ambiguous"
    return {"ambiguous": False, "sign": consensus, "paths": all_paths}
