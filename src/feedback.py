"""Governance loop for the No-Man causal knowledge base.

Edge lifecycle (all transitions require explicit function calls):

  propose_edge()           — session worker_edge → speculative_graph (status: proposed)
  run_adversarial_review() — LLM counterclaim + Semantic Scholar (status unchanged)
  submit_for_human_review()— proposed → speculative; adds to review_queue.json
  human_approve()          — speculative → seed_graph (status: tested)
  flag_edge()              — attach dispute flag; set review_required after ≥3 flags / 90 days

Governance rules:
  proposed → speculative  requires adversarial_result present (auto, feedback.py)
  speculative → tested    requires citation + empirical validation + human approval
  All transitions use graph_io.save_graph() and graph_io.validate() — no raw json.dump().
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Union

from graph_io import (
    GraphIntegrityError,
    load_nodes,
    save_graph,
    validate,
    check_cross_graph_uniqueness,
)

_FLAG_THRESHOLD = 3
_FLAG_REVIEW_DAYS = 90


# ── internal helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_edges(graph_path: Path) -> list[dict]:
    """Load edge list from a graph JSON file. Returns [] if file absent."""
    if not graph_path.exists():
        return []
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)
    edges = data.get("edges", data)
    return edges if isinstance(edges, list) else []


def _find_edge(edges: list[dict], edge_id: str) -> dict | None:
    return next((e for e in edges if e.get("id") == edge_id), None)


def _get_anthropic_client():
    """Return an Anthropic client. Raises EnvironmentError if key absent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required: pip install anthropic"
        ) from exc


def _graph_type_from_path(graph_path: Path) -> str:
    """Infer graph_type from filename: 'seed' if 'seed' in stem, else 'speculative'."""
    return "seed" if "seed" in graph_path.stem else "speculative"


# ── public API ────────────────────────────────────────────────────────────────


def propose_edge(
    edge: dict,
    session: dict,
    graph_path: Union[str, Path],
    nodes_path: Union[str, Path, None] = None,
) -> None:
    """Write a worker-proposed edge to speculative_graph.json with status='proposed'.

    The edge must already exist in session['worker_edges'] (identified by ID).
    Status is forced to 'proposed' regardless of the session's stored status.
    confidence defaults to 'low' and confidence_tier to 1 if not set.

    Args:
        edge: Edge dict from session['worker_edges'].
        session: Session dict containing the worker_edges list.
        graph_path: Path to speculative_graph.json.
        nodes_path: Path to nodes.json. Inferred from graph_path parent if omitted.

    Raises:
        ValueError: If edge not found in session worker_edges.
        GraphIntegrityError: If edge ID already in graph, or validation fails.
    """
    graph_file = Path(graph_path)
    if nodes_path is None:
        nodes_path = graph_file.parent / "nodes.json"

    edge_id = edge.get("id")
    worker_edges = session.get("worker_edges", [])
    if not any(we.get("id") == edge_id for we in worker_edges):
        raise ValueError(
            f"Edge '{edge_id}' not found in session['worker_edges']. "
            "Add it via session.add_worker_edge() before calling propose_edge()."
        )

    existing = _load_edges(graph_file)
    if any(e.get("id") == edge_id for e in existing):
        raise GraphIntegrityError(
            f"Edge '{edge_id}' already exists in {graph_path}."
        )

    # Cross-file uniqueness: reject if ID already in seed graph
    seed_path = graph_file.parent / "seed_graph.json"
    if seed_path.exists():
        seed_edges = _load_edges(seed_path)
        check_cross_graph_uniqueness([edge], seed_edges)

    new_edge = dict(edge)
    new_edge["status"] = "proposed"
    new_edge.setdefault("confidence", "low")
    new_edge.setdefault("confidence_tier", 1)

    updated = existing + [new_edge]
    nodes = load_nodes(nodes_path)
    validate(updated, nodes, graph_type="speculative")
    save_graph(updated, graph_file)


def run_adversarial_review(
    edge_id: str,
    graph_path: Union[str, Path],
    nodes_path: Union[str, Path],
) -> dict:
    """Run adversarial review for a proposed edge.

    Stage 1: LLM produces a signed, falsifiable counterclaim in edge format
             {from, to, sign, basis}.
    Stage 2: Semantic Scholar search for papers supporting the counterclaim.

    Writes adversarial_result onto the edge dict. Does NOT change status.
    If no supporting paper is found for the counterclaim, records
    search_exhausted: True.

    Returns the adversarial_result dict written to the edge.

    Raises:
        ValueError: If edge not found or status is not 'proposed'.
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    graph_file = Path(graph_path)
    existing = _load_edges(graph_file)

    edge = _find_edge(existing, edge_id)
    if edge is None:
        raise ValueError(f"Edge '{edge_id}' not found in {graph_path}")
    if edge.get("status") != "proposed":
        raise ValueError(
            f"Edge '{edge_id}' has status '{edge.get('status')}', expected 'proposed'."
        )

    nodes = load_nodes(nodes_path)
    from_node = edge.get("from", "")
    to_node = edge.get("to", "")
    sign = edge.get("sign", "+")

    from_label = nodes.get(from_node, {}).get("name_en", from_node.replace("_", " "))
    to_label = nodes.get(to_node, {}).get("name_en", to_node.replace("_", " "))
    direction = "negatively affects" if sign == "-" else "positively affects"

    # ── Stage 1: LLM counterclaim ─────────────────────────────────────────────
    client = _get_anthropic_client()
    prompt = (
        "You are a critical reviewer of causal claims in Japanese regional banking economics.\n\n"
        "Proposed causal edge:\n"
        f"  From:      {from_label} ({from_node})\n"
        f"  To:        {to_label} ({to_node})\n"
        f"  Direction: {direction}\n"
        f"  Basis:     {edge.get('basis') or '(not stated)'}\n\n"
        "Produce ONE specific, falsifiable counterclaim in this exact JSON format:\n"
        '{\n'
        '  "from": "<cause node or concept>",\n'
        '  "to": "<effect node or concept>",\n'
        '  "sign": "<+ or ->",\n'
        '  "basis": "<one sentence naming a specific competing causal mechanism>"\n'
        '}\n\n'
        "Requirements:\n"
        "- Propose a specific alternative mechanism that CONTRADICTS the proposed edge.\n"
        "- Do NOT express general doubt. Name a concrete competing mechanism.\n"
        "- 'basis' must be a falsifiable claim (e.g. 'X crowds out Y via the Z channel').\n"
        "- Output only the JSON object, no other text."
    )

    counterclaim: dict = {}
    llm_error: str | None = None
    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        counterclaim = json.loads(raw)
    except Exception as exc:
        llm_error = str(exc)

    # ── Stage 2: Semantic Scholar search for counterclaim support ─────────────
    from literature import search_edge_support

    supporting_papers: list[dict] = []
    search_exhausted = True

    if counterclaim:
        cc_from = counterclaim.get("from", "")
        cc_to = counterclaim.get("to", "")
        cc_sign = counterclaim.get("sign", "+")
        try:
            papers = search_edge_support(cc_from, cc_to, cc_sign)
            supporting_papers = [
                {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "doi": p.get("doi"),
                    "source_type": p.get("source_type"),
                    "is_accepted_source_type": p.get("is_accepted_source_type"),
                }
                for p in papers
                if p.get("is_accepted_source_type")
            ]
            search_exhausted = len(papers) == 0
        except Exception as exc:
            llm_error = (llm_error + "; " if llm_error else "") + f"search: {exc}"

    adversarial_result = {
        "counterclaim": counterclaim,
        "supporting_papers": supporting_papers,
        "search_exhausted": search_exhausted,
        "reviewed_at": _now_iso(),
        "method": "llm_adversarial_review",
    }
    if llm_error:
        adversarial_result["llm_error"] = llm_error

    edge["adversarial_result"] = adversarial_result

    validate(existing, nodes, graph_type="speculative")
    save_graph(existing, graph_file)

    return adversarial_result


def submit_for_human_review(
    edge_id: str,
    graph_path: Union[str, Path],
    nodes_path: Union[str, Path, None] = None,
) -> None:
    """Promote edge from 'proposed' to 'speculative' and add to review_queue.json.

    Governance rule (proposed_to_speculative): adversarial_result must be present.

    Args:
        edge_id: ID of the edge to submit.
        graph_path: Path to speculative_graph.json.
        nodes_path: Path to nodes.json. Inferred from graph_path parent if omitted.

    Raises:
        ValueError: If edge not found, wrong status, or adversarial_result absent.
        GraphIntegrityError: If post-transition validation fails.
    """
    graph_file = Path(graph_path)
    if nodes_path is None:
        nodes_path = graph_file.parent / "nodes.json"
    queue_path = graph_file.parent / "review_queue.json"

    existing = _load_edges(graph_file)
    edge = _find_edge(existing, edge_id)
    if edge is None:
        raise ValueError(f"Edge '{edge_id}' not found in {graph_path}")
    if edge.get("status") != "proposed":
        raise ValueError(
            f"Edge '{edge_id}' has status '{edge.get('status')}', expected 'proposed'."
        )
    if edge.get("adversarial_result") is None:
        raise ValueError(
            f"Edge '{edge_id}' has no adversarial_result. "
            "Run run_adversarial_review() before submitting for human review "
            "(governance.json §proposed_to_speculative)."
        )

    edge["status"] = "speculative"

    nodes = load_nodes(nodes_path)
    validate(existing, nodes, graph_type="speculative")
    save_graph(existing, graph_file)

    # Append to review queue (idempotent)
    if queue_path.exists():
        with open(queue_path, encoding="utf-8") as f:
            queue_data = json.load(f)
    else:
        queue_data = {"queue": [], "created_at": _now_iso()}

    if edge_id not in queue_data["queue"]:
        queue_data["queue"].append(edge_id)
        queue_data["updated_at"] = _now_iso()

    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(queue_data, f, ensure_ascii=False, indent=2)


def human_approve(
    edge_id: str,
    reviewer_name: str,
    graph_path_speculative: Union[str, Path],
    graph_path_seed: Union[str, Path],
    nodes_path: Union[str, Path],
) -> None:
    """Approve a speculative edge and move it to seed_graph.json.

    Fail-safe write order:
      1. Validate the promoted edge against the seed graph; write seed first.
      2. Only after seed write succeeds, remove edge from speculative graph.
      3. Remove edge_id from review_queue.json.

    Args:
        edge_id: ID of the edge to approve.
        reviewer_name: Name or ID of the human reviewer.
        graph_path_speculative: Path to speculative_graph.json.
        graph_path_seed: Path to seed_graph.json.
        nodes_path: Path to nodes.json.

    Raises:
        ValueError: If edge not found or status is not 'speculative'.
        GraphIntegrityError: If edge_id already in seed graph, sources absent,
                             or post-write validation fails.
    """
    spec_file = Path(graph_path_speculative)
    seed_file = Path(graph_path_seed)
    queue_path = spec_file.parent / "review_queue.json"

    nodes = load_nodes(nodes_path)
    spec_edges = _load_edges(spec_file)
    seed_edges = _load_edges(seed_file)

    # Guard: detect pre-existing ID conflict before any writes
    check_cross_graph_uniqueness(spec_edges, seed_edges)

    edge = _find_edge(spec_edges, edge_id)
    if edge is None:
        raise ValueError(f"Edge '{edge_id}' not found in {graph_path_speculative}")
    if edge.get("status") != "speculative":
        raise ValueError(
            f"Edge '{edge_id}' has status '{edge.get('status')}', expected 'speculative'."
        )
    if not edge.get("sources"):
        raise GraphIntegrityError(
            f"Edge '{edge_id}' has no sources. A tested edge must cite at least one "
            "academic source (governance.json §requirement_1_academic_citation)."
        )

    # Build promoted edge
    promoted = dict(edge)
    promoted["status"] = "tested"
    promoted["verified_by"] = reviewer_name
    promoted["verified_at"] = _now_iso()

    # Step 1: write to seed graph and validate (DAG check included)
    new_seed = seed_edges + [promoted]
    validate(new_seed, nodes, graph_type="seed")
    save_graph(new_seed, seed_file)

    # Step 2: remove from speculative graph and validate
    new_spec = [e for e in spec_edges if e.get("id") != edge_id]
    validate(new_spec, nodes, graph_type="speculative")
    save_graph(new_spec, spec_file)

    # Step 3: remove from review queue
    if queue_path.exists():
        with open(queue_path, encoding="utf-8") as f:
            queue_data = json.load(f)
        if edge_id in queue_data.get("queue", []):
            queue_data["queue"].remove(edge_id)
            queue_data["updated_at"] = _now_iso()
            with open(queue_path, "w", encoding="utf-8") as f:
                json.dump(queue_data, f, ensure_ascii=False, indent=2)


def flag_edge(
    edge_id: str,
    flagged_by: str,
    reason: str,
    paper: dict | None,
    graph_path: Union[str, Path],
    nodes_path: Union[str, Path, None] = None,
) -> None:
    """Attach a dispute flag to an edge.

    Appends a flag entry to edge['dispute_flags']. If total flags >= 3 and the
    oldest flag is more than 90 days old with no human resolution (verified_by
    is null), sets review_required=True and review_reason='accumulated_flags'.

    Works on any graph file (seed or speculative). Graph type is inferred from
    the filename ('seed' in stem → seed validation; else speculative).

    Args:
        edge_id: ID of the edge to flag.
        flagged_by: Name or ID of the person raising the dispute.
        reason: Description of the dispute.
        paper: Optional paper dict supporting the dispute; may be null.
        graph_path: Path to the graph file containing the edge.
        nodes_path: Path to nodes.json. Inferred from graph_path parent if omitted.

    Raises:
        ValueError: If edge not found.
        GraphIntegrityError: If post-flag validation fails.
    """
    graph_file = Path(graph_path)
    if nodes_path is None:
        nodes_path = graph_file.parent / "nodes.json"

    graph_type = _graph_type_from_path(graph_file)
    existing = _load_edges(graph_file)

    edge = _find_edge(existing, edge_id)
    if edge is None:
        raise ValueError(f"Edge '{edge_id}' not found in {graph_path}")

    flag_entry = {
        "flagged_by": flagged_by,
        "reason": reason,
        "paper": paper,
        "flagged_at": _now_iso(),
    }

    flags: list[dict] = list(edge.get("dispute_flags") or [])
    flags.append(flag_entry)
    edge["dispute_flags"] = flags

    # 90-day accumulated-flags rule
    if len(flags) >= _FLAG_THRESHOLD and not edge.get("verified_by"):
        oldest_dt: datetime | None = None
        for f in flags:
            ts = f.get("flagged_at", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    if oldest_dt is None or dt < oldest_dt:
                        oldest_dt = dt
                except ValueError:
                    pass

        if oldest_dt is not None:
            age = datetime.now(timezone.utc) - oldest_dt
            if age >= timedelta(days=_FLAG_REVIEW_DAYS):
                edge["review_required"] = True
                edge["review_reason"] = (
                    f"accumulated_flags: {len(flags)} dispute flags over {age.days} days "
                    "with no human resolution."
                )

    nodes = load_nodes(nodes_path)
    validate(existing, nodes, graph_type=graph_type)
    save_graph(existing, graph_file)
