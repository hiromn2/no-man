"""Schema adapter: bridge the No-Man pipeline output to the noman_latex rendering schema.

The pipeline (run_d03_report.py) produces a nested JSON with separate ``session``,
``decision_type``, ``chain_reports``, and ``generated_at`` keys.  noman_latex.py
expects a flattened dict with top-level scalar fields and a ``chains`` list whose
entries carry ``nodes`` (Japanese labels), ``severity``, ``prose``, ``citations``,
and ``signs`` — none of which are directly available in the raw output.

Public API
----------
    adapt_report(raw, nodes=None) -> dict
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

_DEFAULT_NODES_PATH = Path(__file__).parent.parent / "knowledge" / "nodes.json"
_DEFAULT_GOVERNANCE_PATH = Path(__file__).parent.parent / "knowledge" / "governance.json"


# ── public ────────────────────────────────────────────────────────────────────


def adapt_report(
    raw: dict,
    nodes: dict | None = None,
) -> dict:
    """Convert raw pipeline output dict into the noman_latex rendering schema.

    Args:
        raw:   Full output from run_d03() — must contain the keys ``session``,
               ``decision_type``, ``chain_reports``, and ``generated_at``.
        nodes: Optional dict keyed by node ID (as returned by graph_io.load_nodes).
               Auto-loaded from ``knowledge/nodes.json`` when None.

    Returns:
        Dict with keys::

            decision_label   str          Japanese decision-type name
            report_date      str          ISO date (YYYY-MM-DD)
            executive_summary str         経営陣向けサマリー text (may be "")
            graph_version    str          version tag from governance.json
            promotion_mode   str          "human" or "automated"
            chains           list[dict]   adapted chain list (see below)

        Each chain dict has keys:
            chain_id, chain_type, confidence_tier, severity, title,
            path (node IDs), nodes (Japanese labels), signs (edge signs),
            prose, mitigation, citations, is_gap_placeholder (optional)
    """
    if nodes is None:
        nodes = _load_nodes_safe(_DEFAULT_NODES_PATH)

    graph_version, promotion_mode = _load_governance_meta()

    session = raw.get("session", {})
    decision_type = raw.get("decision_type", {})
    chain_reports: list[dict] = raw.get("chain_reports", [])
    generated_at: str = raw.get("generated_at", "")

    decision_label = decision_type.get("name_ja") or decision_type.get("name_en", "")
    report_date = generated_at[:10] if len(generated_at) >= 10 else generated_at
    executive_summary = session.get("executive_summary_ja") or ""

    session_chains: list[dict] = session.get("chains", [])

    adapted_chains: list[dict] = []
    for i, cr in enumerate(chain_reports):
        sc = session_chains[i] if i < len(session_chains) else {}
        adapted_chains.append(_adapt_chain(cr, sc, nodes))

    return {
        "decision_label": decision_label,
        "report_date": report_date,
        "executive_summary": executive_summary,
        "graph_version": graph_version,
        "promotion_mode": promotion_mode,
        "chains": adapted_chains,
    }


# ── private ───────────────────────────────────────────────────────────────────


def _adapt_chain(chain_report: dict, session_chain: dict, nodes: dict) -> dict:
    """Build one adapted chain dict from a (chain_report, session_chain) pair."""
    if chain_report.get("is_gap_placeholder"):
        return {
            "chain_id": "GAP_PLACEHOLDER",
            "chain_type": "gap",
            "confidence_tier": "",
            "severity": "",
            "title": "因果証拠不十分（G3/G4ギャップ）",
            "path": [],
            "nodes": [],
            "signs": [],
            "prose": chain_report.get("causal_chain_text", ""),
            "mitigation": [],
            "citations": [],
            "is_gap_placeholder": True,
        }

    path: list[str] = session_chain.get("path", [])
    signs: list[str] = session_chain.get("signs", [])

    node_labels = [
        nodes.get(nid, {}).get("name_ja") or nid for nid in path
    ]

    # Title: first node → last node in Japanese (abbreviated for header use).
    if len(node_labels) >= 2:
        title = f"{node_labels[0]} → {node_labels[-1]}"
    elif node_labels:
        title = node_labels[0]
    else:
        title = chain_report.get("chain_id", "")

    # Severity: prefer LLM-assessed value; fall back to inferred heuristic.
    snr = chain_report.get("severity_novelty_reversibility") or {}
    severity = snr.get("severity") or ""
    if not severity:
        from noman_latex import infer_severity
        severity = infer_severity(session_chain)

    # Prose: combine premises text + severity/novelty/reversibility narrative.
    prose_parts: list[str] = []
    premises = chain_report.get("premises_text")
    if premises:
        prose_parts.append(str(premises))
    if snr:
        for key_ja, key_en, just_key in (
            ("重大性", "severity", "severity_justification"),
            ("新規性", "novelty", "novelty_justification"),
            ("可逆性", "reversibility", "reversibility_justification"),
        ):
            val = snr.get(key_en, "")
            just = snr.get(just_key, "")
            if val or just:
                prose_parts.append(f"【{key_ja}】{val}：{just}")
    prose = _clean_sign_notation("\n\n".join(prose_parts))

    # Citations: collect unique citation strings from evidence_text.
    citations: list[str] = []
    for pair in chain_report.get("evidence_text", []):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            cit = str(pair[1])
        else:
            cit = str(pair)
        if cit and cit not in citations:
            citations.append(cit)

    return {
        "chain_id": chain_report.get("chain_id", ""),
        "chain_type": session_chain.get("chain_type", "adverse"),
        "confidence_tier": chain_report.get("confidence_tier", ""),
        "severity": severity,
        "title": title,
        "path": path,
        "nodes": node_labels,
        "signs": signs,
        "prose": prose,
        "mitigation": chain_report.get("mitigations", []),
        "citations": citations,
    }


def _clean_sign_notation(text: str) -> str:
    """Remove bracket sign notation ([+], [-]) left by the prose renderer.

    The old renderer embeds ``→[+]→`` and ``[+]`` / ``[-]`` in LLM-generated prose.
    The new template displays sign information through edge colours in the TikZ DAG;
    the prose body should not repeat it in bracket form.

    Transformations:
      ``→[+]→``  →  ``→``
      ``[+]``    →  ``（正）``
      ``[-]`` / ``[−]``  →  ``（負）``
    """
    # Strip sign labels from arrow sequences first
    text = re.sub(r"→\[([+\-−])\]→", "→", text)
    # Replace remaining bracket signs with readable Japanese equivalents
    text = re.sub(r"\[\+\]", "（正）", text)
    text = re.sub(r"\[[-−]\]", "（負）", text)
    return text


def _load_nodes_safe(nodes_path: Union[str, Path]) -> dict:
    """Load nodes.json → dict keyed by node id; return {} on any failure."""
    try:
        with open(nodes_path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("nodes", data)
        if isinstance(raw, list):
            return {n["id"]: n for n in raw}
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _load_governance_meta() -> tuple[str, str]:
    """Return (graph_version, promotion_mode) from governance.json.

    Falls back to ("unknown", "human") if the file is missing or malformed.
    promotion_mode is "automated" only when automated lookup is enabled AND
    requires no human approval; otherwise "human".
    """
    try:
        with open(_DEFAULT_GOVERNANCE_PATH, encoding="utf-8") as f:
            g = json.load(f)
        version = f"v{g.get('version', 'unknown')}"
        acl = g.get("automated_citation_lookup", {})
        automated = acl.get("allowed", False) and not acl.get(
            "requires_human_approval_before_promotion", True
        )
        mode = "automated" if automated else "human"
        return version, mode
    except Exception:
        return "unknown", "human"
