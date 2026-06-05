"""Load and normalize report JSON for the Streamlit prototype.

The loader accepts either:
- the adapted schema used by the dummy fixture / LaTeX adapter, or
- the raw `output/d03_report.json` shape from `run_d03_report.py`.

It never runs report generation.
"""

from __future__ import annotations

import json
from typing import BinaryIO


def load_uploaded_report(uploaded_file: BinaryIO) -> dict:
    raw = json.load(uploaded_file)
    return normalize_report(raw)


def normalize_report(raw: dict) -> dict:
    """Return a UI-ready report dict from supported report shapes."""
    if "chains" in raw and "chain_reports" not in raw:
        return raw
    if {"session", "decision_type", "chain_reports"}.issubset(raw):
        return _normalize_raw_pipeline_report(raw)
    raise ValueError("Unsupported report JSON shape.")


def _normalize_raw_pipeline_report(raw: dict) -> dict:
    session = raw.get("session", {})
    decision_type = raw.get("decision_type", {})
    chain_reports = raw.get("chain_reports", [])
    session_chains = session.get("chains", [])

    chains = []
    for index, chain_report in enumerate(chain_reports):
        session_chain = session_chains[index] if index < len(session_chains) else {}
        chains.append(_normalize_chain(chain_report, session_chain))

    return {
        "decision_label": decision_type.get("name_ja")
        or decision_type.get("name_en")
        or session.get("decision_type_id", "Untitled decision"),
        "report_date": str(raw.get("generated_at", ""))[:10],
        "graph_version": "existing-report",
        "promotion_mode": "loaded",
        "executive_summary": session.get("executive_summary_ja")
        or session.get("executive_summary_en")
        or "",
        "chains": chains,
    }


def _normalize_chain(chain_report: dict, session_chain: dict) -> dict:
    snr = chain_report.get("severity_novelty_reversibility") or {}
    evidence = chain_report.get("evidence_text", [])
    citations = []
    for item in evidence:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            citations.append(f"{item[0]}: {item[1]}")
        elif isinstance(item, str):
            citations.append(item)

    mitigations = []
    for item in chain_report.get("mitigations", []):
        if isinstance(item, dict):
            action = item.get("action_ja") or item.get("action") or ""
            target = item.get("targets_edge")
            mitigations.append(f"{action} ({target})" if target else action)
        else:
            mitigations.append(str(item))

    return {
        "chain_id": chain_report.get("chain_id", "unknown"),
        "title": _title_from_chain(chain_report, session_chain),
        "chain_type": session_chain.get("chain_type", "adverse"),
        "confidence_tier": chain_report.get("confidence_tier", ""),
        "severity": snr.get("severity", "unknown"),
        "path": session_chain.get("path", []),
        "nodes": _nodes_from_chain_text(chain_report, session_chain),
        "signs": session_chain.get("signs", []),
        "prose": chain_report.get("premises_text")
        or chain_report.get("causal_chain_text", ""),
        "citations": citations,
        "mitigation": mitigations,
        "diagnostics": chain_report.get("diagnostics", []),
    }


def _title_from_chain(chain_report: dict, session_chain: dict) -> str:
    nodes = _nodes_from_chain_text(chain_report, session_chain)
    if len(nodes) >= 2:
        return f"{nodes[0]} -> {nodes[-1]}"
    return chain_report.get("chain_id", "Untitled chain")


def _nodes_from_chain_text(chain_report: dict, session_chain: dict) -> list[str]:
    chain_text = chain_report.get("causal_chain_text", "")
    if "→" in chain_text:
        parts = []
        for piece in chain_text.split("→"):
            cleaned = piece.replace("[+]", "").replace("[-]", "").strip()
            cleaned = cleaned.strip("-[] ")
            if cleaned:
                parts.append(cleaned)
        if parts:
            return parts
    return session_chain.get("path", [])
