"""Unit tests for src/report_adapter.py — adapt_report()."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from report_adapter import adapt_report

# ── fixtures ──────────────────────────────────────────────────────────────────

_NODES = {
    "bank_branch_network": {"id": "bank_branch_network", "name_ja": "銀行支店ネットワーク"},
    "regional_sme_credit_access": {
        "id": "regional_sme_credit_access",
        "name_ja": "地域中小企業の信用アクセス",
    },
    "regional_employment": {"id": "regional_employment", "name_ja": "地域雇用"},
}


def _make_raw(
    *,
    chain_type: str = "adverse",
    confidence_tier: str = "plausible",
    severity: str | None = "high",
    chain_length: int = 1,
    path: list[str] | None = None,
    signs: list[str] | None = None,
    is_gap: bool = False,
    premises_text: str | None = "前提テキスト",
    mitigations: list[dict] | None = None,
    evidence_text: list | None = None,
) -> dict:
    """Build a minimal raw pipeline output dict for testing."""
    if path is None:
        path = ["bank_branch_network", "regional_sme_credit_access"]
    if signs is None:
        signs = ["+"] * (len(path) - 1)
    if evidence_text is None:
        evidence_text = [["edge name", "Citation A, Citation B"]]

    session_chain: dict = {
        "chain_type": chain_type,
        "path": path,
        "signs": signs,
        "chain_length": chain_length,
        "confidence_tier": confidence_tier,
    }

    snr = None if severity is None else {
        "severity": severity,
        "severity_justification": "重大だから",
        "novelty": "medium",
        "novelty_justification": "やや新規",
        "reversibility": "low",
        "reversibility_justification": "不可逆",
    }

    chain_report: dict = {
        "report_id": "rpt-001",
        "chain_id": "chain-abc",
        "confidence_tier": confidence_tier,
        "language": "ja",
        "causal_chain_text": "銀行支店ネットワーク →[+]→ 地域中小企業の信用アクセス",
        "evidence_text": evidence_text,
        "confidence_text": "",
        "flags_text": [],
        "premises_text": premises_text,
        "severity_novelty_reversibility": snr,
        "mitigations": mitigations or [
            {
                "action_ja": "緩和策A",
                "targets_edge": "X → Y",
                "adverse_effects_note": "副作用あり",
                "depth_note": "注記：...",
            }
        ],
        "diagnostics": [],
        "is_gap_placeholder": is_gap,
    }
    if is_gap:
        chain_report["causal_chain_text"] = (
            "現時点では因果証拠が不十分です。（G3/G4ギャップ）"
        )
        chain_report["is_gap_placeholder"] = True

    return {
        "session": {
            "session_id": "sess-001",
            "decision_type_id": "D03",
            "chains": [] if is_gap else [session_chain],
            "executive_summary_ja": "経営陣向けサマリー文。",
            "featured_chain_id": "chain-abc",
        },
        "decision_type": {
            "id": "D03",
            "name_ja": "支店閉鎖またはネットワーク縮小",
            "name_en": "Branch closure",
        },
        "chain_reports": [chain_report],
        "generated_at": "2026-05-30T10:00:00+00:00",
    }


# ── test 1: top-level schema keys ─────────────────────────────────────────────


def test_adapt_report_top_level_keys():
    raw = _make_raw()
    result = adapt_report(raw, nodes=_NODES)

    required = {"decision_label", "report_date", "executive_summary",
                 "graph_version", "promotion_mode", "chains"}
    assert required <= set(result.keys()), (
        f"Missing keys: {required - set(result.keys())}"
    )
    assert result["decision_label"] == "支店閉鎖またはネットワーク縮小"
    assert result["report_date"] == "2026-05-30"
    assert result["executive_summary"] == "経営陣向けサマリー文。"


# ── test 2: chain count matches ───────────────────────────────────────────────


def test_chain_count_matches():
    raw = _make_raw()
    result = adapt_report(raw, nodes=_NODES)
    assert len(result["chains"]) == 1

    # Multiple chains
    raw2 = _make_raw()
    sc2 = {
        "chain_type": "adverse",
        "path": ["bank_branch_network", "regional_employment"],
        "signs": ["+"],
        "chain_length": 1,
        "confidence_tier": "well-supported",
    }
    cr2 = dict(raw2["chain_reports"][0])
    cr2["chain_id"] = "chain-def"
    cr2["confidence_tier"] = "well-supported"
    raw2["session"]["chains"].append(sc2)
    raw2["chain_reports"].append(cr2)
    result2 = adapt_report(raw2, nodes=_NODES)
    assert len(result2["chains"]) == 2


# ── test 3: severity comes from SNR if present, else inferred ─────────────────


def test_severity_from_snr_when_present():
    raw = _make_raw(severity="high")
    result = adapt_report(raw, nodes=_NODES)
    assert result["chains"][0]["severity"] == "high"


def test_severity_inferred_when_snr_absent():
    raw = _make_raw(severity=None)
    result = adapt_report(raw, nodes=_NODES)
    sev = result["chains"][0]["severity"]
    assert sev in ("high", "medium", "low"), f"Unexpected severity: {sev!r}"


# ── test 4: citations are a flat list of strings ──────────────────────────────


def test_citations_are_flat_strings():
    raw = _make_raw(
        evidence_text=[
            ["エッジ名A", "Uchida et al. 2008"],
            ["エッジ名B", "Ogura 2010"],
            ["エッジ名C", "Uchida et al. 2008"],   # duplicate — should be deduplicated
        ]
    )
    result = adapt_report(raw, nodes=_NODES)
    cits = result["chains"][0]["citations"]
    assert isinstance(cits, list)
    assert all(isinstance(c, str) for c in cits), "All citations must be strings"
    assert "Uchida et al. 2008" in cits
    assert "Ogura 2010" in cits
    assert cits.count("Uchida et al. 2008") == 1, "Duplicate citations must be deduplicated"


# ── test 5: nodes field carries Japanese names along the path ─────────────────


def test_nodes_field_contains_japanese_names():
    raw = _make_raw(path=["bank_branch_network", "regional_sme_credit_access"])
    result = adapt_report(raw, nodes=_NODES)
    chain = result["chains"][0]

    assert chain["nodes"] == ["銀行支店ネットワーク", "地域中小企業の信用アクセス"]
    # path still holds the original node IDs
    assert chain["path"] == ["bank_branch_network", "regional_sme_credit_access"]
    # signs carried through from session chain
    assert chain["signs"] == ["+"]


# ── test 6: gap placeholder chain ─────────────────────────────────────────────


def test_gap_placeholder_chain():
    raw = _make_raw(is_gap=True)
    result = adapt_report(raw, nodes=_NODES)
    chain = result["chains"][0]
    assert chain.get("is_gap_placeholder") is True
    assert chain["chain_type"] == "gap"
    assert chain["path"] == []
    assert chain["nodes"] == []
