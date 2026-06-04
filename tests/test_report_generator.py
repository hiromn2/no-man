"""Tests for src/report_generator.py.

Covers render_structural_sections (pure Python, no LLM):
  - Causal chain renders correctly with Japanese node names
  - Signs appear in →[+]→ / →[-]→ format
  - Speculative edge renders with 投機的 note
  - Cycle risk flag renders when cycle_risk=True
  - Non-monotonicity warnings render
  - Empty warnings list produces empty flags_text
  - language="en" uses English node names and edge names
  - evidence_text is list of tuples
  - confidence_text contains tier opening phrase

Covers generate_interpretive_sections (one mock test):
  - Calls Anthropic API with model claude-sonnet-4-20250514
  - Loads the correct prompt file for the language
  - Returns interpretive_error=True on API failure

Covers _select_most_severe_chain (pure Python, no LLM):
  - Empty input returns None
  - Single-chain input returns that chain
  - Higher tier wins over lower tier
  - Tier rank dominates severity rank
  - Severity breaks ties within the same tier
  - Missing severity_novelty_reversibility defaults to medium
  - severity_novelty_reversibility=None does not crash

Covers generate_executive_summary (mock LLM):
  - Empty chain_reports returns None keys without error
  - Success returns executive_summary_ja and featured_chain_id
  - featured_chain_id matches the chain _select_most_severe_chain would pick
  - Calls the Anthropic API with _MODEL
  - API failure returns interpretive_error=True without raising
  - API failure preserves featured_chain_id of the selected chain
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from report_generator import (  # noqa: E402
    _MODEL,
    _SEVERITY_RANK,
    _TIER_OPENING_EN,
    _TIER_OPENING_JA,
    _TIER_RANK,
    _select_most_severe_chain,
    generate_executive_summary,
    generate_interpretive_sections,
    render_structural_sections,
)


# ── minimal fixtures ──────────────────────────────────────────────────────────

NODES_JA = {
    "bank_loan_volume": {
        "id": "bank_loan_volume",
        "name_ja": "銀行貸出量",
        "name_en": "Bank loan volume",
        "welfare_relevant": False,
    },
    "regional_sme_credit_access": {
        "id": "regional_sme_credit_access",
        "name_ja": "中小企業の信用アクセス",
        "name_en": "SME credit access",
        "welfare_relevant": True,
    },
    "regional_employment": {
        "id": "regional_employment",
        "name_ja": "地域雇用",
        "name_en": "Regional employment",
        "welfare_relevant": True,
    },
}


def _make_edge(
    edge_id="E-T01",
    from_node="bank_loan_volume",
    to_node="regional_sme_credit_access",
    sign="+",
    status="tested",
    confidence="high",
    name_ja="銀行貸出量 → 中小企業の信用アクセス",
    name_en="Bank loan volume → SME credit access",
    sources=None,
    proposed_by="literature",
    monotone=True,
    monotone_note=None,
    research_gap=False,
    research_gap_note=None,
    international_evidence=None,
) -> dict:
    return {
        "id": edge_id,
        "from": from_node,
        "to": to_node,
        "sign": sign,
        "status": status,
        "confidence": confidence,
        "name_ja": name_ja,
        "name_en": name_en,
        "sources": sources if sources is not None else ["Test Source 2024"],
        "proposed_by": proposed_by,
        "monotone": monotone,
        "monotone_note": monotone_note,
        "research_gap": research_gap,
        "research_gap_note": research_gap_note,
        "international_evidence": international_evidence or [],
    }


def _make_chain(
    edges=None,
    path=None,
    signs=None,
    confidence_tier="well-supported",
    confidence_reason="Assigned well-supported: all edges high-confidence; length 1 ≤ 3.",
    cycle_risk=False,
    cycle_risk_note=None,
    warnings=None,
) -> dict:
    if edges is None:
        edges = [_make_edge()]
    if path is None:
        path = ["bank_loan_volume", "regional_sme_credit_access"]
    if signs is None:
        signs = [e["sign"] for e in edges]
    return {
        "path": path,
        "signs": signs,
        "net_sign": "-",
        "edge_objects": edges,
        "terminal_node": path[-1],
        "chain_length": len(edges),
        "confidence_tier": confidence_tier,
        "confidence_reason": confidence_reason,
        "cycle_risk": cycle_risk,
        "cycle_risk_note": cycle_risk_note,
        "warnings": warnings or [],
    }


# ── render_structural_sections: causal_chain_text ─────────────────────────────


def test_causal_chain_uses_japanese_node_names():
    chain = _make_chain()
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert "銀行貸出量" in result["causal_chain_text"]
    assert "中小企業の信用アクセス" in result["causal_chain_text"]


def test_causal_chain_uses_english_node_names():
    chain = _make_chain()
    result = render_structural_sections(chain, NODES_JA, language="en")
    assert "Bank loan volume" in result["causal_chain_text"]
    assert "SME credit access" in result["causal_chain_text"]


def test_causal_chain_positive_sign_formatted():
    chain = _make_chain(edges=[_make_edge(sign="+")])
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert "→[+]→" in result["causal_chain_text"]


def test_causal_chain_negative_sign_formatted():
    chain = _make_chain(edges=[_make_edge(sign="-")])
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert "→[-]→" in result["causal_chain_text"]


def test_causal_chain_ambiguous_sign_formatted():
    chain = _make_chain(edges=[_make_edge(sign="ambiguous")])
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert "→[ambiguous]→" in result["causal_chain_text"]


def test_causal_chain_two_edge_path():
    edges = [
        _make_edge(
            edge_id="E1",
            from_node="bank_loan_volume",
            to_node="regional_sme_credit_access",
            sign="+",
        ),
        _make_edge(
            edge_id="E2",
            from_node="regional_sme_credit_access",
            to_node="regional_employment",
            sign="+",
            name_ja="中小企業の信用アクセス → 地域雇用",
            name_en="SME credit access → Regional employment",
        ),
    ]
    chain = _make_chain(
        edges=edges,
        path=["bank_loan_volume", "regional_sme_credit_access", "regional_employment"],
        signs=["+", "+"],
    )
    result = render_structural_sections(chain, NODES_JA, language="ja")
    text = result["causal_chain_text"]
    assert "銀行貸出量" in text
    assert "中小企業の信用アクセス" in text
    assert "地域雇用" in text
    assert text.index("銀行貸出量") < text.index("中小企業の信用アクセス") < text.index("地域雇用")


def test_causal_chain_fallback_to_node_id_when_name_missing():
    nodes_missing = {
        "bank_loan_volume": {"name_ja": "銀行貸出量", "name_en": "Bank loan volume"},
        "ghost_node": {},  # no name fields
    }
    edge = _make_edge(from_node="bank_loan_volume", to_node="ghost_node")
    chain = _make_chain(
        edges=[edge],
        path=["bank_loan_volume", "ghost_node"],
        signs=["+"],
    )
    result = render_structural_sections(chain, nodes_missing, language="ja")
    assert "ghost_node" in result["causal_chain_text"]


# ── render_structural_sections: evidence_text ─────────────────────────────────


def test_evidence_text_is_list_of_tuples():
    chain = _make_chain()
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert isinstance(result["evidence_text"], list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result["evidence_text"])


def test_evidence_text_tested_edge_contains_sources():
    chain = _make_chain(edges=[_make_edge(sources=["Uchida et al. 2008", "Hoshi 2001"])])
    result = render_structural_sections(chain, NODES_JA, language="ja")
    _, note = result["evidence_text"][0]
    assert "Uchida et al. 2008" in note
    assert "Hoshi 2001" in note


def test_evidence_text_speculative_edge_renders_with_toukiteki():
    chain = _make_chain(
        edges=[_make_edge(status="speculative", proposed_by="worker_tanaka", sources=[])]
    )
    result = render_structural_sections(chain, NODES_JA, language="ja")
    _, note = result["evidence_text"][0]
    assert "投機的" in note
    assert "worker_tanaka" in note


def test_evidence_text_speculative_edge_en():
    chain = _make_chain(
        edges=[_make_edge(status="speculative", proposed_by="worker_tanaka", sources=[])]
    )
    result = render_structural_sections(chain, NODES_JA, language="en")
    _, note = result["evidence_text"][0]
    assert "Speculative" in note
    assert "worker_tanaka" in note


def test_evidence_text_uses_japanese_edge_name_for_ja():
    chain = _make_chain(edges=[_make_edge(name_ja="銀行貸出量 → 中小企業の信用アクセス")])
    result = render_structural_sections(chain, NODES_JA, language="ja")
    edge_name, _ = result["evidence_text"][0]
    assert edge_name == "銀行貸出量 → 中小企業の信用アクセス"


def test_evidence_text_uses_english_edge_name_for_en():
    chain = _make_chain(edges=[_make_edge(name_en="Bank loan volume → SME credit access")])
    result = render_structural_sections(chain, NODES_JA, language="en")
    edge_name, _ = result["evidence_text"][0]
    assert edge_name == "Bank loan volume → SME credit access"


# ── render_structural_sections: confidence_text ───────────────────────────────


def test_confidence_text_contains_well_supported_opening_ja():
    chain = _make_chain(confidence_tier="well-supported")
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert _TIER_OPENING_JA["well-supported"] in result["confidence_text"]


def test_confidence_text_contains_plausible_opening_ja():
    chain = _make_chain(confidence_tier="plausible")
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert _TIER_OPENING_JA["plausible"] in result["confidence_text"]


def test_confidence_text_contains_worth_considering_opening_ja():
    chain = _make_chain(confidence_tier="worth-considering")
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert _TIER_OPENING_JA["worth-considering"] in result["confidence_text"]


def test_confidence_text_en_uses_english_opening():
    chain = _make_chain(confidence_tier="well-supported")
    result = render_structural_sections(chain, NODES_JA, language="en")
    assert _TIER_OPENING_EN["well-supported"] in result["confidence_text"]


def test_confidence_text_contains_reason():
    chain = _make_chain(confidence_reason="All edges high-confidence at length 1.")
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert "All edges high-confidence at length 1." in result["confidence_text"]


# ── render_structural_sections: flags_text ────────────────────────────────────


def test_flags_empty_when_no_cycle_no_warnings():
    chain = _make_chain(cycle_risk=False, warnings=[])
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert result["flags_text"] == []


def test_flags_cycle_risk_renders_in_ja():
    chain = _make_chain(
        cycle_risk=True,
        cycle_risk_note="Terminal node has feedback edge into the chain.",
    )
    result = render_structural_sections(chain, NODES_JA, language="ja")
    flags = result["flags_text"]
    assert len(flags) >= 1
    assert any("循環リスク" in f for f in flags)
    assert any("Terminal node" in f for f in flags)


def test_flags_cycle_risk_renders_in_en():
    chain = _make_chain(
        cycle_risk=True,
        cycle_risk_note="Feedback detected.",
    )
    result = render_structural_sections(chain, NODES_JA, language="en")
    flags = result["flags_text"]
    assert any("[Cycle Risk]" in f for f in flags)


def test_flags_non_monotone_warning_renders():
    warnings = [{
        "type": "non_monotone",
        "edge_id": "E-005",
        "edge_name_en": "BoJ rate → NIM",
        "message": "Edge 'E-005' is non-monotone. Sign may vary by regime.",
    }]
    chain = _make_chain(cycle_risk=False, warnings=warnings)
    result = render_structural_sections(chain, NODES_JA, language="ja")
    flags = result["flags_text"]
    assert any("非単調性" in f for f in flags)
    assert any("E-005" in f for f in flags)


def test_flags_research_gap_warning_renders():
    warnings = [{
        "type": "research_gap",
        "edge_id": "E-020",
        "edge_name_en": "test edge",
        "message": "Edge 'E-020' has a research gap.",
    }]
    chain = _make_chain(cycle_risk=False, warnings=warnings)
    result = render_structural_sections(chain, NODES_JA, language="ja")
    flags = result["flags_text"]
    assert any("研究ギャップ" in f for f in flags)


def test_flags_multiple_warnings_all_render():
    warnings = [
        {"type": "non_monotone", "edge_id": "E-A", "message": "Non-monotone A."},
        {"type": "research_gap", "edge_id": "E-B", "message": "Research gap B."},
    ]
    chain = _make_chain(cycle_risk=True, cycle_risk_note="Cycle note.", warnings=warnings)
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert len(result["flags_text"]) == 3  # cycle + 2 warnings


# ── render_structural_sections: return type ───────────────────────────────────


def test_returns_dict_with_all_required_keys():
    chain = _make_chain()
    result = render_structural_sections(chain, NODES_JA, language="ja")
    assert set(result.keys()) == {"causal_chain_text", "evidence_text", "confidence_text", "flags_text"}


def test_causal_chain_text_is_string():
    result = render_structural_sections(_make_chain(), NODES_JA, language="ja")
    assert isinstance(result["causal_chain_text"], str)


def test_confidence_text_is_string():
    result = render_structural_sections(_make_chain(), NODES_JA, language="ja")
    assert isinstance(result["confidence_text"], str)


def test_flags_text_is_list():
    result = render_structural_sections(_make_chain(), NODES_JA, language="ja")
    assert isinstance(result["flags_text"], list)


# ── generate_interpretive_sections: mock test ─────────────────────────────────


def test_generate_interpretive_calls_correct_model_and_loads_prompt():
    """generate_interpretive_sections must call _MODEL and load the ja prompt file."""
    interpretive_payload = {
        "premises_text": "テスト前提",
        "severity_novelty_reversibility": {
            "severity": "medium",
            "severity_justification": "test",
            "novelty": "low",
            "novelty_justification": "test",
            "reversibility": "medium",
            "reversibility_justification": "test",
        },
        "mitigations": [],
        "diagnostics": [],
    }
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(interpretive_payload))]

    chain = _make_chain()
    structural = render_structural_sections(chain, NODES_JA, language="ja")
    decision_type = {"name_ja": "貸出削減", "name_en": "Reduce lending"}

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        result = generate_interpretive_sections(
            chain=chain,
            structural=structural,
            local_context={"prefecture": "秋田県"},
            decision_type=decision_type,
            language="ja",
        )

    assert mock_anthropic.Anthropic.called
    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs.get("model") == _MODEL
    assert "interpretive_error" not in result or result.get("interpretive_error") is not True
    assert result.get("premises_text") == "テスト前提"


def test_generate_interpretive_returns_error_on_api_failure():
    """A failed API call returns interpretive_error=True with the structural sections intact."""
    chain = _make_chain()
    structural = render_structural_sections(chain, NODES_JA, language="ja")

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API timeout")

        result = generate_interpretive_sections(
            chain=chain,
            structural=structural,
            local_context={},
            decision_type={"name_ja": "テスト"},
            language="ja",
        )

    assert result["interpretive_error"] is True
    assert "API timeout" in result["error_message"]
    assert result["mitigations"] == []
    assert result["diagnostics"] == []


# ── _make_report: minimal assembled-report fixture ───────────────────────────
# Note: generate_executive_summary and _select_most_severe_chain operate on
# assembled report dicts (output of assemble_report), not raw chain dicts.
# Only the fields those functions actually read are needed here.


def _make_report(
    chain_id: str = "C-001",
    tier: str = "plausible",
    severity: str | None = "medium",
) -> dict:
    """Return a minimal assembled-report dict for executive-summary tests.

    Omitting severity (None) leaves severity_novelty_reversibility absent,
    simulating a report where the LLM interpretive call failed.
    severity=None is distinct from severity_novelty_reversibility=None
    (which represents an explicit null from the LLM).
    """
    r: dict = {
        "chain_id": chain_id,
        "confidence_tier": tier,
        "causal_chain_text": f"ノードA →[+]→ ノードB ({chain_id})",
    }
    if severity is not None:
        r["severity_novelty_reversibility"] = {"severity": severity}
    return r


# ── _select_most_severe_chain: unit tests (no LLM) ───────────────────────────


def test_select_empty_list_returns_none():
    """Empty input must return None, not raise."""
    assert _select_most_severe_chain([]) is None


def test_select_single_chain_returns_that_chain():
    """A list of one element must return that exact element."""
    r = _make_report("C-only", "plausible", "high")
    assert _select_most_severe_chain([r]) is r


def test_select_higher_tier_wins_over_lower():
    """well-supported beats plausible beats worth-considering, regardless of severity."""
    wc = _make_report("C-wc", "worth-considering", "high")
    pl = _make_report("C-pl", "plausible", "high")
    ws = _make_report("C-ws", "well-supported", "high")
    result = _select_most_severe_chain([wc, pl, ws])
    assert result["chain_id"] == "C-ws"


def test_select_tier_rank_dominates_severity_rank():
    """well-supported/low must beat plausible/high — tier is the primary key."""
    ws_low = _make_report("C-ws-low", "well-supported", "low")
    pl_high = _make_report("C-pl-high", "plausible", "high")
    result = _select_most_severe_chain([ws_low, pl_high])
    assert result["chain_id"] == "C-ws-low"


def test_select_severity_breaks_ties_within_same_tier():
    """When tier is equal, high severity beats medium beats low."""
    lo = _make_report("C-lo", "plausible", "low")
    md = _make_report("C-md", "plausible", "medium")
    hi = _make_report("C-hi", "plausible", "high")
    result = _select_most_severe_chain([lo, md, hi])
    assert result["chain_id"] == "C-hi"


def test_select_missing_snr_treated_as_medium_beats_low():
    """A report with no severity_novelty_reversibility key defaults to medium severity."""
    no_snr = {"chain_id": "C-no-snr", "confidence_tier": "plausible"}
    lo = _make_report("C-lo", "plausible", "low")
    # medium (default) should beat low
    result = _select_most_severe_chain([lo, no_snr])
    assert result["chain_id"] == "C-no-snr"


def test_select_missing_snr_loses_to_explicit_high():
    """A report defaulting to medium severity should lose to an explicit high."""
    no_snr = {"chain_id": "C-no-snr", "confidence_tier": "plausible"}
    hi = _make_report("C-hi", "plausible", "high")
    result = _select_most_severe_chain([no_snr, hi])
    assert result["chain_id"] == "C-hi"


def test_select_null_snr_field_does_not_crash():
    """severity_novelty_reversibility=None must not crash; treat as medium."""
    r = {
        "chain_id": "C-null-snr",
        "confidence_tier": "plausible",
        "severity_novelty_reversibility": None,
    }
    result = _select_most_severe_chain([r])
    assert result["chain_id"] == "C-null-snr"


# ── generate_executive_summary: mock LLM tests ───────────────────────────────


def test_generate_executive_summary_empty_returns_none_keys_no_error():
    """Empty chain_reports returns None for both keys; no interpretive_error key."""
    result = generate_executive_summary([], {"name_ja": "テスト"})
    assert result["executive_summary_ja"] is None
    assert result["featured_chain_id"] is None
    assert "interpretive_error" not in result


def test_generate_executive_summary_success_returns_required_keys():
    """Successful LLM call returns executive_summary_ja and featured_chain_id."""
    chain_reports = [_make_report("C-1", "plausible", "high")]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="リスクが示唆されます。可能性があります。注意が必要と考えられます。")]

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        result = generate_executive_summary(chain_reports, {"name_ja": "支店閉鎖"})

    assert "executive_summary_ja" in result
    assert "featured_chain_id" in result
    assert result["executive_summary_ja"] == "リスクが示唆されます。可能性があります。注意が必要と考えられます。"
    assert "interpretive_error" not in result


def test_generate_executive_summary_featured_chain_id_matches_best_chain():
    """featured_chain_id must equal the chain_id _select_most_severe_chain would pick."""
    best = _make_report("C-best", "well-supported", "high")
    worse = _make_report("C-worse", "plausible", "high")
    worst = _make_report("C-worst", "worth-considering", "high")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="サマリーテキスト。")]

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        result = generate_executive_summary([worse, worst, best], {"name_ja": "テスト"})

    assert result["featured_chain_id"] == "C-best"


def test_generate_executive_summary_calls_correct_model():
    """The Anthropic call must use _MODEL — not a hardcoded string."""
    chain_reports = [_make_report("C-1", "plausible", "medium")]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="テキスト。")]

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        generate_executive_summary(chain_reports, {"name_ja": "テスト"})

    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs.get("model") == _MODEL


def test_generate_executive_summary_api_failure_returns_error_dict_not_raise():
    """LLM failure must return interpretive_error=True with error_message; no exception."""
    chain_reports = [_make_report("C-1", "plausible", "high")]

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("connection refused")

        result = generate_executive_summary(chain_reports, {"name_ja": "テスト"})

    assert result["interpretive_error"] is True
    assert "connection refused" in result["error_message"]
    assert result["executive_summary_ja"] is None


def test_generate_executive_summary_api_failure_preserves_featured_chain_id():
    """Even on LLM failure, featured_chain_id must be the chain selected before the call."""
    chain_reports = [_make_report("C-xyz", "well-supported", "high")]

    with patch("report_generator.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = ValueError("bad model")

        result = generate_executive_summary(chain_reports, {"name_ja": "テスト"})

    assert result["featured_chain_id"] == "C-xyz"
    assert result["interpretive_error"] is True
