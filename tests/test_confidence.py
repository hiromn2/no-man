"""Tests for src/confidence.py — one test per table cell plus edge cases.

Table under test (from CLAUDE.md section 5):

    Edge profile           | Length ≤ 3     | Length = 4     | Length > 4
    -----------------------+----------------+----------------+-------------------
    All high, all tested   | well-supported | plausible      | worth-considering
    Mix high/medium tested | plausible      | worth-consid.  | worth-considering
    Any low OR speculative | worth-consid.  | worth-consid.  | worth-considering
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from confidence import (  # noqa: E402
    TIER_PLAUSIBLE,
    TIER_WELL_SUPPORTED,
    TIER_WORTH_CONSIDERING,
    assign_confidence_tier,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _high_edge(edge_id: str = "E-H") -> dict:
    return {"id": edge_id, "confidence": "high", "status": "tested"}


def _medium_edge(edge_id: str = "E-M") -> dict:
    return {"id": edge_id, "confidence": "medium", "status": "tested"}


def _low_edge(edge_id: str = "E-L") -> dict:
    return {"id": edge_id, "confidence": "low", "status": "tested"}


def _speculative_edge(edge_id: str = "E-S") -> dict:
    return {"id": edge_id, "confidence": "high", "status": "speculative"}


def _make_chain(edges: list[dict], length: int | None = None) -> dict:
    """Minimal chain dict for tier assignment."""
    return {
        "edge_objects": edges,
        "chain_length": length if length is not None else len(edges),
        "warnings": [],
    }


# ── Row 1: All high, all tested ───────────────────────────────────────────────


def test_row1_col1_all_high_length1():
    """All high tested, length 1 → well-supported."""
    chain = _make_chain([_high_edge()])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WELL_SUPPORTED


def test_row1_col1_all_high_length2():
    """All high tested, length 2 → well-supported."""
    chain = _make_chain([_high_edge("E1"), _high_edge("E2")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WELL_SUPPORTED


def test_row1_col1_all_high_length3():
    """All high tested, length 3 → well-supported (boundary)."""
    chain = _make_chain([_high_edge("E1"), _high_edge("E2"), _high_edge("E3")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WELL_SUPPORTED


def test_row1_col2_all_high_length4():
    """All high tested, length 4 → plausible."""
    edges = [_high_edge(f"E{i}") for i in range(4)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_PLAUSIBLE


def test_row1_col3_all_high_length5():
    """All high tested, length 5 → worth-considering."""
    edges = [_high_edge(f"E{i}") for i in range(5)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row1_col3_all_high_length6():
    """All high tested, length 6 → worth-considering."""
    edges = [_high_edge(f"E{i}") for i in range(6)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


# ── Row 2: Mix high/medium, all tested ───────────────────────────────────────


def test_row2_col1_mix_length1():
    """Medium edge only, length 1 → plausible."""
    chain = _make_chain([_medium_edge()])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_PLAUSIBLE


def test_row2_col1_mix_length2():
    """One high + one medium, length 2 → plausible."""
    chain = _make_chain([_high_edge("E1"), _medium_edge("E2")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_PLAUSIBLE


def test_row2_col1_mix_length3():
    """Mix high/medium, length 3 → plausible (boundary)."""
    chain = _make_chain([_high_edge("E1"), _medium_edge("E2"), _high_edge("E3")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_PLAUSIBLE


def test_row2_col2_mix_length4():
    """Mix high/medium, length 4 → worth-considering."""
    edges = [_high_edge("E1"), _medium_edge("E2"), _high_edge("E3"), _medium_edge("E4")]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row2_col3_mix_length5():
    """Mix high/medium, length 5 → worth-considering."""
    edges = [_high_edge(f"E{i}") for i in range(3)] + [_medium_edge(f"M{i}") for i in range(2)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


# ── Row 3: Any low OR speculative ─────────────────────────────────────────────


def test_row3_single_low_length1():
    """Single low-confidence edge, length 1 → worth-considering."""
    chain = _make_chain([_low_edge()])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_single_low_length3():
    """Low edge in length-3 chain → worth-considering (not plausible)."""
    chain = _make_chain([_high_edge("E1"), _high_edge("E2"), _low_edge("E3")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_single_low_length4():
    """Low edge, length 4 → worth-considering."""
    edges = [_high_edge("E1"), _high_edge("E2"), _high_edge("E3"), _low_edge("E4")]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_speculative_length1():
    """Single speculative edge → worth-considering."""
    chain = _make_chain([_speculative_edge()])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_speculative_length3():
    """Speculative in length-3 chain → worth-considering (not plausible or well-supported)."""
    chain = _make_chain([_high_edge("E1"), _high_edge("E2"), _speculative_edge("E3")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_speculative_length4():
    """Speculative, length 4 → worth-considering."""
    edges = [_high_edge("E1"), _high_edge("E2"), _high_edge("E3"), _speculative_edge("E4")]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_all_speculative():
    """All-speculative chain → worth-considering."""
    edges = [_speculative_edge(f"S{i}") for i in range(3)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_mixed_speculative_and_high():
    """Mix of speculative and high edges → worth-considering (speculative trumps)."""
    chain = _make_chain([_high_edge("E1"), _speculative_edge("E2")])
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING


def test_row3_low_downgrade_from_plausible():
    """Single low-confidence edge in an otherwise plausible chain → worth-considering."""
    # Without the low edge this would be plausible (mix high/medium, length 2)
    chain_no_low = _make_chain([_high_edge("E1"), _medium_edge("E2")])
    assert assign_confidence_tier(chain_no_low)["confidence_tier"] == TIER_PLAUSIBLE
    # Adding a low edge must downgrade it
    chain_with_low = _make_chain([_high_edge("E1"), _medium_edge("E2"), _low_edge("E3")])
    assert assign_confidence_tier(chain_with_low)["confidence_tier"] == TIER_WORTH_CONSIDERING


# ── Length warning ────────────────────────────────────────────────────────────


def test_length_warning_absent_for_length4():
    """Length 4 must NOT produce a length warning."""
    edges = [_high_edge(f"E{i}") for i in range(4)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    length_warnings = [w for w in result.get("warnings", []) if w.get("type") == "length_warning"]
    assert len(length_warnings) == 0


def test_length_warning_present_for_length5():
    """Length 5 must produce exactly one length_warning."""
    edges = [_high_edge(f"E{i}") for i in range(5)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    length_warnings = [w for w in result.get("warnings", []) if w.get("type") == "length_warning"]
    assert len(length_warnings) == 1


def test_length_warning_present_for_length6():
    """Length 6 must produce exactly one length_warning."""
    edges = [_high_edge(f"E{i}") for i in range(6)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    length_warnings = [w for w in result.get("warnings", []) if w.get("type") == "length_warning"]
    assert len(length_warnings) == 1


def test_length_warning_message_mentions_length():
    """The length_warning message must reference chain length."""
    edges = [_high_edge(f"E{i}") for i in range(5)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    warning = next(w for w in result["warnings"] if w["type"] == "length_warning")
    assert "5" in warning["message"]


def test_existing_warnings_preserved_when_length_warning_added():
    """Pre-existing warnings in chain must be preserved when length warning is appended."""
    edges = [_high_edge(f"E{i}") for i in range(5)]
    prior_warning = {"type": "non_monotone", "edge_id": "E0", "message": "test"}
    chain = _make_chain(edges)
    chain["warnings"] = [prior_warning]
    result = assign_confidence_tier(chain)
    assert any(w["type"] == "non_monotone" for w in result["warnings"])
    assert any(w["type"] == "length_warning" for w in result["warnings"])


def test_low_edge_with_length5_has_length_warning():
    """Low-confidence edge in length-5 chain → worth-considering AND length warning."""
    edges = [_low_edge("E1")] + [_high_edge(f"E{i}") for i in range(4)]
    chain = _make_chain(edges)
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WORTH_CONSIDERING
    assert any(w["type"] == "length_warning" for w in result["warnings"])


# ── No mutation ───────────────────────────────────────────────────────────────


def test_no_mutation_of_input():
    """assign_confidence_tier must not mutate the input chain dict."""
    edges = [_high_edge("E1"), _high_edge("E2")]
    chain = _make_chain(edges)
    original_keys = set(chain.keys())
    original_warnings = list(chain["warnings"])
    assign_confidence_tier(chain)
    assert set(chain.keys()) == original_keys
    assert chain["warnings"] == original_warnings
    assert "confidence_tier" not in chain
    assert "confidence_reason" not in chain


def test_no_mutation_length5_warnings():
    """assign_confidence_tier must not mutate warnings list in input for long chains."""
    edges = [_high_edge(f"E{i}") for i in range(5)]
    chain = _make_chain(edges)
    chain["warnings"] = []
    assign_confidence_tier(chain)
    assert chain["warnings"] == []


# ── confidence_reason content ─────────────────────────────────────────────────


def test_reason_mentions_tier():
    """confidence_reason must name the tier assigned."""
    chain = _make_chain([_high_edge()])
    result = assign_confidence_tier(chain)
    assert TIER_WELL_SUPPORTED in result["confidence_reason"]


def test_reason_mentions_speculative_edge_id():
    """confidence_reason for speculative chain must mention the speculative edge id."""
    chain = _make_chain([_speculative_edge("E-SPEC-001")])
    result = assign_confidence_tier(chain)
    assert "E-SPEC-001" in result["confidence_reason"]


def test_reason_mentions_low_edge_id():
    """confidence_reason for low chain must mention the low-confidence edge id."""
    chain = _make_chain([_low_edge("E-LOW-007")])
    result = assign_confidence_tier(chain)
    assert "E-LOW-007" in result["confidence_reason"]


def test_reason_mentions_medium_edge_id():
    """confidence_reason for mixed chain must mention the medium-confidence edge id."""
    chain = _make_chain([_high_edge("E1"), _medium_edge("E-MED-003")])
    result = assign_confidence_tier(chain)
    assert "E-MED-003" in result["confidence_reason"]


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_edge_list_treated_as_all_high_length0():
    """Empty edge_objects with chain_length=0 — falls into all_high, length ≤ 3 → well-supported."""
    chain = {"edge_objects": [], "chain_length": 0, "warnings": []}
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_WELL_SUPPORTED


def test_chain_length_field_overrides_len_edges():
    """chain_length field takes precedence over len(edge_objects)."""
    # Three edges, but chain_length says 4 → should be plausible (all-high at length 4)
    edges = [_high_edge(f"E{i}") for i in range(3)]
    chain = {"edge_objects": edges, "chain_length": 4, "warnings": []}
    result = assign_confidence_tier(chain)
    assert result["confidence_tier"] == TIER_PLAUSIBLE


def test_returns_new_dict():
    """assign_confidence_tier returns a new dict, not the input."""
    chain = _make_chain([_high_edge()])
    result = assign_confidence_tier(chain)
    assert result is not chain


def test_output_contains_tier_and_reason():
    """Output dict must always have both confidence_tier and confidence_reason."""
    for edges in [
        [_high_edge()],
        [_medium_edge()],
        [_low_edge()],
        [_speculative_edge()],
    ]:
        result = assign_confidence_tier(_make_chain(edges))
        assert "confidence_tier" in result
        assert "confidence_reason" in result
        assert isinstance(result["confidence_reason"], str)
        assert len(result["confidence_reason"]) > 0
