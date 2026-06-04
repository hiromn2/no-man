"""Confidence tier assignment for adverse causal chains, per CLAUDE.md section 5.

The tier is a joint function of edge confidence profile and chain length.
Exact table (do not deviate):

    Edge profile           | Length ≤ 3     | Length = 4     | Length > 4
    -----------------------+----------------+----------------+-------------------
    All high, all tested   | well-supported | plausible      | worth-considering
    Mix high/medium tested | plausible      | worth-consid.  | worth-considering
    Any low OR speculative | worth-consid.  | worth-consid.  | worth-considering

Rules:
  • well-supported  — all edges high-confidence AND all tested; length ≤ 3
  • plausible       — no speculative, no low; length ≤ 3 (mix allowed) OR
                      all-high all-tested at length 4
  • worth-considering — any low OR any speculative (any length), OR
                        mix high/medium at length ≥ 4, OR
                        any profile at length > 4

A length_warning is appended to chain["warnings"] for chains longer than 4.
"""

from __future__ import annotations

TIER_WELL_SUPPORTED = "well-supported"
TIER_PLAUSIBLE = "plausible"
TIER_WORTH_CONSIDERING = "worth-considering"

_MAX_WELL_SUPPORTED_LENGTH = 3   # inclusive upper bound for well-supported / plausible
_MAX_PLAUSIBLE_LENGTH = 4        # inclusive upper bound for plausible (all-high only)
_LENGTH_WARNING_THRESHOLD = 4    # length > this → mandatory length warning


def assign_confidence_tier(chain: dict) -> dict:
    """Enrich a chain dict with confidence_tier and confidence_reason.

    Returns a new dict. Does not mutate the input chain.
    Appends a length_warning entry to chain["warnings"] when chain_length > 4.
    """
    result = dict(chain)
    edges: list[dict] = chain.get("edge_objects", [])
    length: int = chain.get("chain_length", len(edges))

    # ── classify edge population ─────────────────────────────────────────────
    speculative_edges = [e for e in edges if e.get("status") == "speculative"]
    low_edges = [e for e in edges if e.get("confidence") == "low"]
    medium_edges = [e for e in edges if e.get("confidence") == "medium"]

    has_speculative = bool(speculative_edges)
    has_low = bool(low_edges)
    all_high = not medium_edges and not low_edges and not speculative_edges
    # (no need to check all_tested separately; speculative is handled above)

    # ── determine tier ────────────────────────────────────────────────────────

    if has_speculative or has_low:
        # Row 3: Any low OR any speculative → worth-considering always
        tier = TIER_WORTH_CONSIDERING
        parts: list[str] = []
        if has_speculative:
            ids = [e.get("id", "?") for e in speculative_edges]
            parts.append(f"speculative edge(s) {ids}")
        if has_low:
            ids = [e.get("id", "?") for e in low_edges]
            parts.append(f"low-confidence edge(s) {ids}")
        reason = (
            f"Assigned {tier}: chain contains {' and '.join(parts)}. "
            "Any speculative or low-confidence edge mandates worth-considering "
            "regardless of chain length."
        )

    elif all_high:
        # Row 1: All high, all tested
        if length <= _MAX_WELL_SUPPORTED_LENGTH:
            tier = TIER_WELL_SUPPORTED
            reason = (
                f"Assigned {tier}: all {len(edges)} edge(s) are high-confidence "
                f"and tested; chain length {length} ≤ {_MAX_WELL_SUPPORTED_LENGTH}."
            )
        elif length == _MAX_PLAUSIBLE_LENGTH:
            tier = TIER_PLAUSIBLE
            reason = (
                f"Assigned {tier}: all {len(edges)} edges are high-confidence and tested, "
                f"but chain length {length} = {_MAX_PLAUSIBLE_LENGTH} "
                f"downgrades well-supported → plausible."
            )
        else:
            # length > 4
            tier = TIER_WORTH_CONSIDERING
            reason = (
                f"Assigned {tier}: all edges are high-confidence and tested, "
                f"but chain length {length} > {_LENGTH_WARNING_THRESHOLD} "
                "mandates worth-considering regardless of edge quality."
            )

    else:
        # Row 2: Mix high/medium, all tested (no low, no speculative confirmed above)
        medium_ids = [e.get("id", "?") for e in medium_edges]
        if length <= _MAX_WELL_SUPPORTED_LENGTH:
            tier = TIER_PLAUSIBLE
            reason = (
                f"Assigned {tier}: all edges are tested but chain includes "
                f"medium-confidence edge(s) {medium_ids}; "
                f"chain length {length} ≤ {_MAX_WELL_SUPPORTED_LENGTH}."
            )
        else:
            # length ≥ 4 with medium edges
            tier = TIER_WORTH_CONSIDERING
            reason = (
                f"Assigned {tier}: chain includes medium-confidence edge(s) {medium_ids} "
                f"and chain length {length} ≥ {_MAX_PLAUSIBLE_LENGTH}; "
                "combination of medium confidence and length ≥ 4 downgrades plausible → "
                "worth-considering."
            )

    result["confidence_tier"] = tier
    result["confidence_reason"] = reason

    # ── length warning ────────────────────────────────────────────────────────
    if length > _LENGTH_WARNING_THRESHOLD:
        warnings = list(result.get("warnings", []))
        warnings.append({
            "type": "length_warning",
            "message": (
                f"Chain length {length} exceeds {_LENGTH_WARNING_THRESHOLD} edges. "
                "Confidence in multi-step causal inference degrades substantially with "
                "chain length. This chain is reported as worth-considering regardless "
                "of individual edge confidence."
            ),
        })
        result["warnings"] = warnings

    return result
