"""Hybrid report rendering for No-Man adverse chain reports.

Two-stage architecture:
  1. render_structural_sections — pure Python, deterministic, no LLM.
     Formats the causal chain, evidence, confidence tier, and flags from the
     chain dict and nodes dict into display-ready strings.

  2. generate_interpretive_sections — LLM call via Anthropic API.
     Writes the interpretive prose: premises, severity/novelty/reversibility,
     mitigations, and empirical diagnostics. Wraps the API call in try/except
     so that a failed call returns a partial report with interpretive_error=True.

  3. assemble_report — combines both stages into a full report dict matching
     the schema defined in CLAUDE.md section 6.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_MODEL = "claude-sonnet-4-6"


# ── executive summary constants ───────────────────────────────────────────────

# Ranking used to identify the most severe chain when building the executive summary.
# Higher number = more severe / more credible.
_TIER_RANK: dict[str, int] = {"well-supported": 3, "plausible": 2, "worth-considering": 1}
_SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}

_EXEC_SUMMARY_SYSTEM_JA = (
    "あなたは日本の地域銀行の経営陣に向けた簡潔なリスク概要を作成する専門家です。\n"
    "与えられた因果連鎖データに基づき、最も重大な悪影響の経路を2〜3文の平易な日本語で要約してください。\n"
    "断定的な表現は使用せず、「〜の可能性があります」「〜と考えられます」「〜が示唆されます」等の表現を用いてください。\n"
    "「銀行は〜すべきです」のような強い命令形は使用しないでください。\n"
    "出力は日本語の文章のみとし、JSON形式や箇条書きは使用しないでください。"
)

# ── tier opening phrases ───────────────────────────────────────────────────────

_TIER_OPENING_JA: dict[str, str] = {
    "well-supported": "文献に裏付けられた因果経路として、以下が特定されました。",
    "plausible": "以下の因果経路は、複数の証拠と整合的であり、注目に値します。",
    "worth-considering": "以下は証拠が限定的な経路ですが、見落とされがちなリスクとして提示します。",
}

_TIER_OPENING_EN: dict[str, str] = {
    "well-supported": "The following causal pathway is well-supported by the literature.",
    "plausible": "The following causal pathway is consistent with multiple lines of evidence and warrants attention.",
    "worth-considering": "The following pathway has limited evidentiary support but is presented as an easily-overlooked risk.",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── public API ────────────────────────────────────────────────────────────────


def render_structural_sections(
    chain: dict,
    nodes: dict,
    language: str = "ja",
) -> dict:
    """Pure Python. No LLM. Deterministic.

    Returns a dict with keys:
      causal_chain_text — formatted chain string using node names and edge signs
      evidence_text     — list of (edge_name, source_note) tuples
      confidence_text   — tier opening phrase + reason in target language
      flags_text        — list of warning/flag strings in target language
    """
    name_field = "name_ja" if language == "ja" else "name_en"
    tier_openings = _TIER_OPENING_JA if language == "ja" else _TIER_OPENING_EN

    # ── causal chain text ─────────────────────────────────────────────────────
    path: list[str] = chain["path"]
    signs: list[str] = chain["signs"]

    parts: list[str] = [_node_name(nodes, path[0], name_field)]
    for i, sign in enumerate(signs):
        parts.append(f" →[{sign}]→ ")
        parts.append(_node_name(nodes, path[i + 1], name_field))
    causal_chain_text = "".join(parts)

    # ── evidence text ─────────────────────────────────────────────────────────
    evidence_text: list[tuple[str, str]] = []
    for edge in chain["edge_objects"]:
        edge_name = edge.get(f"name_{language}", edge.get("name_en", ""))
        if edge.get("status") == "speculative":
            proposed_by = edge.get("proposed_by", "不明" if language == "ja" else "unknown")
            if language == "ja":
                note = f"投機的（{proposed_by}による提案）"
            else:
                note = f"Speculative (proposed by {proposed_by})"
        else:
            sources = edge.get("sources", [])
            if sources:
                note = ", ".join(sources)
            else:
                note = "出典なし" if language == "ja" else "No sources listed"
        evidence_text.append((edge_name, note))

    # ── confidence text ───────────────────────────────────────────────────────
    tier = chain.get("confidence_tier", "")
    opening = tier_openings.get(tier, "")
    reason = chain.get("confidence_reason", "")
    confidence_text = f"{opening}\n{reason}" if reason else opening

    # ── flags text ────────────────────────────────────────────────────────────
    flags_text: list[str] = []

    if chain.get("cycle_risk"):
        note = chain.get("cycle_risk_note", "")
        if language == "ja":
            flags_text.append(f"【循環リスク】{note}")
        else:
            flags_text.append(f"[Cycle Risk] {note}")

    for warning in chain.get("warnings", []):
        msg = warning.get("message", "")
        wtype = warning.get("type", "")
        if language == "ja":
            prefix_map = {
                "non_monotone": "【非単調性】",
                "research_gap": "【研究ギャップ】",
                "international_evidence": "【国際証拠の転用可能性】",
                "length_warning": "【長鎖警告】",
            }
            prefix = prefix_map.get(wtype, "【警告】")
        else:
            prefix_map = {
                "non_monotone": "[Non-monotone] ",
                "research_gap": "[Research Gap] ",
                "international_evidence": "[International Evidence] ",
                "length_warning": "[Length Warning] ",
            }
            prefix = prefix_map.get(wtype, "[Warning] ")
        flags_text.append(f"{prefix}{msg}")

    return {
        "causal_chain_text": causal_chain_text,
        "evidence_text": evidence_text,
        "confidence_text": confidence_text,
        "flags_text": flags_text,
    }


def generate_interpretive_sections(
    chain: dict,
    structural: dict,
    local_context: dict,
    decision_type: dict,
    language: str = "ja",
) -> dict:
    """LLM call via Anthropic API. Returns dict with keys:
      premises_text, severity_novelty_reversibility, mitigations, diagnostics.

    On any failure returns a partial dict with interpretive_error=True.
    """
    try:
        if anthropic is None:
            raise ImportError("anthropic package is not installed")

        prompt_file = "report_ja.txt" if language == "ja" else "report_en.txt"
        prompt_path = _PROMPTS_DIR / prompt_file
        with open(prompt_path, encoding="utf-8") as f:
            system_prompt = f.read()

        causal_chain_text = structural.get("causal_chain_text", "")
        evidence_lines = "\n".join(
            f"  - {name}: {note}"
            for name, note in structural.get("evidence_text", [])
        )
        confidence_text = structural.get("confidence_text", "")
        flags = structural.get("flags_text", [])
        flags_lines = (
            "\n".join(f"  - {f}" for f in flags)
            if flags
            else ("  なし" if language == "ja" else "  None")
        )

        if language == "ja":
            decision_label = decision_type.get("name_ja", decision_type.get("name_en", ""))
            user_message = (
                f"## 意思決定タイプ\n{decision_label}\n\n"
                f"## 因果連鎖\n{causal_chain_text}\n\n"
                f"## 根拠\n{evidence_lines}\n\n"
                f"## 信頼度\n{confidence_text}\n\n"
                f"## 注意事項\n{flags_lines}\n\n"
                f"## ローカルコンテキスト\n"
                f"{json.dumps(local_context, ensure_ascii=False, indent=2)}\n"
            )
        else:
            decision_label = decision_type.get("name_en", "")
            user_message = (
                f"## Decision Type\n{decision_label}\n\n"
                f"## Causal Chain\n{causal_chain_text}\n\n"
                f"## Evidence\n{evidence_lines}\n\n"
                f"## Confidence\n{confidence_text}\n\n"
                f"## Flags\n{flags_lines}\n\n"
                f"## Local Context\n"
                f"{json.dumps(local_context, ensure_ascii=False, indent=2)}\n"
            )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = response.content[0].text.strip()
        if response_text.startswith("```"):
            parts = response_text.split("```", 2)
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            response_text = inner.rsplit("```", 1)[0].strip()

        return json.loads(response_text)

    except Exception as exc:
        return {
            "interpretive_error": True,
            "error_message": str(exc),
            "premises_text": None,
            "severity_novelty_reversibility": None,
            "mitigations": [],
            "diagnostics": [],
        }


def assemble_report(
    chain: dict,
    nodes: dict,
    local_context: dict,
    decision_type: dict,
    language: str = "ja",
) -> dict:
    """Combines structural + interpretive sections into a full report dict.

    Matches the schema from CLAUDE.md section 6. Also adds:
      report_id       — UUID
      generated_at    — ISO timestamp
      chain_id        — from chain dict, or a new UUID if absent
      confidence_tier — from chain dict
      language        — as passed
    """
    structural = render_structural_sections(chain, nodes, language=language)
    interpretive = generate_interpretive_sections(
        chain=chain,
        structural=structural,
        local_context=local_context,
        decision_type=decision_type,
        language=language,
    )

    return {
        "report_id": str(uuid.uuid4()),
        "generated_at": _now_iso(),
        "chain_id": chain.get("chain_id", str(uuid.uuid4())),
        "confidence_tier": chain.get("confidence_tier", ""),
        "language": language,
        # structural sections
        "causal_chain_text": structural["causal_chain_text"],
        "evidence_text": structural["evidence_text"],
        "confidence_text": structural["confidence_text"],
        "flags_text": structural["flags_text"],
        # interpretive sections (from LLM or error fallback)
        **interpretive,
    }


def generate_executive_summary(
    chain_reports: list[dict],
    decision_type: dict,
    language: str = "ja",
) -> dict:
    """Generate a 経営陣向けサマリー (executive summary) for the most severe finding.

    Intended to appear at the TOP of the full session report, before the detailed
    chain list. Identifies the most severe chain from chain_reports by confidence_tier
    (well-supported > plausible > worth-considering) and severity rating (high > medium
    > low), then calls the LLM to produce a 2–3 sentence plain-Japanese summary.

    Only Japanese is supported (language="en" is accepted but the summary is always
    written in Japanese because this section is designed for Japanese board reporting).

    Args:
        chain_reports: List of assembled report dicts from assemble_report().
                       May be empty — returns None summary if no chains.
        decision_type: Decision type dict (name_ja, name_en, description_ja, etc.)
        language:      "ja" (default) or "en". Summary is always in Japanese.

    Returns:
        Dict with keys:
          executive_summary_ja  — 2–3 sentence summary string (None if no chains)
          featured_chain_id     — chain_id of the chain used as the basis (None if empty)
        On API failure:
          executive_summary_ja  — None
          featured_chain_id     — chain_id of the selected chain (if selection succeeded)
          interpretive_error    — True
          error_message         — str
    """
    if not chain_reports:
        return {
            "executive_summary_ja": None,
            "featured_chain_id": None,
        }

    featured = _select_most_severe_chain(chain_reports)
    if featured is None:
        return {
            "executive_summary_ja": None,
            "featured_chain_id": None,
        }

    try:
        if anthropic is None:
            raise ImportError("anthropic package is not installed")

        decision_label = decision_type.get("name_ja") or decision_type.get("name_en", "")
        chain_text = featured.get("causal_chain_text", "")
        tier = featured.get("confidence_tier", "")
        snr = featured.get("severity_novelty_reversibility") or {}
        severity = snr.get("severity", "不明")
        severity_note = snr.get("severity_justification", "")

        user_message = (
            f"## 意思決定タイプ\n{decision_label}\n\n"
            f"## 最も重大な因果連鎖\n{chain_text}\n\n"
            f"## 信頼度ティア\n{tier}\n\n"
            f"## 重大性\n{severity}：{severity_note}\n\n"
            "上記の情報をもとに、経営陣向けのリスク概要を2〜3文で記述してください。"
        )

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_EXEC_SUMMARY_SYSTEM_JA,
            messages=[{"role": "user", "content": user_message}],
        )

        summary_text = response.content[0].text.strip()

        return {
            "executive_summary_ja": summary_text,
            "featured_chain_id": featured.get("chain_id"),
        }

    except Exception as exc:
        return {
            "executive_summary_ja": None,
            "featured_chain_id": featured.get("chain_id") if featured else None,
            "interpretive_error": True,
            "error_message": str(exc),
        }


# ── private helpers ───────────────────────────────────────────────────────────


def _select_most_severe_chain(chain_reports: list[dict]) -> dict | None:
    """Return the most severe chain report by confidence tier, then by severity rating.

    Tier order:     well-supported (3) > plausible (2) > worth-considering (1).
    Severity order: high (3) > medium (2) > low (1).
    Chains with no severity_novelty_reversibility field default to medium (2).
    Returns None only when chain_reports is empty.
    """
    if not chain_reports:
        return None

    def _sort_key(r: dict) -> tuple[int, int]:
        tier = _TIER_RANK.get(r.get("confidence_tier", ""), 0)
        snr = r.get("severity_novelty_reversibility") or {}
        severity = _SEVERITY_RANK.get(snr.get("severity", "medium"), 2)
        return (tier, severity)

    return max(chain_reports, key=_sort_key)


def _node_name(nodes: dict, node_id: str, name_field: str) -> str:
    node = nodes.get(node_id, {})
    return node.get(name_field) or node.get("name_en") or node_id
