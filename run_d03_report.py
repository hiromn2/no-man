#!/usr/bin/env python3
"""D03 report runner for No-Man.

Run:   python run_d03_report.py
Writes:
  output/d03_report.json  — full assembled output (executive summary + all chain reports)
  output/d03_report.txt   — human-readable plain-text version
  output/d03_report.pdf   — compiled PDF (skipped if NOMAN_SKIP_LATEX=1)
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from confidence import assign_confidence_tier
from graph_io import load_graph, load_nodes
from report_generator import assemble_report
from session import (
    attach_chains,
    attach_executive_summary,
    attach_report,
    create_session,
    save_session,
)
from traversal import (
    find_adverse_chains,
    flag_cycle_risks,
    flag_non_monotonicities,
)
from noman_latex import render as _render_latex
from report_adapter import adapt_report

_KNOWLEDGE = Path("knowledge")
_DECISIONS_FILE = Path("decisions") / "decision_types.json"
_OUTPUT_DIR = Path("output")

_RULE_HEAVY = "=" * 70
_RULE_LIGHT = "-" * 70


# ── helpers ───────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gap_placeholder_report() -> dict:
    """Return a sentinel chain-report dict for decision types with zero adverse chains.

    Inserted when traversal finds no chains (known gaps G3, G4, and any future
    decision type with no Japan-specific causal evidence in the seed graph).
    """
    return {
        "report_id": str(uuid.uuid4()),
        "generated_at": _now_iso(),
        "chain_id": "GAP_PLACEHOLDER",
        "confidence_tier": "",
        "language": "ja",
        "causal_chain_text": (
            "現時点では、この意思決定タイプに関する日本固有の因果証拠が不十分です。\n"
            "将来のグラフ更新により対応予定です。（G3/G4ギャップ）"
        ),
        "evidence_text": [],
        "confidence_text": "",
        "flags_text": [],
        "premises_text": None,
        "severity_novelty_reversibility": None,
        "mitigations": [],
        "diagnostics": [],
        "is_gap_placeholder": True,
    }


def _load_decision_type(decision_id: str) -> dict:
    with open(_DECISIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    for dt in data["decision_types"]:
        if dt["id"] == decision_id:
            return dt
    raise ValueError(f"Decision type {decision_id!r} not found in {_DECISIONS_FILE}")


# ── main pipeline ─────────────────────────────────────────────────────────────


def run_d03() -> dict:
    """Run the full D03 session and return the assembled output dict."""

    # ── load graph and nodes ──────────────────────────────────────────────────
    edges = load_graph(_KNOWLEDGE / "seed_graph.json", _KNOWLEDGE / "nodes.json")
    nodes = load_nodes(_KNOWLEDGE / "nodes.json")
    decision_type = _load_decision_type("D03")

    # ── create session ────────────────────────────────────────────────────────
    session = create_session(
        decision_type_id="D03",
        local_context={
            "region": "地方銀行（一般）",
            "interest_rate_regime": "normalizing",
            "note": "金利正常化局面。人口減少が進む地域を想定。",
            "date": datetime.now(timezone.utc).date().isoformat(),
        },
        language="ja",
        sessions_root="sessions",
    )

    # ── find and annotate adverse chains ─────────────────────────────────────
    raw_chains = find_adverse_chains(
        decision_type["primary_node"],
        decision_type["shock_sign"],
        edges,
        nodes,
        max_length=4,
        secondary_nodes=decision_type.get("secondary_nodes"),
    )
    print(f"  Traversal found {len(raw_chains)} adverse chains.")

    annotated: list[dict] = []
    for chain in raw_chains:
        chain = flag_cycle_risks(chain, edges)
        chain = flag_non_monotonicities(chain)
        chain = assign_confidence_tier(chain)
        annotated.append(chain)

    session = attach_chains(session, annotated)

    # ── assemble per-chain reports (one LLM call each) ────────────────────────
    chain_reports: list[dict] = []
    print(f"  Assembling {len(annotated)} chain reports via LLM...")
    for i, chain in enumerate(annotated, 1):
        chain_label = " → ".join(chain["path"])
        print(f"    [{i:2d}/{len(annotated)}] {chain_label}")
        report = assemble_report(
            chain=chain,
            nodes=nodes,
            local_context=session["local_context"],
            decision_type=decision_type,
            language="ja",
        )
        if report.get("interpretive_error"):
            print(f"           ⚠ LLM error: {report.get('error_message', '')}")
        chain_reports.append(report)
        session = attach_report(session, report["report_id"])

    # ── G3/G4 graceful degradation: no chains found ──────────────────────────
    if not chain_reports:
        chain_reports = [_gap_placeholder_report()]
        print("  No adverse chains found — inserting gap placeholder (G3/G4).")

    # ── generate executive summary (one LLM call) ────────────────────────────
    print("  Generating executive summary...")
    session = attach_executive_summary(session, chain_reports, decision_type)
    if session.get("executive_summary_error"):
        print(f"  ⚠ Executive summary error: {session['executive_summary_error']}")
    else:
        print("  Executive summary OK.")

    # ── persist session ───────────────────────────────────────────────────────
    save_session(session)

    return {
        "session": session,
        "decision_type": decision_type,
        "chain_reports": chain_reports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── writers ───────────────────────────────────────────────────────────────────


def write_json(output: dict) -> Path:
    _OUTPUT_DIR.mkdir(exist_ok=True)
    path = _OUTPUT_DIR / "d03_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


def write_text(output: dict) -> Path:
    _OUTPUT_DIR.mkdir(exist_ok=True)
    path = _OUTPUT_DIR / "d03_report.txt"

    session = output["session"]
    dt = output["decision_type"]
    chain_reports: list[dict] = output["chain_reports"]
    generated_at = output["generated_at"]
    n = len(chain_reports)

    lines: list[str] = []

    # ── header ────────────────────────────────────────────────────────────────
    lines += [
        _RULE_HEAVY,
        f"D03 報告書 — {dt.get('name_ja', '')}",
        f"生成日時: {generated_at}",
        f"連鎖数: {n}  |  Session: {session.get('session_id', 'N/A')}",
        _RULE_HEAVY,
        "",
    ]

    # ── executive summary ─────────────────────────────────────────────────────
    lines.append("【経営陣向けサマリー】")
    lines.append("")
    summary = session.get("executive_summary_ja")
    if summary:
        lines.append(summary)
    else:
        err = session.get("executive_summary_error", "原因不明")
        lines.append(f"（生成エラー: {err}）")
    lines.append("")
    featured = session.get("featured_chain_id")
    if featured:
        lines.append(f"（最重要連鎖ID: {featured}）")
    lines += ["", _RULE_HEAVY, ""]

    # ── per-chain sections ────────────────────────────────────────────────────
    for i, report in enumerate(chain_reports, 1):
        chain_id = report.get("chain_id", "N/A")
        tier = report.get("confidence_tier", "")
        chain_text = report.get("causal_chain_text", "")
        is_featured = (chain_id == featured)
        featured_marker = "  ★ 経営陣向けサマリー参照連鎖" if is_featured else ""

        lines += [
            f"[{i}/{n}]  Chain: {chain_id}{featured_marker}",
            f"Tier: {tier}",
            "",
            "【因果連鎖】",
            f"  {chain_text}",
            "",
        ]

        # Flags
        flags = report.get("flags_text", [])
        if flags:
            lines.append("【注意事項】")
            for flag in flags:
                lines.append(f"  · {flag}")
            lines.append("")

        if report.get("interpretive_error"):
            lines += [
                f"  （解釈セクション生成エラー: {report.get('error_message', '')}）",
                "",
            ]
        else:
            # Premises
            premises = report.get("premises_text")
            if premises:
                lines += ["【前提】", f"  {premises}", ""]

            # Severity / novelty / reversibility
            snr = report.get("severity_novelty_reversibility") or {}
            if snr:
                lines.append("【重大性・新規性・可逆性】")
                for key_ja, key_en, just_key in (
                    ("重大性", "severity", "severity_justification"),
                    ("新規性", "novelty", "novelty_justification"),
                    ("可逆性", "reversibility", "reversibility_justification"),
                ):
                    val = snr.get(key_en, "—")
                    just = snr.get(just_key, "")
                    lines.append(f"  {key_ja}: {val}  — {just}")
                lines.append("")

            # Mitigations
            mitigations = report.get("mitigations", [])
            if mitigations:
                lines.append("【緩和策】")
                for j, m in enumerate(mitigations, 1):
                    lines.append(f"  {j}. {m.get('action_ja', '（記述なし）')}")
                    targets = m.get("targets_edge", "")
                    if targets:
                        lines.append(f"     対象エッジ: {targets}")
                    adverse = m.get("adverse_effects_note", "")
                    if adverse:
                        lines.append(f"     一次的悪影響: {adverse}")
                    depth = m.get("depth_note", "")
                    if depth:
                        lines.append(f"     {depth}")
                lines.append("")
            else:
                lines += ["【緩和策】", "  （なし）", ""]

            # Diagnostics
            diagnostics = report.get("diagnostics", [])
            if diagnostics:
                lines.append("【診断的推奨事項】")
                for d in diagnostics:
                    lines.append(f"  · 監視変数: {d.get('variable_ja', '')}")
                    if d.get("data_source"):
                        lines.append(f"    データソース: {d['data_source']}")
                    if d.get("test_suggested"):
                        lines.append(f"    推奨検証手法: {d['test_suggested']}")
                    if d.get("interpretation"):
                        lines.append(f"    解釈: {d['interpretation']}")
                lines.append("")

        lines += [_RULE_LIGHT, ""]

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ── entrypoint ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("No-Man — D03 Report Runner")
    print(_RULE_HEAVY)

    output = run_d03()

    json_path = write_json(output)
    txt_path = write_text(output)

    # Adapt schema, then render PDF via new noman_report.tex template.
    adapted = adapt_report(output)
    pdf_path = _render_latex(adapted, output_path=json_path.with_suffix(".pdf"))

    print(f"\n  JSON → {json_path}")
    print(f"  TXT  → {txt_path}")
    if pdf_path:
        print(f"  PDF  → {pdf_path}")
    print("")
    print(_RULE_HEAVY)
    print(txt_path.read_text(encoding="utf-8"))
