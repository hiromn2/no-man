r"""LaTeX/PDF renderer for No-Man adverse chain reports.

Compile a No-Man output JSON into a PDF using XeLaTeX.
Set NOMAN_SKIP_LATEX=1 to skip rendering without error (for CI / no-XeLaTeX envs).

Two rendering modes
-------------------
Legacy path (backward-compat):
    pdf = render(Path("output/d03_report.json"))
    # Reads JSON, builds inline LaTeX template, writes d03_report.pdf.

Adapted-schema path (new noman_report.tex template):
    from report_adapter import adapt_report
    adapted = adapt_report(raw_output)
    pdf = render(adapted, output_path=Path("output/d03_report.pdf"))
    # Uses noman_report.tex: dashboard strip, tcolorbox blocks, no [+] notation.

The module also exposes:
    chains_to_tikz(chains, nodes) -> str   — TikZ tikzpicture string
    infer_severity(chain)         -> str   — "high"/"medium"/"low" heuristic
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

# ── constants ─────────────────────────────────────────────────────────────────

_DEFAULT_NODES_PATH = Path(__file__).parent.parent / "knowledge" / "nodes.json"
_TEMPLATE_PATH = Path(__file__).parent.parent / "noman_report.tex"

# Confidence tier → (colback, colframe) for tcolorbox
_TIER_COLORS: dict[str, tuple[str, str]] = {
    "well-supported":    ("tierwell!8",      "tierwell!60"),
    "plausible":         ("tierplausible!8", "tierplausible!50"),
    "worth-considering": ("tierworth!8",     "tierworth!50"),
}

# Severity → (Japanese label, color name)
_SEVERITY_DISPLAY: dict[str, tuple[str, str]] = {
    "high":   ("高", "sevhigh"),
    "medium": ("中", "sevmedium"),
    "low":    ("低", "sevlow"),
}

# ── LaTeX template ────────────────────────────────────────────────────────────
# Raw string: LaTeX backslashes are literals, not Python escape sequences.

_LATEX_TEMPLATE = r"""
\documentclass[a4paper,12pt]{article}
\usepackage{fontspec}
\usepackage{xeCJK}
\setCJKmainfont{Hiragino Mincho ProN}
\setCJKsansfont{Hiragino Sans}
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{parskip}
\usepackage{fancyhdr}
\usepackage{hyperref}

\setlength{\parindent}{0pt}

\pagestyle{fancy}
\fancyhf{}
\rhead{No-Man 因果リスク分析}
\lhead{\VAR{decision_type_name_ja}}
\cfoot{\thepage}

\begin{document}

\begin{center}
  {\Large \textbf{No-Man 因果リスク分析レポート}}\\[0.5em]
  {\large \VAR{decision_type_name_ja}}\\[0.3em]
  {\small 生成日時: \VAR{generated_at}}\\[0.2em]
  {\small セッションID: \VAR{session_id}}\\[0.2em]
  {\small 検出連鎖数: \VAR{chain_count}}
\end{center}

\hrule
\vspace{1em}

\section*{経営陣向けサマリー}
\VAR{executive_summary}

\vspace{1em}
\section*{因果連鎖グラフ（全連鎖）}

\begin{figure}[h]
\centering
\resizebox{\textwidth}{!}{%
\VAR{tikz_dag}
}
\caption{意思決定から悪影響ノードへの因果経路（青: 正の効果、赤: 負の効果）}
\end{figure}

\clearpage
\section*{詳細分析}
\VAR{chain_sections}

\end{document}
"""

# ── public API ────────────────────────────────────────────────────────────────


def infer_severity(chain: dict) -> str:
    """Infer a severity label from chain properties when the LLM assessment is absent.

    Uses ``confidence_tier`` and chain length as proxies for adverse impact.
    Returns one of ``"high"``, ``"medium"``, or ``"low"``.

    Args:
        chain: A session chain dict (has ``chain_length``, ``confidence_tier``)
               or an adapted chain dict (has ``path``, ``confidence_tier``).
               Unknown keys are ignored gracefully.
    """
    tier = chain.get("confidence_tier", "")
    try:
        length = int(chain.get("chain_length", len(chain.get("path", [])) - 1))
    except (TypeError, ValueError):
        length = 2
    if tier == "well-supported" and length <= 2:
        return "high"
    if tier in ("well-supported", "plausible") and length <= 3:
        return "medium"
    return "low"


def chains_to_tikz(chains: list[dict], nodes: dict) -> str:
    r"""Generate a TikZ tikzpicture string from an adverse_chains list.

    Nodes are laid out left-to-right: decision (source) node on the left,
    terminal adverse outcome on the right, intermediate nodes between.
    Shared nodes across chains are deduplicated — each node appears once.
    Node labels use name_ja from the nodes dict, falling back to the node id.
    Edge color: blue for positive sign (+), red for negative sign (-).

    Args:
        chains: List of chain dicts from session["chains"]. Each must have
                ``path`` (list of node IDs) and ``signs`` (list of edge signs).
        nodes:  Dict keyed by node ID as returned by graph_io.load_nodes().

    Returns:
        A complete TikZ tikzpicture string for inclusion in a LaTeX document.
        When chains is empty, returns a single placeholder node.
    """
    if not chains:
        return (
            r"\begin{tikzpicture}" + "\n"
            r"  \node[draw, rounded corners, text width=10cm, align=center," + "\n"
            r"        font=\normalsize] at (0,0)" + "\n"
            r"    {因果連鎖なし（G3/G4ギャップ）};" + "\n"
            r"\end{tikzpicture}"
        )

    # ── assign each node its minimum depth (= column index) ──────────────────
    node_min_depth: dict[str, int] = {}
    first_seen_order: list[str] = []

    for chain in chains:
        path = chain.get("path", [])
        for depth, node_id in enumerate(path):
            if node_id not in node_min_depth:
                node_min_depth[node_id] = depth
                first_seen_order.append(node_id)
            elif depth < node_min_depth[node_id]:
                node_min_depth[node_id] = depth

    # ── group nodes by column ─────────────────────────────────────────────────
    columns: dict[int, list[str]] = defaultdict(list)
    for node_id in first_seen_order:
        columns[node_min_depth[node_id]].append(node_id)

    # ── compute (x, y) positions ──────────────────────────────────────────────
    col_width = 5.5    # cm between column centres
    row_height = 2.5   # cm between row centres

    positions: dict[str, tuple[float, float]] = {}
    for col_idx in sorted(columns.keys()):
        col_nodes = columns[col_idx]
        n = len(col_nodes)
        for row_idx, node_id in enumerate(col_nodes):
            x = col_idx * col_width
            y = -(row_idx - (n - 1) / 2.0) * row_height
            positions[node_id] = (x, y)

    # ── assign short TikZ node names (N0, N1, …) to avoid special-char issues ─
    tikz_name: dict[str, str] = {
        nid: f"N{i}" for i, nid in enumerate(first_seen_order)
    }

    # ── collect unique (from, to) edges with their sign ──────────────────────
    edges: dict[tuple[str, str], str] = {}
    for chain in chains:
        path = chain.get("path", [])
        signs = chain.get("signs", [])
        for i, sign in enumerate(signs):
            key = (path[i], path[i + 1])
            if key not in edges:
                edges[key] = sign

    # ── build TikZ source lines ───────────────────────────────────────────────
    lines: list[str] = [
        r"\begin{tikzpicture}[",
        r"  every node/.style={draw, rounded corners, text width=3cm,"
        r" align=center, font=\small},",
        r"  ->, >=Stealth, thick",
        r"]",
    ]

    for node_id in first_seen_order:
        x, y = positions[node_id]
        label = _node_label(node_id, nodes)
        name = tikz_name[node_id]
        lines.append(f"  \\node ({name}) at ({x:.1f},{y:.1f}) {{{label}}};")

    for (from_id, to_id), sign in edges.items():
        color = "blue" if sign == "+" else "red"
        fn = tikz_name[from_id]
        tn = tikz_name[to_id]
        lines.append(f"  \\draw[{color}] ({fn}) -- ({tn});")

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def render(
    source: Union[str, Path, dict],
    output_path: Union[str, Path, None] = None,
    nodes_path: Union[str, Path, None] = None,
) -> Path | None:
    r"""Compile a No-Man report into a PDF using XeLaTeX.

    Two calling modes
    -----------------
    **Legacy** — pass a path to the raw pipeline JSON::

        pdf = render(Path("output/d03_report.json"))

    **Adapted** — pass the dict returned by ``report_adapter.adapt_report``
    together with an explicit output path::

        pdf = render(adapted_dict, output_path=Path("output/d03_report.pdf"))

    The adapted mode uses ``noman_report.tex`` (dashboard strip, tcolorbox
    chain blocks, no ``[+]`` sign notation).  The legacy mode uses the inline
    template embedded in this module.

    Args:
        source:      JSON path (str/Path) **or** adapted dict.
        output_path: Destination for the PDF.  Required when *source* is a dict;
                     auto-derived from *source* stem when *source* is a path.
        nodes_path:  Path to ``knowledge/nodes.json``.
                     Auto-detected when None (legacy mode only).

    Returns:
        Path to the generated PDF, or None when ``NOMAN_SKIP_LATEX=1``.

    Raises:
        RuntimeError: ``xelatex`` is not on PATH (includes install instructions).
        RuntimeError: ``xelatex`` ran but produced no PDF (includes log tail).
        ValueError:   *source* is a dict but *output_path* was not supplied.
    """
    if os.environ.get("NOMAN_SKIP_LATEX"):
        print(
            "NOMAN_SKIP_LATEX=1 — skipping PDF generation. "
            "Unset this variable and ensure XeLaTeX is installed to generate PDFs."
        )
        return None

    if not shutil.which("xelatex"):
        raise RuntimeError(
            "xelatex not found on PATH.\n"
            "To generate PDFs, install XeLaTeX:\n"
            "  macOS:  brew install --cask mactex\n"
            "          (or: brew install basictex, then: "
            "sudo tlmgr install collection-langcjk)\n"
            "  Linux:  sudo apt-get install texlive-xetex texlive-lang-japanese\n"
            "Set NOMAN_SKIP_LATEX=1 to skip PDF generation without error."
        )

    # ── route by source type ──────────────────────────────────────────────────
    if isinstance(source, dict):
        # ── Adapted-schema path ───────────────────────────────────────────────
        if output_path is None:
            raise ValueError(
                "output_path is required when source is an adapted dict. "
                "Pass the desired PDF destination, e.g. Path('output/d03_report.pdf')."
            )
        pdf_dst = Path(output_path)
        nodes = _load_nodes_safe(nodes_path or _DEFAULT_NODES_PATH)
        chains_for_tikz = [
            {"path": c.get("path", []), "signs": c.get("signs", [])}
            for c in source.get("chains", [])
        ]
        tikz_str = chains_to_tikz(chains_for_tikz, nodes)
        latex_source = _build_latex_adapted(source, tikz_str)

    else:
        # ── Legacy path-based path ────────────────────────────────────────────
        json_path = Path(source)
        pdf_dst = Path(output_path) if output_path else json_path.with_suffix(".pdf")

        if nodes_path is None:
            candidate = json_path.parent.parent / "knowledge" / "nodes.json"
            nodes_path = candidate if candidate.exists() else _DEFAULT_NODES_PATH
        nodes = _load_nodes_safe(nodes_path)

        with open(json_path, encoding="utf-8") as f:
            report_data = json.load(f)

        chains = report_data.get("session", {}).get("chains", [])
        tikz_str = chains_to_tikz(chains, nodes)
        latex_source = _build_latex(report_data, tikz_str)

    _compile_latex(latex_source, pdf_dst)
    return pdf_dst


# ── private helpers ───────────────────────────────────────────────────────────


def _load_nodes_safe(nodes_path: Union[str, Path]) -> dict:
    """Load nodes.json; return empty dict on any failure (so rendering still works)."""
    try:
        import sys
        _src = Path(__file__).parent
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from graph_io import load_nodes
        return load_nodes(nodes_path)
    except Exception:
        try:
            with open(nodes_path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("nodes", data)
            if isinstance(raw, list):
                return {n["id"]: n for n in raw}
        except Exception:
            pass
    return {}


def _node_label(node_id: str, nodes: dict) -> str:
    """Return the TikZ display label for a node (Japanese name_ja preferred)."""
    node = nodes.get(node_id, {})
    name_ja = node.get("name_ja")
    if name_ja:
        # Japanese text: no LaTeX special-char escaping needed for CJK chars.
        return name_ja
    return _escape_latex(node_id)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in a plain ASCII string."""
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("$", r"\$"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("^", r"\^{}"),
        ("~", r"\~{}"),
    ):
        text = text.replace(old, new)
    return text


def _escape_latex_text(text: str) -> str:
    """Escape LaTeX special characters in mixed text (including Japanese).

    Japanese/CJK characters are safe in XeLaTeX with xeCJK and need no escaping.
    Only ASCII LaTeX specials are replaced.
    """
    if not text:
        return ""
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("$", r"\$"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("^", r"\^{}"),
        ("~", r"\~{}"),
    ):
        text = text.replace(old, new)
    return text


def _chain_section_latex(report: dict, idx: int, total: int) -> str:
    """Render one chain report dict as a LaTeX subsection block."""
    if report.get("is_gap_placeholder"):
        text = _escape_latex_text(report.get("causal_chain_text", ""))
        return (
            f"\\subsection*{{[{idx}/{total}]\\ 因果証拠不十分}}\n"
            f"{text}\n\n"
        )

    chain_id = report.get("chain_id", "N/A")
    tier = _escape_latex_text(report.get("confidence_tier", ""))
    chain_text = _escape_latex_text(report.get("causal_chain_text", ""))
    flags = report.get("flags_text", [])
    evidence = report.get("evidence_text", [])

    # Shorten chain_id for display (first 8 chars are enough to be unique).
    short_id = _escape_latex(str(chain_id)[:8])

    lines: list[str] = [
        f"\\subsection*{{[{idx}/{total}]\\ 連鎖 \\texttt{{{short_id}}}}}",
        f"\\textbf{{信頼度ティア:}} {tier}",
        "",
        "\\textbf{【因果連鎖】}",
        f"\\begin{{quote}}{chain_text}\\end{{quote}}",
    ]

    if evidence:
        lines += ["\\textbf{【根拠】}", "\\begin{itemize}"]
        for pair in evidence:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                edge_name, note = pair
            else:
                edge_name, note = str(pair), ""
            lines.append(
                f"  \\item {_escape_latex_text(edge_name)}: "
                f"{_escape_latex_text(note)}"
            )
        lines.append("\\end{itemize}")

    if flags:
        lines += ["\\textbf{【注意事項】}", "\\begin{itemize}"]
        for flag in flags:
            lines.append(f"  \\item {_escape_latex_text(flag)}")
        lines.append("\\end{itemize}")

    if not report.get("interpretive_error"):
        premises = report.get("premises_text")
        if premises:
            lines += [
                "\\textbf{【前提】}",
                _escape_latex_text(str(premises)),
                "",
            ]

        snr = report.get("severity_novelty_reversibility") or {}
        if snr:
            lines.append("\\textbf{【重大性・新規性・可逆性】}")
            for key_ja, key_en, just_key in (
                ("重大性", "severity", "severity_justification"),
                ("新規性", "novelty", "novelty_justification"),
                ("可逆性", "reversibility", "reversibility_justification"),
            ):
                val = snr.get(key_en, "---")
                just = snr.get(just_key, "")
                lines.append(
                    f"  {key_ja}: {_escape_latex_text(str(val))}"
                    f" --- {_escape_latex_text(str(just))}"
                )
            lines.append("")

        mitigations = report.get("mitigations", [])
        if mitigations:
            lines += ["\\textbf{【緩和策】}", "\\begin{enumerate}"]
            for m in mitigations:
                action = m.get("action_ja", "（記述なし）")
                lines.append(f"  \\item {_escape_latex_text(action)}")
            lines.append("\\end{enumerate}")

    lines += ["\\hrule", ""]
    return "\n".join(lines)


def _build_latex(report_data: dict, tikz_str: str) -> str:
    """Fill _LATEX_TEMPLATE with content from the assembled report JSON."""
    session = report_data.get("session", {})
    decision_type = report_data.get("decision_type", {})
    chain_reports = report_data.get("chain_reports", [])
    generated_at = report_data.get(
        "generated_at", datetime.now(timezone.utc).isoformat()
    )

    dt_name_ja = _escape_latex_text(
        decision_type.get("name_ja") or decision_type.get("name_en", "N/A")
    )
    session_id = _escape_latex(str(session.get("session_id", "N/A"))[:16])
    chain_count = str(len(chain_reports))

    exec_summary_raw = session.get("executive_summary_ja")
    if exec_summary_raw:
        exec_summary = _escape_latex_text(exec_summary_raw)
    else:
        err = session.get("executive_summary_error", "原因不明")
        exec_summary = f"（生成エラー: {_escape_latex_text(err)}）"

    total = len(chain_reports)
    chain_sections = "\n".join(
        _chain_section_latex(r, i + 1, total) for i, r in enumerate(chain_reports)
    )
    if not chain_sections:
        chain_sections = "（詳細連鎖なし）"

    doc = _LATEX_TEMPLATE
    doc = doc.replace(r"\VAR{decision_type_name_ja}", dt_name_ja)
    doc = doc.replace(r"\VAR{generated_at}", _escape_latex_text(generated_at[:19]))
    doc = doc.replace(r"\VAR{session_id}", session_id)
    doc = doc.replace(r"\VAR{chain_count}", chain_count)
    doc = doc.replace(r"\VAR{executive_summary}", exec_summary)
    doc = doc.replace(r"\VAR{tikz_dag}", tikz_str)
    doc = doc.replace(r"\VAR{chain_sections}", chain_sections)
    return doc


def _render_adapted_chain_block(chain: dict, idx: int, total: int) -> str:
    """Render one adapted chain dict as a tcolorbox LaTeX block (new template)."""
    if chain.get("is_gap_placeholder"):
        prose = _escape_latex_text(chain.get("prose", ""))
        return (
            "\\begin{tcolorbox}[enhanced, colback=gray!8, colframe=gray!50,"
            " boxrule=0.6pt, arc=4pt, fonttitle=\\bfseries,"
            f" title={{[{idx}/{total}]\\ 因果証拠不十分}}]\n"
            f"{prose}\n"
            "\\end{tcolorbox}\n\\vspace{0.5em}\n"
        )

    tier = chain.get("confidence_tier", "")
    bgcolor, framecolor = _TIER_COLORS.get(tier, ("gray!8", "gray!50"))

    severity = chain.get("severity", "")
    sev_ja, sev_color = _SEVERITY_DISPLAY.get(severity, ("---", "tiernone"))

    tier_ja = {
        "well-supported": "文献支持",
        "plausible":      "蓋然性あり",
        "worth-considering": "考慮に値する",
    }.get(tier, tier or "---")

    title = _escape_latex_text(chain.get("title", ""))
    node_labels = chain.get("nodes") or []
    path_str = _escape_latex_text(" → ".join(node_labels)) if node_labels else ""
    chain_len = max(len(chain.get("path", [])) - 1, 0)

    lines: list[str] = [
        f"\\begin{{tcolorbox}}[enhanced, colback={bgcolor}, colframe={framecolor},",
        "  boxrule=0.6pt, arc=4pt, fonttitle=\\bfseries,",
        f"  title={{[{idx}/{total}]\\ {title}}}]",
        f"{{\\small \\textbf{{信頼度ティア:}} {_escape_latex_text(tier_ja)}"
        f" \\quad \\textbf{{重大性:}} "
        f"\\textcolor{{{sev_color}}}{{\\textbf{{{sev_ja}}}}}"
        f" \\quad \\textbf{{連鎖長:}} {chain_len}}}",
        "",
        "\\medskip",
        "\\textbf{因果経路:}",
        f"\\begin{{quote}}{path_str}\\end{{quote}}",
    ]

    citations = chain.get("citations", [])
    if citations:
        lines += [
            "\\textbf{根拠:}",
            "\\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]",
        ]
        for cit in citations:
            lines.append(f"  \\item {_escape_latex_text(cit)}")
        lines.append("\\end{itemize}")

    prose = chain.get("prose", "")
    if prose:
        lines += ["", "\\medskip", "\\textbf{分析:}", "", _escape_latex_text(prose)]

    mitigations = chain.get("mitigation", [])
    if mitigations:
        lines += [
            "",
            "\\medskip",
            "\\textbf{緩和策:}",
            "\\begin{enumerate}[leftmargin=*,topsep=2pt,itemsep=2pt]",
        ]
        for m in mitigations:
            action = m.get("action_ja", "（記述なし）")
            lines.append(f"  \\item {_escape_latex_text(action)}")
        lines.append("\\end{enumerate}")

    lines += ["\\end{tcolorbox}", "\\vspace{0.5em}", ""]
    return "\n".join(lines)


def _build_latex_adapted(adapted: dict, tikz_str: str) -> str:
    """Fill noman_report.tex with content from an adapted report dict."""
    if not _TEMPLATE_PATH.exists():
        raise RuntimeError(
            f"Template not found: {_TEMPLATE_PATH}.\n"
            "Ensure noman_report.tex is present in the project root."
        )
    doc = _TEMPLATE_PATH.read_text(encoding="utf-8")

    chains = adapted.get("chains", [])
    total = len(chains)
    chain_blocks = "\n".join(
        _render_adapted_chain_block(c, i + 1, total)
        for i, c in enumerate(chains)
    ) or "（詳細連鎖なし）"

    substitutions = {
        r"\VAR{decision_label}":   _escape_latex_text(adapted.get("decision_label", "")),
        r"\VAR{report_date}":      _escape_latex_text(adapted.get("report_date", "")),
        r"\VAR{chain_count}":      str(total),
        r"\VAR{graph_version}":    _escape_latex_text(adapted.get("graph_version", "")),
        r"\VAR{promotion_mode}":   _escape_latex_text(adapted.get("promotion_mode", "")),
        r"\VAR{executive_summary}": _escape_latex_text(adapted.get("executive_summary", "")),
        r"\VAR{tikz_dag}":         tikz_str,
        r"\VAR{chain_blocks}":     chain_blocks,
    }
    for placeholder, value in substitutions.items():
        doc = doc.replace(placeholder, value)
    return doc


def _compile_latex(latex_source: str, pdf_dst: Path) -> None:
    """Write LaTeX source to a temp dir, compile with xelatex, copy PDF to pdf_dst."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "report.tex"
        tex_file.write_text(latex_source, encoding="utf-8")

        last_result = None
        for _ in range(2):  # two passes to resolve any cross-references
            last_result = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "report.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

        pdf_src = Path(tmpdir) / "report.pdf"
        if not pdf_src.exists():
            log_tail = (last_result.stdout if last_result else "")[-3000:]
            raise RuntimeError(
                f"xelatex failed to produce a PDF.\n"
                f"LaTeX log (last 3000 chars):\n{log_tail}"
            )

        shutil.copy2(pdf_src, pdf_dst)
        print(f"  PDF  → {pdf_dst}")
