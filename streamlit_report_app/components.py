"""Reusable Streamlit components for the interactive report viewer."""

from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

from i18n import t
from styles import tier_label


def render_header(report: dict, lang: str) -> None:
    st.title(t("app_title", lang))
    st.caption(
        f"{report.get('decision_label', 'Untitled decision')} | "
        f"{report.get('report_date', 'No date')}"
    )


def render_overview(report: dict, lang: str) -> None:
    chains = report.get("chains", [])
    severities = [chain.get("severity", "unknown") for chain in chains]

    col_count, col_high, col_tier = st.columns(3)
    col_count.metric(t("chains", lang), len(chains))
    col_high.metric(t("high_severity", lang), severities.count("high"))
    col_tier.metric(
        t("lowest_confidence", lang),
        _lowest_confidence_label(chains, lang),
    )

    st.subheader(t("executive_summary", lang))
    st.write(report.get("executive_summary", "No executive summary available."))


def render_filters(chains: list[dict], lang: str) -> tuple[str, str]:
    all_label = t("all", lang)
    confidence_values = [all_label] + sorted(
        {tier_label(chain.get("confidence_tier", ""), lang) for chain in chains}
    )
    severity_values = [all_label] + sorted(
        {chain.get("severity", "unknown").title() for chain in chains}
    )

    col_confidence, col_severity = st.columns(2)
    confidence_filter = col_confidence.selectbox(t("confidence", lang), confidence_values)
    severity_filter = col_severity.selectbox(t("severity", lang), severity_values)
    return confidence_filter, severity_filter


def filter_chains(
    chains: list[dict],
    confidence_filter: str,
    severity_filter: str,
    lang: str,
) -> list[dict]:
    result = []
    all_label = t("all", lang)
    for chain in chains:
        confidence = tier_label(chain.get("confidence_tier", ""), lang)
        severity = chain.get("severity", "unknown").title()
        if confidence_filter != all_label and confidence != confidence_filter:
            continue
        if severity_filter != all_label and severity != severity_filter:
            continue
        result.append(chain)
    return result


def render_network(chains: list[dict], lang: str, show_labels: bool = False) -> None:
    st.subheader(t("network", lang))
    if not show_labels:
        st.caption(t("network_caption", lang))
    graph = _build_network_model(chains)
    components.html(_network_html(graph, lang, show_labels), height=335, scrolling=False)


def render_chain_list(chains: list[dict], lang: str) -> None:
    if not chains:
        st.info(t("no_match", lang))
        return

    for chain in chains:
        title = chain.get("title", chain.get("chain_id", "Untitled chain"))
        tier = chain.get("confidence_tier", "")
        st.markdown(
            _chain_summary_html(title, tier, lang),
            unsafe_allow_html=True,
        )
        with st.expander(t("details", lang)):
            col_meta, col_text = st.columns([1, 2])
            with col_meta:
                st.caption(t("confidence", lang))
                st.write(tier_label(tier, lang))
                st.caption(t("severity", lang))
                st.write(chain.get("severity", "unknown").title())
                st.caption(t("chain_id", lang))
                st.code(chain.get("chain_id", "unknown"), language=None)
            with col_text:
                render_chain_path(chain)
                st.write(chain.get("prose", ""))

            render_chain_details(chain, lang)


def render_chain_path(chain: dict) -> None:
    nodes = chain.get("nodes", [])
    signs = chain.get("signs", [])
    parts: list[str] = []
    for index, node in enumerate(nodes):
        parts.append(f'<span class="node-pill">{html.escape(node)}</span>')
        if index < len(signs):
            sign = html.escape(signs[index])
            parts.append(f'<span class="edge-sign">-[{sign}]-></span>')

    st.markdown(
        f'<div class="chain-path">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_chain_details(chain: dict, lang: str) -> None:
    tab_evidence, tab_mitigation, tab_diagnostics = st.tabs(
        [t("evidence", lang), t("mitigations", lang), t("diagnostics", lang)]
    )

    with tab_evidence:
        citations = chain.get("citations", [])
        if citations:
            for citation in citations:
                st.write(f"- {citation}")
        else:
            st.write(t("no_citations", lang))

    with tab_mitigation:
        mitigation = chain.get("mitigation", [])
        if mitigation:
            for item in mitigation:
                st.write(f"- {item}")
        else:
            st.write(t("no_mitigation", lang))

    with tab_diagnostics:
        diagnostics = chain.get("diagnostics", [])
        if diagnostics:
            st.dataframe(diagnostics, use_container_width=True, hide_index=True)
        else:
            st.write(t("no_diagnostics", lang))


def _lowest_confidence_label(chains: list[dict], lang: str) -> str:
    rank = {"well-supported": 3, "plausible": 2, "worth-considering": 1}
    if not chains:
        return "None"
    lowest = min(chains, key=lambda chain: rank.get(chain.get("confidence_tier", ""), 0))
    return tier_label(lowest.get("confidence_tier", ""), lang)


def _chain_summary_html(title: str, tier: str, lang: str) -> str:
    safe_title = html.escape(title)
    safe_tier = html.escape(tier_label(tier, lang))
    tier_class = _tier_class(tier)
    return (
        '<div class="chain-summary">'
        f'<span class="chain-summary-title">{safe_title}</span>'
        f'<span class="tier-badge {tier_class}">{safe_tier}</span>'
        "</div>"
    )


def _tier_class(tier: str) -> str:
    if tier in {"well-supported", "plausible", "worth-considering"}:
        return f"tier-{tier}"
    return "tier-worth-considering"


def _build_network_model(chains: list[dict]) -> dict:
    node_index: dict[str, dict] = {}
    links: set[tuple[str, str, str]] = set()

    for chain in chains:
        node_ids = chain.get("path", [])
        node_names = chain.get("nodes", node_ids)
        signs = chain.get("signs", [])
        for index, node_id in enumerate(node_ids):
            label = node_names[index] if index < len(node_names) else node_id
            node_index.setdefault(
                node_id,
                {
                    "id": node_id,
                    "label": label,
                    "terminal": False,
                    "chains": [],
                },
            )
            node_index[node_id]["chains"].append(chain.get("chain_id", "unknown"))

        if node_ids:
            node_index[node_ids[-1]]["terminal"] = True

        for index in range(max(0, len(node_ids) - 1)):
            sign = signs[index] if index < len(signs) else "+"
            links.add((node_ids[index], node_ids[index + 1], sign))

    nodes = list(node_index.values())
    for node in nodes:
        node["chains"] = sorted(set(node["chains"]))

    return {
        "nodes": nodes,
        "links": [
            {"source": source, "target": target, "sign": sign}
            for source, target, sign in sorted(links)
        ],
    }


def _network_html(graph: dict, lang: str, show_labels: bool) -> str:
    graph_json = json.dumps(graph, ensure_ascii=False)
    selected_node = t("selected_node", lang)
    label_class = "show-labels" if show_labels else ""
    return f"""
    <div id="network-root" class="{label_class}">
      <svg id="network-svg" viewBox="0 0 980 250" role="img"></svg>
      <div id="node-detail">{html.escape(selected_node)}: <span>-</span></div>
    </div>
    <style>
      #network-root {{
        height: 315px;
        border: 1px solid #d8ddd8;
        border-radius: 8px;
        background: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      #network-svg {{
        width: 100%;
        height: 255px;
      }}
      .edge {{
        stroke: #9aa39c;
        stroke-width: 2;
        marker-end: url(#arrow);
      }}
      .edge-label {{
        fill: #5e6963;
        font-size: 13px;
      }}
      .node {{
        fill: #ffffff;
        stroke: #587465;
        stroke-width: 2;
        cursor: pointer;
        transition: fill 120ms ease, stroke-width 120ms ease;
      }}
      .node.terminal {{
        stroke: #a65f3b;
        stroke-width: 3;
      }}
      .node:hover,
      .node.selected {{
        fill: #e8f0ea;
        stroke-width: 4;
      }}
      .node-label {{
        opacity: 0;
        pointer-events: none;
        fill: #1f2924;
        font-size: 14px;
        font-weight: 650;
      }}
      .node-group:hover .node-label,
      .node-group.selected .node-label,
      #network-root.show-labels .node-label {{
        opacity: 1;
      }}
      #node-detail {{
        border-top: 1px solid #d8ddd8;
        padding: 0.75rem 0.9rem;
        color: #2d352f;
        font-size: 14px;
      }}
      #node-detail span {{
        font-weight: 650;
      }}
    </style>
    <script>
      const graph = {graph_json};
      const svg = document.getElementById("network-svg");
      const detail = document.querySelector("#node-detail span");
      const width = 980;
      const height = 250;
      const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));

      svg.innerHTML = `
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4"
            orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L10,4 L0,8 Z" fill="#9aa39c"></path>
          </marker>
        </defs>
      `;

      const columns = new Map();
      graph.links.forEach((link) => {{
        columns.set(link.source, Math.min(columns.get(link.source) ?? 0, 0));
        columns.set(link.target, Math.max((columns.get(link.source) ?? 0) + 1, columns.get(link.target) ?? 0));
      }});
      graph.nodes.forEach((node) => {{
        if (!columns.has(node.id)) columns.set(node.id, 0);
      }});

      const grouped = new Map();
      graph.nodes.forEach((node) => {{
        const col = columns.get(node.id);
        if (!grouped.has(col)) grouped.set(col, []);
        grouped.get(col).push(node);
      }});

      const maxCol = Math.max(...columns.values(), 1);
      const positions = new Map();
      [...grouped.entries()].forEach(([col, nodes]) => {{
        nodes.sort((a, b) => a.label.localeCompare(b.label));
        nodes.forEach((node, idx) => {{
          const x = 90 + (col * (width - 180)) / maxCol;
          const y = 62 + ((idx + 1) * (height - 112)) / (nodes.length + 1);
          positions.set(node.id, {{x, y}});
        }});
      }});

      graph.links.forEach((link) => {{
        const source = positions.get(link.source);
        const target = positions.get(link.target);
        if (!source || !target) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", "edge");
        line.setAttribute("x1", source.x + 16);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x - 18);
        line.setAttribute("y2", target.y);
        svg.appendChild(line);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "edge-label");
        label.setAttribute("x", (source.x + target.x) / 2);
        label.setAttribute("y", (source.y + target.y) / 2 - 8);
        label.setAttribute("text-anchor", "middle");
        label.textContent = link.sign;
        svg.appendChild(label);
      }});

      graph.nodes.forEach((node) => {{
        const pos = positions.get(node.id);
        if (!pos) return;
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", "node-group");

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", node.terminal ? "node terminal" : "node");
        circle.setAttribute("cx", pos.x);
        circle.setAttribute("cy", pos.y);
        circle.setAttribute("r", node.terminal ? 17 : 15);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "node-label");
        label.setAttribute("x", pos.x);
        label.setAttribute("y", pos.y - 25);
        label.setAttribute("text-anchor", "middle");
        label.textContent = node.label;

        group.appendChild(circle);
        group.appendChild(label);
        group.addEventListener("mouseenter", () => {{
          detail.textContent = node.label;
        }});
        group.addEventListener("click", () => {{
          document.querySelectorAll(".node-group").forEach((el) => el.classList.remove("selected"));
          group.classList.add("selected");
          circle.classList.add("selected");
          detail.textContent = `${{node.label}} (${{node.id}})`;
        }});
        svg.appendChild(group);
      }});
    </script>
    """
