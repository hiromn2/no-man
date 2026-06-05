"""Small UI translation table for the Streamlit prototype."""

from __future__ import annotations


LANGUAGES = {
    "日本語": "ja",
    "English": "en",
}

_TEXT = {
    "app_title": {
        "en": "No-Man Interactive Report",
        "ja": "No-Man インタラクティブ報告書",
    },
    "source": {"en": "Report source", "ja": "報告書ソース"},
    "all": {"en": "All", "ja": "すべて"},
    "dummy": {"en": "Sample data", "ja": "サンプルデータ"},
    "upload": {"en": "Upload existing JSON", "ja": "既存JSONをアップロード"},
    "upload_prompt": {
        "en": "Choose an existing report JSON",
        "ja": "既存の報告書JSONを選択",
    },
    "upload_help": {
        "en": "Use an already-generated report. This app does not run report generation.",
        "ja": "既に生成済みの報告書を使用します。このアプリは報告書生成を実行しません。",
    },
    "loaded_dummy": {
        "en": "Showing dummy fixture. No real report was generated.",
        "ja": "ダミーデータを表示中です。実際の報告書生成は実行していません。",
    },
    "loaded_upload": {
        "en": "Loaded uploaded report JSON.",
        "ja": "アップロードされた報告書JSONを読み込みました。",
    },
    "chains": {"en": "Causal Chains", "ja": "因果連鎖"},
    "confidence": {"en": "Confidence", "ja": "信頼度"},
    "severity": {"en": "Severity", "ja": "重大性"},
    "lowest_confidence": {"en": "Lowest tier", "ja": "最低信頼度"},
    "high_severity": {"en": "High severity", "ja": "重大性High"},
    "executive_summary": {"en": "Executive Summary", "ja": "経営陣向けサマリー"},
    "network": {"en": "Network View", "ja": "ネットワーク図"},
    "chain_id": {"en": "Chain ID", "ja": "連鎖ID"},
    "evidence": {"en": "Evidence", "ja": "証拠"},
    "mitigations": {"en": "Mitigations", "ja": "緩和策"},
    "diagnostics": {"en": "Diagnostics", "ja": "診断"},
    "details": {"en": "Details", "ja": "詳細"},
    "no_match": {
        "en": "No chains match the current filters.",
        "ja": "現在のフィルターに一致する連鎖はありません。",
    },
    "no_citations": {"en": "No citations attached.", "ja": "引用はありません。"},
    "no_mitigation": {
        "en": "No mitigation notes attached.",
        "ja": "緩和策メモはありません。",
    },
    "no_diagnostics": {
        "en": "No diagnostic recommendations attached.",
        "ja": "診断的推奨事項はありません。",
    },
    "network_caption": {
        "en": "Hover or click a node to reveal its name. Labels stay hidden by default.",
        "ja": "ノードにマウスを置くかクリックすると名前が表示されます。通常時は非表示です。",
    },
    "show_node_labels": {
        "en": "Show node labels for export",
        "ja": "書き出し用にノード名を常時表示",
    },
    "selected_node": {"en": "Selected node", "ja": "選択中のノード"},
}


def t(key: str, lang: str) -> str:
    return _TEXT.get(key, {}).get(lang, key)
