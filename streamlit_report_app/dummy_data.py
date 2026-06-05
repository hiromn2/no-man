"""Dummy report data for Streamlit UI development.

This fixture is intentionally local to the Streamlit sandbox. It avoids running
the real No-Man report pipeline while preserving the shape the UI will need.
"""

from __future__ import annotations


DUMMY_REPORT: dict = {
    "decision_label": "支店閉鎖またはネットワーク縮小",
    "report_date": "2026-06-05",
    "graph_version": "ui-preview",
    "promotion_mode": "not_connected",
    "executive_summary": (
        "支店ネットワークの縮小は、地域中小企業の信用アクセスやメインバンク関係の"
        "信頼に影響する可能性があります。特に対面接点に依存する地域では、信用情報"
        "の収集機会が減少し、代替チャネルへの移行負担が顕在化する可能性があります。"
    ),
    "chains": [
        {
            "chain_id": "dummy-chain-001",
            "title": "支店ネットワーク -> 信用アクセス",
            "chain_type": "adverse",
            "confidence_tier": "plausible",
            "severity": "high",
            "path": ["bank_branch_network", "regional_sme_credit_access"],
            "nodes": ["銀行支店ネットワーク", "地域中小企業の信用アクセス"],
            "signs": ["+"],
            "prose": (
                "支店ネットワークが縮小すると、対面接点とソフト情報の収集機会が減り、"
                "地域中小企業の信用アクセスが弱まる可能性があります。"
            ),
            "citations": ["Uchida et al. 2008", "Ogura 2010"],
            "mitigation": [
                "閉鎖地域に巡回担当者を配置する",
                "融資相談のデジタル窓口を整備する",
            ],
            "diagnostics": [
                {
                    "variable_ja": "中小企業向け新規融資件数",
                    "data_source": "銀行内部管理データ",
                    "test_suggested": "支店閉鎖前後の差分分析",
                    "interpretation": "閉鎖後に低下すれば、信用アクセス低下の兆候として扱う。",
                }
            ],
        },
        {
            "chain_id": "dummy-chain-002",
            "title": "支店ネットワーク -> 関係信頼 -> 預金基盤",
            "chain_type": "adverse",
            "confidence_tier": "worth-considering",
            "severity": "medium",
            "path": [
                "bank_branch_network",
                "main_bank_relationship_trust",
                "local_deposit_base",
            ],
            "nodes": ["銀行支店ネットワーク", "メインバンク関係の信頼", "地域預金基盤"],
            "signs": ["+", "+"],
            "prose": (
                "物理的接点の減少は、地域との関係性が弱まったというシグナルとして"
                "受け止められ、預金基盤に影響する可能性があります。"
            ),
            "citations": ["Hoshi & Kashyap 2001"],
            "mitigation": [
                "閉鎖前後に主要取引先との個別面談を実施する",
                "地域拠点の代替接点を明示する",
            ],
            "diagnostics": [
                {
                    "variable_ja": "閉鎖地域の預金残高",
                    "data_source": "店舗別預金残高",
                    "test_suggested": "閉鎖店舗周辺と非閉鎖地域の比較",
                    "interpretation": "閉鎖地域だけで低下すれば関係信頼の変化を疑う。",
                }
            ],
        },
    ],
}
