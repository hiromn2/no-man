# No-Man Streamlit Report App

This folder is a sandbox for a visually interactive Streamlit report viewer.
It intentionally does not call `run_d03_report.py` or generate a fresh report.
For now, the app reads dummy data from `dummy_data.py`.

## Architecture

- `app.py` is the Streamlit entry point.
- `dummy_data.py` defines a small report-shaped fixture for UI work.
- `components.py` contains reusable presentation functions.
- `report_loader.py` loads uploaded existing report JSON without generating a report.
- `i18n.py` contains lightweight English/Japanese UI labels.
- `styles.py` centralizes small CSS helpers.

## Intended Data Boundary

The UI should consume an already-assembled report object. Report generation,
graph traversal, LLM calls, session creation, and PDF rendering should remain
outside this folder until the integration point is explicit.

The current prototype supports:

- dummy fixture viewing;
- upload of an existing JSON report;
- a dependency-free SVG network view with hidden node names that appear on hover
  or click;
- a small English/Japanese UI toggle for app chrome.

Expected future adapter shape:

```python
{
    "decision_label": str,
    "report_date": str,
    "executive_summary": str,
    "chains": [
        {
            "chain_id": str,
            "title": str,
            "confidence_tier": str,
            "severity": str,
            "path": list[str],
            "nodes": list[str],
            "signs": list[str],
            "prose": str,
            "citations": list[str],
            "mitigation": list[str],
            "diagnostics": list[dict],
        }
    ],
}
```

## Run Locally

Install Streamlit in your environment, then run:

```bash
streamlit run streamlit_report_app/app.py
```
