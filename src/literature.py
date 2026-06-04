"""
literature.py — Citation verification and Semantic Scholar search for No-Man.

Public API
----------
search_edge_support(from_node, to_node, sign) -> list[PaperResult]
    Build a targeted query from node vocabulary and return candidate papers
    that may support the causal relationship.

verify_citation(citation_string) -> CitationResult
    Search Semantic Scholar for a citation string (e.g. "Hosono 2006") and
    return the best match with governance acceptance classification.

lookup_paper(doi_or_title) -> PaperResult
    Resolve a DOI or title to a full PaperResult.

Types
-----
PaperResult (TypedDict):
    title, year, authors, venue, doi, abstract,
    semantic_scholar_id, is_accepted_source_type, source_type,
    publication_types, citation_count

CitationResult (TypedDict):
    found, citation_string, paper, candidates, is_accepted_source_type

Design constraints
------------------
- All HTTP calls go through the single _api_get() function (rate limiting +
  caching). No other function may call requests directly.
- Disk cache at .cache/literature/ with 30-day TTL.
- SEMANTIC_SCHOLAR_API_KEY loaded from environment; unauthenticated if absent.
- Results are CANDIDATES only — governance.json §speculative_to_tested still
  applies before any edge reaches seed_graph.json.

CLI
---
python -m src.literature search <from_node> <to_node> <sign>
python -m src.literature verify <citation_string>
python -m src.literature lookup <doi_or_title>
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TypedDict

import requests

# ---------------------------------------------------------------------------
# Optional .env loading (dev convenience; silent if dotenv not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_VOCAB_PATH = _REPO_ROOT / "knowledge" / "search_vocabulary.json"

CACHE_DIR: Path = _REPO_ROOT / ".cache" / "literature"
CACHE_TTL_SECONDS: int = 30 * 24 * 3600  # 30 days

_SS_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SS_SEARCH_URL = f"{_SS_BASE_URL}/paper/search"
_SS_PAPER_URL = f"{_SS_BASE_URL}/paper"

_SS_FIELDS = (
    "title,year,authors,venue,externalIds,"
    "abstract,publicationTypes,citationCount"
)

# Semantic Scholar rate limits
# Unauthenticated : ~100 req / 5 min  → ≥ 3 s per req to stay safe
# Authenticated   : ~1000 req / 5 min → ≥ 0.35 s per req
_RATE_LIMIT_UNAUTHENTICATED: float = 3.0
_RATE_LIMIT_AUTHENTICATED: float = 0.35

# Accepted source types as defined in knowledge/governance.json
ACCEPTED_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "peer_reviewed_journal_article",
        "working_paper_boj",
        "working_paper_rieti",
        "working_paper_imf",
        "working_paper_bis",
        "working_paper_recognized_academic_institution",
    }
)

# Venue / title patterns for source-type classification (all lowercase)
_BOJ_PATTERNS = [
    "bank of japan",
    "boj working paper",
    "boj research paper",
    "institute for monetary and economic studies",
    "imes discussion paper",
    "日本銀行",
]
_RIETI_PATTERNS = [
    "rieti",
    "research institute of economy, trade and industry",
    "rieti discussion paper",
]
_IMF_PATTERNS = [
    "imf working paper",
    "international monetary fund",
    "imf staff discussion note",
    "imf economic review",
]
_BIS_PATTERNS = [
    "bis working paper",
    "bank for international settlements",
    "bis quarterly review",
    "bis research paper",
]
_ACADEMIC_WP_PATTERNS = [
    "nber working paper",
    "national bureau of economic research",
    "cepr discussion paper",
    "federal reserve bank",
    "european central bank",
    "ecb working paper",
    "world bank policy research",
    "discussion paper series",
    "working paper series",
    "cesifo working paper",
    "iza discussion paper",
    "cambridge working paper",
    "oxford working paper",
]


# ---------------------------------------------------------------------------
# TypedDict return types
# ---------------------------------------------------------------------------


class PaperResult(TypedDict):
    """Single paper result from Semantic Scholar, with governance classification."""

    title: str
    year: int | None
    authors: list[str]
    venue: str | None
    doi: str | None
    abstract: str | None
    semantic_scholar_id: str | None
    is_accepted_source_type: bool
    source_type: str
    publication_types: list[str]
    citation_count: int | None


class CitationResult(TypedDict):
    """Result of verify_citation(): best match + candidates + acceptance flag."""

    found: bool
    citation_string: str
    paper: PaperResult | None
    candidates: list[PaperResult]
    is_accepted_source_type: bool


# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------

_last_request_time: float = 0.0
_vocabulary_cache: dict[str, dict] | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_vocabulary() -> dict[str, dict]:
    """Load and memoize knowledge/search_vocabulary.json, keyed by node_id."""
    global _vocabulary_cache
    if _vocabulary_cache is None:
        with _VOCAB_PATH.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        _vocabulary_cache = {
            entry["node_id"]: entry for entry in raw.get("vocabulary", [])
        }
    return _vocabulary_cache


def _classify_source_type(
    title: str, venue: str | None, pub_types: list[str]
) -> str:
    """
    Map Semantic Scholar metadata to a governance.json accepted_source_types value.

    Priority order: institution-specific WPs (BoJ > RIETI > IMF > BIS) →
    peer-reviewed journal → other recognised academic WP → unknown.
    """
    combined = f"{(venue or '').lower()} {(title or '').lower()}"

    if any(p in combined for p in _BOJ_PATTERNS):
        return "working_paper_boj"
    if any(p in combined for p in _RIETI_PATTERNS):
        return "working_paper_rieti"
    if any(p in combined for p in _IMF_PATTERNS):
        return "working_paper_imf"
    if any(p in combined for p in _BIS_PATTERNS):
        return "working_paper_bis"
    if "JournalArticle" in pub_types:
        return "peer_reviewed_journal_article"
    if any(p in combined for p in _ACADEMIC_WP_PATTERNS):
        return "working_paper_recognized_academic_institution"
    if "WorkingPaper" in pub_types:
        return "working_paper_recognized_academic_institution"
    return "unknown"


def _paper_from_ss(data: dict) -> PaperResult:
    """Convert a raw Semantic Scholar API paper dict into a PaperResult."""
    title: str = data.get("title") or ""
    year: int | None = data.get("year")
    authors: list[str] = [a.get("name", "") for a in (data.get("authors") or [])]
    venue: str | None = data.get("venue") or None
    ext_ids: dict = data.get("externalIds") or {}
    doi: str | None = ext_ids.get("DOI") or None
    abstract: str | None = data.get("abstract") or None
    ss_id: str | None = data.get("paperId") or None
    pub_types: list[str] = data.get("publicationTypes") or []
    citation_count: int | None = data.get("citationCount")

    source_type = _classify_source_type(title, venue, pub_types)

    return {  # type: ignore[return-value]
        "title": title,
        "year": year,
        "authors": authors,
        "venue": venue,
        "doi": doi,
        "abstract": abstract,
        "semantic_scholar_id": ss_id,
        "is_accepted_source_type": source_type in ACCEPTED_SOURCE_TYPES,
        "source_type": source_type,
        "publication_types": pub_types,
        "citation_count": citation_count,
    }


def _empty_paper(label: str) -> PaperResult:
    """Return an empty PaperResult for failed lookups."""
    return {  # type: ignore[return-value]
        "title": label,
        "year": None,
        "authors": [],
        "venue": None,
        "doi": None,
        "abstract": None,
        "semantic_scholar_id": None,
        "is_accepted_source_type": False,
        "source_type": "unknown",
        "publication_types": [],
        "citation_count": None,
    }


def _parse_citation_hint(citation_string: str) -> tuple[str, int | None]:
    """
    Extract a search-friendly query and optional year from a citation string.

    Examples
    --------
    "Hosono 2006"            → ("Hosono", 2006)
    "Peek & Rosengren 2000"  → ("Peek Rosengren", 2000)
    "Uchida et al. 2008"     → ("Uchida", 2008)
    "BoJ regional reports"   → ("BoJ regional reports", None)
    """
    year_match = re.search(r"\b(19|20)\d{2}\b", citation_string)
    year: int | None = int(year_match.group()) if year_match else None

    query = citation_string
    if year_match:
        query = query[: year_match.start()] + query[year_match.end() :]
    query = re.sub(r"\bet\s+al\.?\b", "", query, flags=re.IGNORECASE)
    query = re.sub(r"[&,.()\[\]]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query, year


def _api_get(url: str, params: dict) -> dict:
    """
    Single rate-limited, disk-cached HTTP GET against the Semantic Scholar API.

    - Cache: .cache/literature/<sha256>.json, TTL = 30 days.
    - Rate limit: 3.0 s/req unauthenticated, 0.35 s/req with API key.
    - Auth: x-api-key header when SEMANTIC_SCHOLAR_API_KEY is set.
    - Retry: on HTTP 429 waits 5 s then retries once.
    - Raises: requests.HTTPError on unrecoverable errors.

    This is the ONLY function in this module that may call requests.get().
    """
    global _last_request_time

    # ── cache lookup ────────────────────────────────────────────────────────
    cache_key = hashlib.sha256(
        f"{url}|{json.dumps(params, sort_keys=True)}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            with cache_path.open(encoding="utf-8") as fh:
                return json.load(fh)

    # ── rate limit ──────────────────────────────────────────────────────────
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    min_interval = _RATE_LIMIT_AUTHENTICATED if api_key else _RATE_LIMIT_UNAUTHENTICATED
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    headers = {"x-api-key": api_key} if api_key else {}

    # ── HTTP request with exponential backoff on 429 ─────────────────────────
    backoff = 5.0
    data: dict = {}
    try:
        for attempt in range(3):
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                if attempt < 2:
                    print(
                        f"[literature] Rate-limited by Semantic Scholar — "
                        f"waiting {backoff:.0f} s (attempt {attempt + 1}/3) …",
                        file=sys.stderr,
                    )
                    time.sleep(backoff)
                    backoff *= 4  # 5 s → 20 s → 80 s
                    continue
                # Third failure: let raise_for_status() propagate
            resp.raise_for_status()
            data = resp.json()
            break
    finally:
        _last_request_time = time.time()

    # ── persist to cache ─────────────────────────────────────────────────────
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    return data


def _load_nodes(nodes_path: str | Path) -> dict[str, dict]:
    """Load nodes.json and return a dict keyed by node id."""
    path = Path(nodes_path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("nodes", data)
    if isinstance(raw, list):
        return {n["id"]: n for n in raw}
    if isinstance(raw, dict):
        return raw
    raise ValueError(
        f"nodes file {path} must contain a list under key 'nodes' or a dict of node objects"
    )


def _build_search_query(
    from_node: str, to_node: str, sign: str, *, fallback: bool = False
) -> str:
    """
    Construct a targeted Semantic Scholar search query for a causal edge.

    Loads academic vocabulary from knowledge/search_vocabulary.json to map
    node IDs to real academic terminology.  For negative-sign edges, appends
    "adverse" to surface risk-transmission and signaling literature.

    Parameters
    ----------
    from_node : Node ID of the cause node.
    to_node   : Node ID of the effect node.
    sign      : "+", "-", or "ambiguous".
    fallback  : If True, use terms_en[1] instead of terms_en[0] for both
                nodes.  Called by search_edge_support when the primary
                query returns fewer than 3 results.
    """
    vocab = _load_vocabulary()

    from_terms: list[str] = vocab.get(from_node, {}).get(
        "terms_en", [from_node.replace("_", " ")]
    )
    to_terms: list[str] = vocab.get(to_node, {}).get(
        "terms_en", [to_node.replace("_", " ")]
    )

    idx = 1 if fallback else 0
    from_q = (
        from_terms[idx] if len(from_terms) > idx
        else (from_terms[0] if from_terms else from_node.replace("_", " "))
    )
    to_q = (
        to_terms[idx] if len(to_terms) > idx
        else (to_terms[0] if to_terms else to_node.replace("_", " "))
    )

    sign_context = "adverse" if sign == "-" else ""

    parts = [from_q, to_q, "Japan", "regional bank", sign_context]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

_USE_COLORS: bool = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    """Apply ANSI colour code if the terminal supports it."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLORS else text


def _acceptance_badge(paper: PaperResult) -> str:
    if paper["is_accepted_source_type"]:
        return _c("✓ ACCEPTED SOURCE", "32")  # green
    return _c("✗ NOT ACCEPTED  ", "33")  # yellow


def _print_paper(paper: PaperResult, rank: int | None = None) -> None:
    prefix = f"  {rank:2d}." if rank is not None else "    "
    print(f"\n{prefix} {_acceptance_badge(paper)}")
    print(f"       Title   : {paper['title'] or '(no title)'}")
    print(f"       Year    : {paper['year'] if paper['year'] is not None else '—'}")
    print(f"       Venue   : {paper['venue'] or '—'}")

    authors = paper["authors"]
    if authors:
        short = ", ".join(authors[:3])
        if len(authors) > 3:
            short += f" + {len(authors) - 3} more"
        print(f"       Authors : {short}")

    print(f"       Src type: {paper['source_type']}")

    if paper["doi"]:
        print(f"       DOI     : {paper['doi']}")
    if paper["semantic_scholar_id"]:
        print(f"       SS ID   : {paper['semantic_scholar_id']}")
    if paper["citation_count"] is not None:
        print(f"       Cited   : {paper['citation_count']}")


def _print_search_results(
    results: list[PaperResult],
    from_node: str,
    to_node: str,
    sign: str,
) -> None:
    query = _build_search_query(from_node, to_node, sign)
    arrow = f"─({sign})─▶"

    print(f"\n{'═' * 72}")
    print(f"  No-Man Literature Search  │  Semantic Scholar")
    print(f"{'─' * 72}")
    print(f"  Edge   : {from_node}  {arrow}  {to_node}")
    print(f"  Query  : {query!r}")
    print(f"{'═' * 72}")

    if not results:
        print("\n  No results returned from Semantic Scholar.\n")
        return

    accepted_n = sum(1 for p in results if p["is_accepted_source_type"])
    print(f"\n  {len(results)} result(s) — {accepted_n} accepted source type(s)\n")

    for i, paper in enumerate(results, 1):
        _print_paper(paper, rank=i)

    print(f"\n{'─' * 72}")
    print(f"  {_c('GOVERNANCE NOTE', '1')}: These are search CANDIDATES only.")
    print(f"  Promotion speculative → tested requires human review +")
    print(f"  academic citation + second citation or quantitative test.")
    print(f"  See knowledge/governance.json §speculative_to_tested")
    print(f"{'─' * 72}\n")


def _print_citation_result(result: CitationResult) -> None:
    print(f"\n{'═' * 72}")
    print(f"  No-Man Citation Verification  │  Semantic Scholar")
    print(f"{'─' * 72}")
    print(f"  Input: {result['citation_string']!r}")
    print(f"{'═' * 72}")

    if result["found"] and result["paper"]:
        print(f"\n  Best match:")
        _print_paper(result["paper"])
        print(f"\n  is_accepted_source_type: {result['is_accepted_source_type']}")
        if len(result["candidates"]) > 1:
            print(f"\n  Other candidates ({len(result['candidates']) - 1}):")
            for p in result["candidates"][1:]:
                _print_paper(p)
    else:
        print("\n  No match found on Semantic Scholar.")
        print(
            "  Institutional references (e.g. 'BoJ regional reports')"
            " are not indexed."
        )

    print(f"\n{'─' * 72}\n")


def _print_usage() -> None:
    print(
        "\nNo-Man Literature Search\n"
        "\n"
        "Usage:\n"
        "  python -m src.literature search <from_node> <to_node> <sign>\n"
        "  python -m src.literature verify <citation_string>\n"
        "  python -m src.literature lookup <doi_or_title>\n"
        "\n"
        "Examples:\n"
        "  python -m src.literature search bank_capital_ratio "
        "bank_reputation_regional -\n"
        '  python -m src.literature verify "Hosono 2006"\n'
        '  python -m src.literature lookup "10.1016/j.jfi.2012.04.001"\n'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_edge_support(
    from_node: str,
    to_node: str,
    sign: str,
) -> list[PaperResult]:
    """
    Search Semantic Scholar for papers that may support the causal edge
    from_node → to_node with the predicted sign.

    Parameters
    ----------
    from_node : str
        Node ID of the cause node (should be in knowledge/nodes.json).
    to_node : str
        Node ID of the effect node (should be in knowledge/nodes.json).
    sign : str
        Predicted sign of the causal effect: "+", "-", or "ambiguous".

    Returns
    -------
    list[PaperResult]
        Up to 10 papers sorted by Semantic Scholar relevance.
        Each result carries ``is_accepted_source_type`` and ``source_type``
        per the rules in knowledge/governance.json.
        Returns an empty list on API failure (error printed to stderr).

    Notes
    -----
    Results are CANDIDATES only.  Promotion to seed_graph.json requires human
    review satisfying governance.json §speculative_to_tested.
    """
    query = _build_search_query(from_node, to_node, sign)

    try:
        data = _api_get(
            _SS_SEARCH_URL,
            {"query": query, "fields": _SS_FIELDS, "limit": 10},
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[literature] API error for edge {from_node}→{to_node}: {exc}",
            file=sys.stderr,
        )
        return []

    results = [_paper_from_ss(p) for p in (data.get("data") or [])]

    # Fallback: if the primary query is sparse, retry once with terms_en[1]
    if len(results) < 3:
        fb_query = _build_search_query(from_node, to_node, sign, fallback=True)
        if fb_query != query:
            print(
                f"[literature] Query returned {len(results)} results — "
                f"retrying with fallback terms",
                file=sys.stderr,
            )
            try:
                fb_data = _api_get(
                    _SS_SEARCH_URL,
                    {"query": fb_query, "fields": _SS_FIELDS, "limit": 10},
                )
                results = [_paper_from_ss(p) for p in (fb_data.get("data") or [])]
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[literature] Fallback API error for edge "
                    f"{from_node}→{to_node}: {exc}",
                    file=sys.stderr,
                )

    return results


def verify_citation(citation_string: str) -> CitationResult:
    """
    Attempt to verify a citation string against Semantic Scholar.

    Parameters
    ----------
    citation_string : str
        Citation in any informal format, e.g. "Hosono 2006",
        "Peek & Rosengren 2000", "Uchida et al. 2008".
        Institutional references like "BoJ regional reports" will
        return ``found=False``.

    Returns
    -------
    CitationResult
        ``paper`` is the best match (year-filtered when a year is present),
        ``candidates`` contains up to 5 results, and
        ``is_accepted_source_type`` reflects the best match's classification.
    """
    query, year_hint = _parse_citation_hint(citation_string)

    try:
        data = _api_get(
            _SS_SEARCH_URL,
            {"query": query, "fields": _SS_FIELDS, "limit": 5},
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[literature] API error verifying {citation_string!r}: {exc}",
            file=sys.stderr,
        )
        return {  # type: ignore[return-value]
            "found": False,
            "citation_string": citation_string,
            "paper": None,
            "candidates": [],
            "is_accepted_source_type": False,
        }

    candidates = [_paper_from_ss(p) for p in (data.get("data") or [])]

    # Year-match preference
    if year_hint and candidates:
        year_matches = [p for p in candidates if p["year"] == year_hint]
        best: PaperResult | None = year_matches[0] if year_matches else candidates[0]
    else:
        best = candidates[0] if candidates else None

    return {  # type: ignore[return-value]
        "found": best is not None,
        "citation_string": citation_string,
        "paper": best,
        "candidates": candidates,
        "is_accepted_source_type": best["is_accepted_source_type"] if best else False,
    }


def lookup_paper(doi_or_title: str) -> PaperResult:
    """
    Resolve a DOI or title string to a PaperResult.

    Parameters
    ----------
    doi_or_title : str
        Either a DOI (e.g. "10.1016/j.jfi.2012.04.001" or
        "doi:10.xxx/yyy") or a free-text title.  DOIs use the
        Semantic Scholar /paper/DOI: endpoint; titles use /paper/search.

    Returns
    -------
    PaperResult
        Full metadata, or an empty PaperResult (source_type="unknown")
        on failure.
    """
    cleaned = doi_or_title.strip()
    if cleaned.lower().startswith("doi:"):
        cleaned = cleaned[4:].strip()

    is_doi = cleaned.startswith("10.") and "/" in cleaned

    if is_doi:
        url = f"{_SS_PAPER_URL}/DOI:{cleaned}"
        try:
            data = _api_get(url, {"fields": _SS_FIELDS})
            return _paper_from_ss(data)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[literature] DOI lookup failed for {cleaned!r}: {exc}",
                file=sys.stderr,
            )
            return _empty_paper(doi_or_title)

    # Title search
    try:
        data = _api_get(
            _SS_SEARCH_URL,
            {"query": doi_or_title, "fields": _SS_FIELDS, "limit": 1},
        )
        papers = data.get("data") or []
        return _paper_from_ss(papers[0]) if papers else _empty_paper(doi_or_title)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[literature] Title lookup failed for {doi_or_title!r}: {exc}",
            file=sys.stderr,
        )
        return _empty_paper(doi_or_title)


# ---------------------------------------------------------------------------
# Claude-powered query generation and paper scoring
# ---------------------------------------------------------------------------


def _get_anthropic_client():
    """Return an Anthropic client. Raises EnvironmentError if key absent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before calling build_search_query() or score_paper()."
        )
    try:
        import anthropic  # local import keeps anthropic optional at module level
        return anthropic.Anthropic(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required. Install it with: pip install anthropic"
        ) from exc


def build_search_query(edge: dict, nodes: dict) -> str:
    """
    Generate an academic search query for a causal edge using Claude.

    Makes a single Claude API call (max_tokens=100) and returns the stripped
    query string suitable for Semantic Scholar.

    Parameters
    ----------
    edge : dict
        Edge dict with at minimum 'from', 'to', 'sign' fields.
    nodes : dict
        Dict of node_id → node_dict (from load_nodes).

    Returns
    -------
    str
        5-8 word academic search query.

    Raises
    ------
    EnvironmentError
        If ANTHROPIC_API_KEY is not set.
    """
    client = _get_anthropic_client()

    from_node = edge.get("from", "")
    to_node = edge.get("to", "")
    sign = edge.get("sign", "+")

    from_label = nodes.get(from_node, {}).get("name_en", from_node.replace("_", " "))
    to_label = nodes.get(to_node, {}).get("name_en", to_node.replace("_", " "))
    sign_word = "negatively" if sign == "-" else "positively"

    prompt = (
        f"Generate a short academic search query (5-8 words) for finding papers about "
        f"this causal relationship: {from_label} {sign_word} affects {to_label}. "
        f"Return only the query string, nothing else."
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query.

    Uses the module-level _api_get() for rate limiting and disk caching.
    Skips papers with no abstract.

    Parameters
    ----------
    query : str
        Search query string.
    limit : int
        Maximum number of results to request (default 10).

    Returns
    -------
    list[dict]
        Raw Semantic Scholar paper dicts filtered to those with abstracts.
        Returns empty list on any HTTP error (status code printed to stderr).
    """
    try:
        data = _api_get(
            _SS_SEARCH_URL,
            {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,citationCount,externalIds",
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[literature] search_semantic_scholar HTTP error for {query!r}: {exc}",
            file=sys.stderr,
        )
        return []

    papers = data.get("data") or []
    return [p for p in papers if p.get("abstract")]


def score_paper(paper: dict, edge: dict, nodes: dict) -> dict:
    """
    Score a single paper's relevance to a causal edge claim using Claude.

    Parameters
    ----------
    paper : dict
        Raw paper dict from search_semantic_scholar (needs 'title', 'abstract').
    edge : dict
        Edge dict with 'from', 'to', 'sign' fields.
    nodes : dict
        Dict of node_id → node_dict.

    Returns
    -------
    dict
        {"verdict": "SUPPORTS"|"CONTRADICTS"|"IRRELEVANT",
         "justification": str,
         "paper": paper}
        On parse failure, returns verdict "IRRELEVANT" with justification
        "parse error".
    """
    client = _get_anthropic_client()

    from_node = edge.get("from", "")
    to_node = edge.get("to", "")
    sign = edge.get("sign", "+")

    from_label = nodes.get(from_node, {}).get("name_en", from_node.replace("_", " "))
    to_label = nodes.get(to_node, {}).get("name_en", to_node.replace("_", " "))
    sign_word = "negatively" if sign == "-" else "positively"

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    prompt = (
        f"Does this paper provide empirical evidence that {from_label} {sign_word} "
        f"affects {to_label}? "
        f"Paper title: {title}. "
        f"Abstract: {abstract}. "
        f"Respond with exactly one of: SUPPORTS, CONTRADICTS, IRRELEVANT. "
        f"Then on a new line, one sentence of justification."
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        lines = text.split("\n", 1)
        first = lines[0].strip().upper()

        verdict = "IRRELEVANT"
        for v in ("SUPPORTS", "CONTRADICTS", "IRRELEVANT"):
            if first.startswith(v):
                verdict = v
                break

        justification = lines[1].strip() if len(lines) > 1 else text
    except Exception as exc:  # noqa: BLE001
        print(f"[literature] score_paper error: {exc}", file=sys.stderr)
        verdict = "IRRELEVANT"
        justification = "parse error"

    return {"verdict": verdict, "justification": justification, "paper": paper}


def run_literature_review(
    edge_id: str,
    graph_path: str,
    nodes_path: str,
    dry_run: bool = False,
) -> dict:
    """
    Run a literature review for a single speculative edge.

    Searches Semantic Scholar, scores each abstract with Claude, and annotates
    the edge with the review result.  The edge 'status' field is NOT changed
    (automated promotion is prohibited by governance.json §speculative_to_tested).
    Instead, a 'tier' annotation and 'literature_review' sub-object are written.

    Parameters
    ----------
    edge_id : str
        ID of the speculative edge to review.
    graph_path : str
        Path to the graph JSON file (typically speculative_graph.json).
    nodes_path : str
        Path to nodes.json.
    dry_run : bool
        If True, does not write changes to disk.

    Returns
    -------
    dict
        One of:
          {"result": "promoted",       "votes": n, "papers": [...]}
          {"result": "literature_gap", "votes": 0, "papers_tried": [...]}
          {"result": "no_results",     "query": str}

    Raises
    ------
    ValueError
        If edge_id is not found, or if the edge status is not 'speculative'.
    """
    from datetime import datetime, timezone

    nodes = _load_nodes(nodes_path)

    graph_file = Path(graph_path)
    with open(graph_file, encoding="utf-8") as fh:
        graph_data = json.load(fh)

    edges = graph_data.get("edges", graph_data)
    if not isinstance(edges, list):
        raise ValueError(f"Graph file {graph_path} does not contain an edge list")

    edge = next((e for e in edges if e.get("id") == edge_id), None)
    if edge is None:
        raise ValueError(f"Edge '{edge_id}' not found in {graph_path}")
    if edge.get("status") != "speculative":
        raise ValueError(
            f"Edge '{edge_id}' has status='{edge.get('status')}', expected 'speculative'. "
            f"Only speculative edges can be submitted for automated literature review."
        )

    query = build_search_query(edge, nodes)
    papers = search_semantic_scholar(query)

    if not papers:
        return {"result": "no_results", "query": query}

    scored = [score_paper(p, edge, nodes) for p in papers]
    supporting = [s for s in scored if s["verdict"] == "SUPPORTS"]
    votes = len(supporting)
    reviewed_at = datetime.now(timezone.utc).isoformat()

    from graph_io import save_graph as _save_graph, validate as _validate_graph

    if votes >= 1:
        edge["tier"] = "tested"
        edge["literature_review"] = {
            "papers": supporting,
            "reviewed_at": reviewed_at,
            "method": "semantic_scholar_llm",
        }
        if not dry_run:
            _validate_graph(edges, nodes, graph_type="speculative")
            _save_graph(edges, graph_file)
        return {"result": "promoted", "votes": votes, "papers": supporting}

    edge["tier"] = "literature_gap"
    edge["literature_review"] = {
        "papers_tried": scored,
        "reviewed_at": reviewed_at,
        "method": "semantic_scholar_llm",
    }
    if not dry_run:
        _validate_graph(edges, nodes, graph_type="speculative")
        _save_graph(edges, graph_file)
    return {"result": "literature_gap", "votes": 0, "papers_tried": scored}


def batch_review_speculative(
    graph_path: str,
    nodes_path: str,
    dry_run: bool = False,
) -> list[dict]:
    """
    Run automated literature review for every speculative edge in a graph file.

    Prints progress to stdout.  Sleeps 1 second between edges to respect
    Semantic Scholar rate limits.

    Parameters
    ----------
    graph_path : str
        Path to the graph JSON file.
    nodes_path : str
        Path to nodes.json.
    dry_run : bool
        If True, does not write changes to disk.

    Returns
    -------
    list[dict]
        Result dicts from run_literature_review, one per speculative edge,
        each augmented with 'edge_id'.
    """
    graph_file = Path(graph_path)
    with open(graph_file, encoding="utf-8") as fh:
        graph_data = json.load(fh)

    edges = graph_data.get("edges", graph_data)
    speculative = [e for e in edges if e.get("status") == "speculative"]
    total = len(speculative)

    results: list[dict] = []
    for i, edge in enumerate(speculative, 1):
        edge_id = edge.get("id", f"<unknown-{i}>")
        print(f"Reviewing edge {edge_id} ({i}/{total})...", flush=True)
        try:
            result = run_literature_review(
                edge_id, graph_path, nodes_path, dry_run=dry_run
            )
            result["edge_id"] = edge_id
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[literature] Error reviewing edge {edge_id}: {exc}", file=sys.stderr
            )
            results.append({"edge_id": edge_id, "result": "error", "error": str(exc)})

        if i < total:
            time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args = sys.argv[1:]

    if not _args or _args[0] in ("-h", "--help"):
        _print_usage()
        print(
            "Literature review (review pipeline) usage:\n"
            "  python literature.py --edge EDGE_ID          # single edge, live run\n"
            "  python literature.py --batch                 # all speculative edges\n"
            "  python literature.py --batch --dry-run       # preview without writing\n"
        )
        sys.exit(0)

    # ── new review-pipeline modes ────────────────────────────────────────────
    _REPO_ROOT_CLI = Path(__file__).parent.parent
    _DEFAULT_GRAPH = str(_REPO_ROOT_CLI / "knowledge" / "speculative_graph.json")
    _DEFAULT_NODES = str(_REPO_ROOT_CLI / "knowledge" / "nodes.json")

    if _args[0] == "--batch":
        _dry = "--dry-run" in _args
        print(
            f"[literature] batch review — graph: {_DEFAULT_GRAPH}"
            + (" (DRY RUN — no writes)" if _dry else "")
        )
        _batch_results = batch_review_speculative(
            _DEFAULT_GRAPH, _DEFAULT_NODES, dry_run=_dry
        )
        print(f"\n[literature] batch complete — {len(_batch_results)} edge(s) processed")
        for _r in _batch_results:
            print(f"  {_r.get('edge_id')}: {_r.get('result')}"
                  + (f" (votes={_r['votes']})" if "votes" in _r else ""))
        sys.exit(0)

    if _args[0] == "--edge":
        if len(_args) < 2:
            print("Usage: python literature.py --edge EDGE_ID")
            sys.exit(1)
        _edge_id = _args[1]
        _dry = "--dry-run" in _args
        print(
            f"[literature] reviewing edge {_edge_id!r}"
            + (" (DRY RUN)" if _dry else "")
        )
        _result = run_literature_review(
            _edge_id, _DEFAULT_GRAPH, _DEFAULT_NODES, dry_run=_dry
        )
        print(f"  result : {_result['result']}")
        if "votes" in _result:
            print(f"  votes  : {_result['votes']}")
        if "query" in _result:
            print(f"  query  : {_result['query']}")
        sys.exit(0)

    # ── legacy search / verify / lookup modes ───────────────────────────────
    _cmd = _args[0]

    if _cmd == "search":
        if len(_args) != 4:
            print(
                "Usage: python -m src.literature search "
                "<from_node> <to_node> <sign>"
            )
            sys.exit(1)
        _, _from, _to, _sign = _args
        _results = search_edge_support(_from, _to, _sign)
        _print_search_results(_results, _from, _to, _sign)

    elif _cmd == "verify":
        if len(_args) < 2:
            print("Usage: python -m src.literature verify <citation_string>")
            sys.exit(1)
        _cit = " ".join(_args[1:])
        _cresult = verify_citation(_cit)
        _print_citation_result(_cresult)

    elif _cmd == "lookup":
        if len(_args) < 2:
            print("Usage: python -m src.literature lookup <doi_or_title>")
            sys.exit(1)
        _target = " ".join(_args[1:])
        _paper = lookup_paper(_target)
        _print_paper(_paper, rank=None)
        print()

    else:
        print(f"Unknown command: {_cmd!r}")
        _print_usage()
        sys.exit(1)
