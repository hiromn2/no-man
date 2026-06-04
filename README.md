# No-Man（ノーマン）

**A causal risk-intelligence agent.** You give it a business decision; it returns a structured report of the adverse consequences that decision could set in motion — each one a traceable chain through a signed causal graph, grounded in academic literature, with an explicit confidence tier and suggested mitigations.

No-Man reasons about **signs, not magnitudes**. It does not claim "this will cost ¥40M." It claims "this decision pushes A up, which pushes B down, which raises the risk of an adverse outcome — and here is the literature behind each link." This is a deliberate design choice, not a limitation: see [Why signs only](#why-signs-only).

The name refers to the institutionalized devil's-advocate role (the *nay-man*) and to the adversarial-robustness literature it draws from. Its job is to argue against you, carefully.

---

## What it does

```
Decision ──▶ [ retrieval over causal literature ]
                       │
                       ▼
          [ sign-constrained DAG traversal ]   decision → A → B → adverse outcome
                       │
                       ▼
          [ confidence tiering + mitigations ]
                       │
                       ▼
   Structured report  (premises → chain → conclusion, with epistemic hedging)
```

The output is constrained JSON rendered into a Japanese-language report with a strict *premises → chain → conclusion* structure, so every claim carries its assumptions and its evidence tier on its face.

---

## Why this is an LLM-systems project, not a prompt

The hard problem with LLMs in a risk setting is not generating plausible text — it is *stopping* them from generating plausible text that isn't grounded. No-Man's architecture is built around that constraint:

- **The graph is the source of truth, not the model.** The LLM does not free-associate consequences. It traverses a curated signed causal graph (currently 28 nodes, 41 edges) and is only permitted to report chains that exist as edges. This is retrieval-grounded generation where the retrieval structure is a causal DAG rather than a vector store.
- **Sign propagation is mechanical.** Whether a chain ends in an adverse outcome is computed by propagating +/− signs along the path, not asked of the model. The model's job is explanation and surfacing, not adjudication.
- **A governance layer gates what the graph is allowed to assert.** Edges live in a three-tier epistemic ladder — *speculative* / *literature-supported* / *empirically-tested*. Critically, **only data or published research can promote an edge** up the ladder. Human opinion cannot. This is the mechanism that keeps the system honest as it grows.
- **Dual-loop operation.** A synchronous loop answers worker queries against the current graph; an asynchronous LLM-driven loop proposes new edges from literature for governance review. The product is designed to *grow through use*, not to ship complete.

---

## Why signs only

Magnitude predictions in a system that people act on are self-undermining — the Lucas critique. If No-Man said "branch closure costs ¥40M" and banks acted on it, the ¥40M would stop being true. Sign claims ("this raises herding risk") are far more robust to that performativity, because the *direction* of an effect survives behavioral response in a way the *size* does not. Reasoning over signs is what lets the system make claims that remain valid once it's deployed at scale.

---

## What it's honest about

The repo ships with a diagnosed limitations section, because a risk tool that hides its own risks is self-refuting.

- **Coverage gaps.** Of 12 canonical decision types, several currently produce zero adverse chains — traced to welfare nodes defined too narrowly and structural gaps in the graph for those decisions. Documented, not hidden.
- **Self-application.** No-Man was run against the decision to deploy No-Man. The findings (automation bias, a standard-of-care ratchet, herding risk under wide deployment) all sit in the lowest "worth considering" tier — which is exactly where the system's own rules say they belong.

---

## Repository structure

```
no-man/
├── knowledge/
│   ├── seed_graph.json        ← tested causal edges (literature-backed)
│   ├── speculative_graph.json ← proposed edges pending validation
│   ├── nodes.json             ← node registry
│   ├── governance.json        ← voting rules for edge promotion
│   └── changelog.md           ← immutable change log
├── decisions/
│   └── decision_types.json    ← 12 canonical decision types
├── src/
│   ├── traversal.py           ← DAG traversal and sign propagation
│   ├── confidence.py          ← confidence tier assignment
│   ├── graph_io.py            ← graph load/save/validate
│   ├── feedback.py            ← edge promotion and voting
│   ├── literature.py          ← literature retrieval
│   └── report_generator.py   ← LLM-powered report (Japanese/English)
├── prompts/
│   ├── report_ja.txt
│   └── report_en.txt
├── tests/
├── output/                    ← sample report output
└── CLAUDE.md                  ← agent instructions (design constraints)
```

---

## Quickstart

```bash
git clone https://github.com/hiromn2/no-man.git
cd no-man
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Set ANTHROPIC_API_KEY and optionally SEMANTIC_SCHOLAR_API_KEY

python run_d03_report.py
```

---

## Stack

Python · JSON-based graph storage · LLM orchestration with constrained/structured output · retrieval over a curated causal-literature corpus.

## Status

Research prototype. MVP implemented with a passing test suite. Built as an independent exploration of how to put a foundation model on top of a governed, auditable reasoning structure rather than letting it reason unconstrained.

## License

MIT. See [LICENSE](LICENSE).
