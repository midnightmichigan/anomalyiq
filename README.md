# AnomalyIQ — evidence-backed anomaly explanations for intelligent industry

An intelligent-industry agent that investigates production anomalies, gathers
evidence, and produces cited explanations with first-response maintenance
actions. Built for the **AMD Developer Hackathon: ACT III** (lablab.ai × AMD),
Intelligent Industry track.

## What it does

Feed it a production anomaly alert; it runs a 4-angle investigation plan,
gathers key-free live evidence (Wikipedia / arXiv), and renders a cited
markdown report with:

- alert card with confidence score + backend badge
- findings with evidence citations
- first-response maintenance checklist
- explicit limitations line

Two interfaces: a CLI (`anomaly/demo.py`) and a screen-recordable web demo
(`anomaly/webdemo.py`, stdlib-only `http.server`).

## The AMD seam (honest)

The model backend (`amdcloud/model.py`) is an OpenAI-compatible client aimed
at a vLLM endpoint on **AMD Developer Cloud** via `AMD_BASE_URL`,
`AMD_MODEL`, `AMD_API_KEY`. When no endpoint is configured it degrades to a
deterministic stub (`StubAmd`) and says so in the output — the same engine,
same reports, honest badge. During the hackathon build window the AMD
Developer Cloud endpoint was not reachable (Developer Program credits were
pending), so the live-AMD path is validated against the stub; see `test_flow.py`.

## Run it

```bash
python3 -m anomaly.demo                       # 3-anomaly end-to-end demo
python3 -m anomaly.demo --markdown --anomaly ANO-101
python3 -m anomaly.webdemo --port 8136        # web demo UI
python3 test_flow.py                           # 29/29 checks, stdlib only
```

No dependencies beyond the Python standard library. No API keys required for
the stub path.

## Layout

- `amdcloud/model.py` — OpenAI-compatible vLLM client + `StubAmd` fallback
- `anomaly/backend.py` — `AmdModelBackend` adapter (`chat(system, user)`)
- `anomaly/explain.py` — fixture corpus, investigation plan, report renderer
- `anomaly/demo.py` — CLI
- `anomaly/webdemo.py` — web demo UI
- `test_flow.py` — 29 automated checks
- `video-assets/` — terminal demo recording assets
