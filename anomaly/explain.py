"""anomaly/explain.py — Intelligent Industry anomaly explanation engine.

The AMD ACT III Intelligent Industry track asks for: explain production
anomalies, help technicians search manuals / maintenance records, recommend
maintenance actions. This engine does exactly that:

1. plan(anomaly)      -> investigation angles with search queries
2. gather(anomaly, search) -> evidence per angle (key-free backends by
                              default: Wikipedia encyclopedic + arXiv scholarly)
3. investigate(...)   -> cited explanation + maintenance recommendations,
                         synthesized with AmdModelBackend (AMD Developer
                         Cloud when configured; honest extractive fallback
                         on the stub)
4. report_markdown(...) -> human-readable report for the demo video

Anomaly inputs are equipment telemetry fixtures until the event's AMD cloud
pod exposes real endpoints. All fixture data is labeled synthetic wherever
it surfaces.

Stdlib only.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEbius_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "nebius-nvidia-hackathon"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _NEbius_DIR)

from deepdive.search import (  # noqa: E402
    StubSearch, WikipediaSearch, ArxivSearch, MultiSearch, best_backend)

from .backend import AmdModelBackend  # noqa: E402

# --- fixture anomaly corpus (synthetic, labeled) ------------------------------
ANOMALIES = [
    {
        "id": "ANO-101",
        "equipment": "CNC milling machine (CNC-3)",
        "signal": "spindle vibration +62% above baseline for 40 minutes, "
                  "bearing temperature rising to 78 C",
        "error_code": "VIB-OVERSHOOT",
        "since": "2026-10-11T06:12",
    },
    {
        "id": "ANO-102",
        "equipment": "hydraulic press (P-2)",
        "signal": "clamp-circuit pressure sagging from 210 bar to 165 bar "
                  "mid-cycle, cycle time up 12%",
        "error_code": "PRESSURE-SAG",
        "since": "2026-10-11T08:40",
    },
    {
        "id": "ANO-103",
        "equipment": "rooftop HVAC unit (RTU-7)",
        "signal": "compressor short-cycling: 14 starts per hour, "
                  "discharge pressure unstable",
        "error_code": "SHORT-CYCLE",
        "since": "2026-10-11T10:05",
    },
]

# Deterministic first-response actions per error code, from the (fixture)
# maintenance table — these are guidance, not commands. Labeled as such in
# the report.
FIRST_RESPONSE = {
    "VIB-OVERSHOOT": [
        "Stop the spindle and check bearing play by hand; listen for "
        "grinding or feel for notchiness.",
        "Verify the tool is seated and torqued to spec; a loose tool holder "
        "mimics bearing failure.",
        "If vibration persists after re-seating, quarantine the machine and "
        "schedule a bearing inspection — do not run a full-speed cycle.",
    ],
    "PRESSURE-SAG": [
        "Check hydraulic fluid level and temperature; top up with the "
        "specified grade only.",
        "Inspect visible fittings and hoses for leaks; tighten or tag out "
        "any weeping line.",
        "If pressure still sags, suspect the pump or relief valve — do not "
        "bypass the pressure switch to keep running.",
    ],
    "SHORT-CYCLE": [
        "Check the thermostat placement and calibration; a sensor in a "
        "cold draft causes rapid cycling.",
        "Inspect the condenser coil for fouling; clean with a coil-safe "
        "cleaner, never a pressure washer.",
        "Measure run vs off times; cycling above 8 starts/hr shortens "
        "compressor life — schedule a controls check.",
    ],
}

_HONESTY_LINE = (
    "Limitations: anomaly readings are synthetic fixture data for the demo; "
    "evidence comes from the configured search backends (key-free Wikipedia/"
    "arXiv by default); confidence is a heuristic, not a calibrated "
    "probability. A licensed technician should verify before acting.")


def plan(anomaly: dict) -> list[dict]:
    """Investigation angles for an anomaly."""
    eq = anomaly["equipment"]
    sig = anomaly["signal"]
    return [
        {"angle": "Likely causes",
         "query": "what causes %s failure symptoms" % eq},
        {"angle": "Failure modes",
         "query": "industrial maintenance %s common failure modes" % eq},
        {"angle": "Recommended maintenance actions",
         "query": "recommended maintenance procedure for: %s" % sig},
        {"angle": "Similar field cases",
         "query": "field case study %s anomaly" % eq},
    ]


def _keyfree_search() -> MultiSearch:
    """Default search: Wikipedia + arXiv, both key-free, degrade-safe."""
    return MultiSearch([WikipediaSearch(), ArxivSearch()], per_backend=2)


def gather(anomaly: dict, search=None) -> dict:
    """Run the plan against the search backend -> angle -> evidence items."""
    if search is None:
        search = StubSearch()
    evidence = {}
    for step in plan(anomaly):
        try:
            evidence[step["query"]] = search.search(step["query"], max_results=4) or []
        except Exception:  # noqa: BLE001 - backends degrade to [] by design
            evidence[step["query"]] = []
    return evidence


def _extractive(angle: str, items: list) -> str:
    lines = ["%s (evidence-backed):" % angle]
    for i, it in enumerate(items, 1):
        lines.append("- [%d] %s (%s): %s"
                     % (i, it.get("title", "?"), it.get("url", "?"),
                        it.get("snippet", "")[:160]))
    if len(items) == 0:
        lines.append("- (no evidence returned for this angle)")
    return "\n".join(lines)


def _confidence(evidence: dict) -> float:
    n = sum(len(v) for v in evidence.values())
    return round(min(0.9, 0.3 + 0.12 * n), 2)


def investigate(anomaly: dict, search=None, model: AmdModelBackend | None = None,
                live_search: bool = False) -> dict:
    """Full pipeline: plan -> evidence -> cited explanation + recommendations."""
    model = model or AmdModelBackend()
    if search is None:
        search = _keyfree_search() if live_search else StubSearch()
    evidence = gather(anomaly, search)

    findings = []
    for step in plan(anomaly):
        items = evidence.get(step["query"], [])
        if model.mode == "amd-cloud-live":
            context = "\n".join(
                "[%d] %s (%s): %s" % (i + 1, it.get("title", "?"),
                                       it.get("url", "?"), it.get("snippet", ""))
                for i, it in enumerate(items))
            prose = model.chat(
                "You are a precise industrial-maintenance assistant. "
                "Explain the anomaly angle in 2-4 sentences, citing sources "
                "by [n]. Never invent facts or measurements.",
                "Anomaly: %s\nEquipment: %s\nAngle: %s\nEvidence:\n%s"
                % (anomaly["signal"], anomaly["equipment"], step["angle"], context))
            if prose.startswith("[MODEL-ERROR"):
                prose = _extractive(step["angle"], items)
        else:
            prose = _extractive(step["angle"], items)
        findings.append({"angle": step["angle"], "prose": prose,
                         "sources": len(items)})

    recommendations = list(
        FIRST_RESPONSE.get(anomaly.get("error_code", ""),
                           ["No first-response checklist for this error code "
                            "in the fixture table — escalate to maintenance."]))

    return {
        "anomaly_id": anomaly["id"],
        "equipment": anomaly["equipment"],
        "signal": anomaly["signal"],
        "error_code": anomaly["error_code"],
        "since": anomaly["since"],
        "backend": model.mode,
        "findings": findings,
        "recommendations": recommendations,
        "confidence": _confidence(evidence),
        "limitations": _HONESTY_LINE,
        "fixture_data": True,
    }


def report_markdown(result: dict) -> str:
    """Human-readable report for the demo video / judging."""
    L = ["# Anomaly Report %s (synthetic fixture)" % result["anomaly_id"],
         "",
         "Equipment: %s" % result["equipment"],
         "Signal: %s" % result["signal"],
         "Error code: %s | Since: %s" % (result["error_code"], result["since"]),
         "Model backend: %s" % result["backend"],
         "Confidence (heuristic): %.2f" % result["confidence"],
         ""]
    for f in result["findings"]:
        L += ["## %s" % f["angle"], "", f["prose"], ""]
    L += ["## First-response checklist (fixture maintenance table)", ""]
    for i, rec in enumerate(result["recommendations"], 1):
        L.append("%d. %s" % (i, rec))
    L += ["", "_%s_" % result["limitations"]]
    return "\n".join(L)
