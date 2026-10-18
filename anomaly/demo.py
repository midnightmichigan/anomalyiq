"""anomaly/demo.py — end-to-end CLI demo of the anomaly explanation engine.

Run from ~/workspace/bounty-hunts/amd-act-iii/:
    python3 -m anomaly.demo [--live-search] [--markdown]

Default uses the stub search + stub AMD backend: fully deterministic,
zero keys, zero network. --live-search uses the key-free Wikipedia/arXiv
backends (requires network). The model backend is chosen via environment:
set AMD_BASE_URL / AMD_MODEL / AMD_API_KEY to route synthesis through
AMD Developer Cloud (the prize-eligibility workload); otherwise the
extractive fallback labels everything synthetic.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from anomaly.explain import ANOMALIES, investigate, report_markdown  # noqa: E402
from anomaly.backend import AmdModelBackend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-search", action="store_true",
                    help="use key-free Wikipedia/arXiv backends")
    ap.add_argument("--markdown", action="store_true",
                    help="print the full markdown report per anomaly")
    ap.add_argument("--anomaly", default=None,
                    help="run only one anomaly id (default: all)")
    args = ap.parse_args()

    model = AmdModelBackend.auto()
    print("model backend: %s" % model.mode)

    targets = ANOMALIES
    if args.anomaly:
        targets = [a for a in ANOMALIES if a["id"] == args.anomaly]
        if not targets:
            print("unknown anomaly id; choose from: %s"
                  % ", ".join(a["id"] for a in ANOMALIES))
            return 2

    results = []
    for anomaly in targets:
        result = investigate(anomaly, model=model, live_search=args.live_search)
        results.append(result)
        if args.markdown:
            print()
            print(report_markdown(result))
            print()
            print("=" * 70)
        else:
            src = sum(f["sources"] for f in result["findings"])
            print("%s | %s | backend=%s | sources=%d | confidence=%.2f | "
                  "recommendations=%d"
                  % (result["anomaly_id"], result["equipment"], result["backend"],
                     src, result["confidence"], len(result["recommendations"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
