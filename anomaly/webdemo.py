"""AMD ACT III — anomaly explainer web demo UI.

Stdlib-only (`http.server` + `html`). Serves an anomaly selector form, runs
the full explain pipeline on submit, and renders the cited findings +
first-response maintenance checklist + heuristic confidence + limitations
as one screen-recordable page.

Default uses the stub search + stub AMD backend: fully deterministic, zero
keys, zero network. The live checkbox switches to the key-free
Wikipedia/arXiv backends. The model backend comes from the environment
(`AmdModelBackend.auto()`): stub-synthetic until the AMD Developer Cloud
vLLM endpoint is configured — then the same UI runs on AMD infra (the
prize-mandated AMD workload).

Run: python3 -m anomaly.webdemo [--port 8136]
Then open http://localhost:8136/
"""

from __future__ import annotations

import argparse
import html
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from anomaly.explain import ANOMALIES, investigate
from anomaly.backend import AmdModelBackend

PAGE_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:2rem auto;
padding:0 1rem;color:#1a1a1a;background:#fafafa}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:1.2rem;margin:1rem 0}
h1{font-size:1.6rem} h2{font-size:1.15rem;margin-top:1.4rem}
.alert{background:#fef2f2;border-left:4px solid #dc2626;padding:0.8rem 1rem;border-radius:0 8px 8px 0}
.conf{display:inline-block;font-size:0.85rem;background:#eef2ff;color:#3730a3;
border-radius:999px;padding:0.15rem 0.7rem;margin-left:0.5rem}
.finding{border-left:3px solid #2563eb;padding-left:0.9rem;margin:0.9rem 0}
.prose{white-space:pre-wrap;font-size:0.95rem}
.recs{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:0.6rem 1rem}
.recs li{margin:0.45rem 0}
.lim{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:0.6rem 1rem}
select{padding:0.55rem;font-size:1rem;border:1px solid #ccc;border-radius:6px}
button{padding:0.55rem 1.2rem;font-size:1rem;border:0;border-radius:6px;background:#dc2626;color:#fff}
label{font-size:0.9rem;margin-left:0.8rem}
.badge{font-size:0.75rem;padding:0.1rem 0.55rem;border-radius:999px}
.badge.live{background:#dcfce7;color:#166534}.badge.fixture{background:#fef3c7;color:#92400e}
select{width:100%}
.hist{font-size:0.85rem;color:#555}
"""

_history: list[dict] = []
_lock = threading.Lock()


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def render_form() -> str:
    opts = "".join(
        "<option value=\"%s\">%s — %s (%s)</option>"
        % (esc(a["id"]), esc(a["id"]), esc(a["equipment"]), esc(a["error_code"]))
        for a in ANOMALIES
    )
    return (
        "<div class=\"card\">"
        "<h1>&#9888;&#65039; Anomaly Explainer — Intelligent Industry</h1>"
        "<p>Pick a production anomaly. The agent plans investigation angles, "
        "gathers evidence (Wikipedia + arXiv, key-free), and returns a cited "
        "explanation with a first-response maintenance checklist.</p>"
        "<form method=\"post\" action=\"/investigate\">"
        "<select name=\"anomaly\" required>%s</select><br><br>"
        "<button type=\"submit\">Investigate</button>"
        "<label><input type=\"checkbox\" name=\"live\" value=\"1\"> live evidence "
        "(Wikipedia + arXiv, no key needed)</label>"
        "</form>"
        "</div>" % opts
    )


def render_report(result: dict, live: bool) -> str:
    badge = ('<span class="badge live">LIVE EVIDENCE</span>' if live
             else '<span class="badge fixture">FIXTURE DEMO</span>')
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Anomaly %s</title>" % esc(result["anomaly_id"]),
        "<style>%s</style></head><body>" % PAGE_CSS,
        render_form(),
        "<div class='card'>",
        "<div class='alert'><strong>Anomaly %s</strong> — %s %s"
        "<span class='conf'>confidence (heuristic) %.2f</span></div>"
        % (esc(result["anomaly_id"]), esc(result["equipment"]), badge,
           result["confidence"]),
        "<p><strong>Signal:</strong> %s<br><strong>Error code:</strong> %s "
        "| <strong>Since:</strong> %s<br><strong>Model backend:</strong> %s "
        "| <strong>Fixture data:</strong> %s</p>"
        % (esc(result["signal"]), esc(result["error_code"]), esc(result["since"]),
           esc(result["backend"]), esc(result["fixture_data"])),
        "<h2>Findings</h2>",
    ]
    for f in result["findings"]:
        parts.append(
            "<div class='finding'><h3>%s</h3>"
            "<div class='prose'>%s</div>"
            "<div class='hist'>evidence items: %d</div></div>"
            % (esc(f["angle"]), esc(f["prose"]), f["sources"]))
    parts.append("<h2>First-response checklist (fixture maintenance table)</h2>")
    parts.append("<ol class='recs'>")
    parts += ["<li>%s</li>" % esc(r) for r in result["recommendations"]]
    parts.append("</ol>")
    parts.append("<div class='lim'><strong>Limitations:</strong> %s</div>"
                 % esc(result["limitations"]))
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _form_data(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    raw = handler.rfile.read(length).decode("utf-8", "replace")
    return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, code: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(404)
            return
        with _lock:
            hist = list(_history)
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Anomaly Explainer</title>",
            "<style>%s</style></head><body>" % PAGE_CSS,
            render_form(),
        ]
        if hist:
            parts.append("<div class='card hist'><h2>Session history</h2><ul>")
            parts += ["<li>%s — %s</li>" % (esc(h["anomaly_id"]), esc(h["equipment"]))
                      for h in hist]
            parts.append("</ul></div>")
        parts.append("</body></html>")
        self._send("\n".join(parts))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/investigate":
            self.send_error(404)
            return
        form = _form_data(self)
        anomaly_id = form.get("anomaly", "")
        target = next((a for a in ANOMALIES if a["id"] == anomaly_id), None)
        if target is None:
            self._send("<h1>Unknown anomaly id</h1>", code=400)
            return
        live = form.get("live") == "1"
        try:
            result = investigate(target, model=AmdModelBackend.auto(),
                                 live_search=live)
        except Exception as e:  # noqa: BLE001 - degrade to an error page
            self._send("<h1>Investigation failed</h1><p>%s</p>" % esc(e),
                       code=500)
            return
        with _lock:
            _history.append({"anomaly_id": result["anomaly_id"],
                             "equipment": result["equipment"]})
        self._send(render_report(result, live))

    def log_message(self, *args) -> None:  # quiet for demo recording
        pass


def serve(port: int) -> None:
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8136)
    args = ap.parse_args()
    print("Anomaly Explainer demo at http://localhost:%d/" % args.port)
    serve(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
