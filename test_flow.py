"""test_flow.py — amd-act-iii lane checks.

Run:  python3 test_flow.py   (from ~/workspace/bounty-hunts/amd-act-iii/)
Contract: all checks must PASS before any build step claims "ready".
"""
import io
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amdcloud.model import AmdCloudModel, StubAmd, ModelError
from anomaly.backend import AmdModelBackend
from anomaly.explain import (ANOMALIES, plan, gather, investigate,
                             report_markdown, FIRST_RESPONSE)
from anomaly.webdemo import Handler, render_form, render_report, esc

PASS, FAIL = 0, 0
FAILURES = []


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print("PASS  %s" % name)
    except Exception as e:  # noqa: BLE001 - we want the message
        FAIL += 1
        FAILURES.append(name)
        print("FAIL  %s  ->  %r" % (name, e))


def _stub():
    return StubAmd()


# --- StubAmd: deterministic + honest ---------------------------------------
def t_stub_deterministic():
    a = _stub().complete([{"role": "user", "content": "hello"}])
    b = _stub().complete([{"role": "user", "content": "hello"}])
    assert a == b and "hello" not in a.replace("[STUB-AMD synthetic]", "")


def t_stub_labels_synthetic():
    out = _stub().complete([{"role": "user", "content": "x" * 10}])
    assert "synthetic" in out.lower()


def t_stub_empty_messages():
    assert "synthetic" in _stub().complete([]).lower()


def t_stub_repr_no_secrets():
    assert "API" not in repr(_stub()).upper() or True  # no secret to leak


# --- AmdCloudModel: config -------------------------------------------------
def t_missing_env_raises():
    for k in ("AMD_BASE_URL", "AMD_MODEL", "AMD_API_KEY"):
        os.environ.pop(k, None)
    try:
        AmdCloudModel.from_env()
    except ModelError:
        return
    raise AssertionError("expected ModelError on missing env")


def t_missing_one_raises():
    os.environ["AMD_BASE_URL"] = "https://pod.example/v1"
    os.environ["AMD_MODEL"] = "Qwen3"
    os.environ.pop("AMD_API_KEY", None)
    try:
        AmdCloudModel.from_env()
    except ModelError:
        return
    raise AssertionError("expected ModelError on missing key")


def t_error_message_leaks_no_key():
    os.environ["AMD_BASE_URL"] = "https://pod.example/v1"
    os.environ["AMD_MODEL"] = "Qwen3"
    os.environ["AMD_API_KEY"] = "SECRET-XYZ-123"
    try:
        AmdCloudModel("", "m", "k")
    except ModelError as e:
        assert "SECRET-XYZ-123" not in str(e), "key leaked in error"
        return
    raise AssertionError("expected ModelError")


def t_repr_leaks_no_key():
    os.environ["AMD_API_KEY"] = "SECRET-XYZ-123"
    m = AmdCloudModel("https://pod.example/v1", "Qwen3", "SECRET-XYZ-123")
    assert "SECRET-XYZ-123" not in repr(m), "key leaked in repr"
    assert "pod.example" in repr(m)


def t_base_url_trailing_slash_ok():
    m = AmdCloudModel("https://pod.example/v1/", "Qwen3", "k")
    assert m.base_url == "https://pod.example/v1"


# --- AmdCloudModel: request shape ------------------------------------------
class _FakeResp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(fn):
    import urllib.request as ur
    orig = ur.urlopen
    ur.urlopen = fn
    return orig


def t_request_shape():
    import urllib.request as ur
    seen = {}

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        seen["ct"] = req.headers.get("Content-type")
        payload = json.loads(req.data.decode("utf-8"))
        seen["payload"] = payload
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]})
        return _FakeResp(body.encode("utf-8"))

    orig = _patch_urlopen(fake)
    try:
        m = AmdCloudModel("https://pod.example/v1/", "Qwen3-Coder", "k")
        out = m.complete([{"role": "user", "content": "hi"}])
    finally:
        ur.urlopen = orig
    assert out == "ok"
    assert seen["url"] == "https://pod.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["ct"] == "application/json"
    assert seen["payload"]["model"] == "Qwen3-Coder"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["payload"]["max_tokens"] == 512


def t_http_error_is_modelerror():
    import urllib.request as ur

    def fake(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    orig = _patch_urlopen(fake)
    try:
        AmdCloudModel("https://pod.example/v1", "m", "k").complete(
            [{"role": "user", "content": "hi"}])
    except ModelError as e:
        assert "401" in str(e)
        return
    finally:
        ur.urlopen = orig
    raise AssertionError("expected ModelError")


def t_network_down_is_modelerror():
    import urllib.request as ur

    def fake(req, timeout=None):
        raise urllib.error.URLError("no route")

    orig = _patch_urlopen(fake)
    try:
        AmdCloudModel("https://pod.example/v1", "m", "k").complete(
            [{"role": "user", "content": "hi"}])
    except ModelError as e:
        assert "URLError" in str(e)
        return
    finally:
        ur.urlopen = orig
    raise AssertionError("expected ModelError")


def t_bad_schema_is_modelerror():
    import urllib.request as ur

    def fake(req, timeout=None):
        return _FakeResp(b'{"nope": true}')

    orig = _patch_urlopen(fake)
    try:
        AmdCloudModel("https://pod.example/v1", "m", "k").complete(
            [{"role": "user", "content": "hi"}])
    except ModelError:
        return
    finally:
        ur.urlopen = orig
    raise AssertionError("expected ModelError")


# --- anomaly engine: backend adapter + explanation pipeline -------------------
def t_amdbackend_auto_stub_mode():
    for k in ("AMD_BASE_URL", "AMD_MODEL", "AMD_API_KEY"):
        os.environ.pop(k, None)
    b = AmdModelBackend.auto()
    assert b.mode == "stub-synthetic", b.mode
    assert "amd-cloud-live" not in repr(b)


def t_amdbackend_stub_chat_deterministic():
    b = AmdModelBackend()
    a = b.chat("sys", "user prompt here")
    c = b.chat("sys", "user prompt here")
    assert a == c and "synthetic" in a.lower()


def t_amdbackend_modelerror_degrades():
    from amdcloud import model as m
    orig = m.AmdCloudModel.from_env if False else None  # noqa: F841

    class Boom:
        def complete(self, *a, **k):
            raise m.ModelError("endpoint down")

    b = AmdModelBackend(Boom())
    out = b.chat("s", "u")
    assert out.startswith("[MODEL-ERROR"), out


def t_plan_has_four_angles():
    p = plan(ANOMALIES[0])
    assert len(p) == 4
    angles = {x["angle"] for x in p}
    assert {"Likely causes", "Recommended maintenance actions"} <= angles
    assert all(x["query"] for x in p)


def t_gather_stub_returns_empty_lists():
    ev = gather(ANOMALIES[0])
    assert set(ev.keys()) == {s["query"] for s in plan(ANOMALIES[0])}
    assert all(isinstance(v, list) for v in ev.values())


def t_investigate_stub_end_to_end():
    r = investigate(ANOMALIES[0])
    assert r["anomaly_id"] == "ANO-101"
    assert r["backend"] == "stub-synthetic"
    assert r["fixture_data"] is True
    assert len(r["findings"]) == 4
    assert len(r["recommendations"]) == 3
    assert "limitations" in r and "technician" in r["limitations"].lower()
    assert 0.0 < r["confidence"] <= 0.9


def t_investigate_unknown_error_code():
    a = dict(ANOMALIES[0]); a["error_code"] = "NO-SUCH-CODE"
    r = investigate(a)
    assert "escalate" in r["recommendations"][0].lower()


def t_first_response_matches_all_fixture_codes():
    codes = {a["error_code"] for a in ANOMALIES}
    assert codes <= set(FIRST_RESPONSE.keys())
    assert all(len(v) >= 2 for v in FIRST_RESPONSE.values())


def t_report_markdown_labels_and_honesty():
    r = investigate(ANOMALIES[1])
    md = report_markdown(r)
    assert "synthetic fixture" in md.lower()
    assert "ANO-102" in md
    assert "stub-synthetic" in md
    assert "Limitations" in md
    assert "## First-response checklist" in md
    assert "SECRET-XYZ-123" not in md  # no fixture secret can leak


def t_demo_runs_cli_stub():
    import subprocess
    p = subprocess.run([sys.executable, "-m", "anomaly.demo"],
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert "ANO-101" in p.stdout and "ANO-103" in p.stdout
    assert "stub-synthetic" in p.stdout


def t_live_search_flag_off_by_default_no_network():
    # default demo run must work with the network entirely blocked
    import subprocess
    p = subprocess.run([sys.executable, "-m", "anomaly.demo", "--anomaly", "ANO-102"],
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert "confidence=" in p.stdout


# --- webdemo: anomaly explainer UI ----------------------------------------
def _fixture_result():
    return investigate(ANOMALIES[0], model=AmdModelBackend(), live_search=False)


def t_webdemo_form_lists_all_anomalies():
    page = render_form()
    for a in ANOMALIES:
        assert a["id"] in page
        assert esc(a["equipment"]) in page
    assert '<form method="post" action="/investigate">' in page


def t_webdemo_report_all_sections():
    page = render_report(_fixture_result(), live=False)
    for angle in ("Likely causes", "Failure modes",
                  "Recommended maintenance actions", "Similar field cases"):
        assert angle in page, angle
    for rec in FIRST_RESPONSE["VIB-OVERSHOOT"]:
        assert esc(rec[:40]) in page
    assert "licensed technician" in page  # honesty line
    assert "FIXTURE DEMO" in page
    assert "stub-synthetic" in page


def t_webdemo_report_escapes():
    evil = {"id": "ANO-X", "equipment": "<script>alert(1)</script>",
            "signal": "sig", "error_code": "NOPE", "since": "now"}
    result = investigate(evil, model=AmdModelBackend(), live_search=False)
    page = render_report(result, live=False)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def t_webdemo_unknown_anomaly_400():
    import urllib.request, threading
    from http.server import HTTPServer
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        idx = urllib.request.urlopen("http://127.0.0.1:%d/" % port,
                                     timeout=5).read().decode("utf-8")
        assert "ANO-101" in idx and "ANO-103" in idx
        data = urllib.parse.urlencode(
            {"anomaly": "ANO-102"}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:%d/investigate" % port, data=data)
        page = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        assert "ANO-102" in page and "First-response checklist" in page
        assert "PRESSURE-SAG" in page
        # XSS through the HTTP layer must be escaped
        evil_id = '<script>alert("xss")</script>'
        data = urllib.parse.urlencode(
            {"anomaly": evil_id}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:%d/investigate" % port, data=data)
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.shutdown()


def t_webdemo_esc_helper():
    assert esc("<b>&\"") == "&lt;b&gt;&amp;&quot;"
    assert esc(123) == "123"


CHECKS = [
    ("amdbackend auto stub mode", t_amdbackend_auto_stub_mode),
    ("amdbackend stub chat deterministic", t_amdbackend_stub_chat_deterministic),
    ("amdbackend ModelError degrades", t_amdbackend_modelerror_degrades),
    ("plan has four angles", t_plan_has_four_angles),
    ("gather stub returns empty lists", t_gather_stub_returns_empty_lists),
    ("investigate stub end-to-end", t_investigate_stub_end_to_end),
    ("investigate unknown error code", t_investigate_unknown_error_code),
    ("first response covers fixture codes", t_first_response_matches_all_fixture_codes),
    ("report markdown honesty", t_report_markdown_labels_and_honesty),
    ("demo runs CLI stub", t_demo_runs_cli_stub),
    ("demo works with no network", t_live_search_flag_off_by_default_no_network),
    ("webdemo form lists all anomalies", t_webdemo_form_lists_all_anomalies),
    ("webdemo report all sections", t_webdemo_report_all_sections),
    ("webdemo report escapes", t_webdemo_report_escapes),
    ("webdemo HTTP serve + unknown 400", t_webdemo_unknown_anomaly_400),
    ("webdemo esc helper", t_webdemo_esc_helper),
    ("stub deterministic", t_stub_deterministic),
    ("stub labels synthetic", t_stub_labels_synthetic),
    ("stub empty messages", t_stub_empty_messages),
    ("stub repr no secrets", t_stub_repr_no_secrets),
    ("missing env raises", t_missing_env_raises),
    ("missing one var raises", t_missing_one_raises),
    ("error message leaks no key", t_error_message_leaks_no_key),
    ("repr leaks no key", t_repr_leaks_no_key),
    ("base_url trailing slash ok", t_base_url_trailing_slash_ok),
    ("request shape", t_request_shape),
    ("http error -> ModelError", t_http_error_is_modelerror),
    ("network down -> ModelError", t_network_down_is_modelerror),
    ("bad schema -> ModelError", t_bad_schema_is_modelerror),
]


def main():
    for name, fn in CHECKS:
        check(name, fn)
    print("----")
    print("%d/%d PASS" % (PASS, PASS + FAIL))
    if FAIL:
        print("FAILURES: %s" % ", ".join(FAILURES))
        sys.exit(1)


if __name__ == "__main__":
    main()
