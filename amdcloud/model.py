"""amdcloud/model.py — model-backend seam for AMD Developer Cloud.

The AMD Developer Hackathon: ACT III requires a meaningful part of the
workload to run on AMD infrastructure. The official docs point at vLLM
endpoints on AMD Developer Cloud (OpenAI-compatible). This module is that
seam:

- AmdCloudModel: stdlib-only OpenAI-compatible /chat/completions client.
  Configured ONLY via environment (AMD_BASE_URL, AMD_MODEL, AMD_API_KEY).
  Never logs or raises the API key. Never crashes on network failure —
  returns an honest ModelError and lets the agent degrade explicitly.
- StubAmd: deterministic local stand-in for tests and for pre-event builds
  (fixture answers, clearly labeled synthetic).

Wiring note: DeepDive/WhyAgent model backends plug in here — swap their
StubModel for AmdCloudModel.from_env() once the AMD endpoint is live (his
AMD AI Developer Program signup).
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error


class ModelError(Exception):
    """Raised when the remote model cannot be reached or misbehaves."""
    pass


class StubAmd:
    """Deterministic local stand-in. Answers are synthetic fixtures."""

    def complete(self, messages, max_tokens=512):
        last = messages[-1].get("content", "") if messages else ""
        return ("[STUB-AMD synthetic] acknowledged prompt (%d chars). "
                "Real AMD Developer Cloud endpoint not configured."
                % len(last))

    def __repr__(self):
        return "StubAmd(synthetic-fixtures)"


class AmdCloudModel:
    """OpenAI-compatible chat client for a vLLM endpoint on AMD Developer Cloud.

    Config comes only from the environment:
      AMD_BASE_URL  e.g. https://<pod>.amdcloud.example/v1  (no trailing path
                    beyond /v1; the client appends /chat/completions)
      AMD_MODEL     model id served by the endpoint
      AMD_API_KEY   bearer key — NEVER logged, never echoed in exceptions
    """

    def __init__(self, base_url, model, api_key, timeout_s=60):
        if not base_url or not model or not api_key:
            raise ModelError("AmdCloudModel requires AMD_BASE_URL, AMD_MODEL "
                             "and AMD_API_KEY (values redacted in this "
                             "message by design).")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls):
        return cls(
            os.environ.get("AMD_BASE_URL", ""),
            os.environ.get("AMD_MODEL", ""),
            os.environ.get("AMD_API_KEY", ""),
        )

    def complete(self, messages, max_tokens=512):
        url = self.base_url + "/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": m.get("role", "user"),
                          "content": m.get("content", "")} for m in messages],
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ModelError("AMD endpoint HTTP %d" % e.code)
        except urllib.error.URLError as e:
            raise ModelError("AMD endpoint unreachable: %s"
                             % type(e).__name__)
        except (ValueError, KeyError):
            raise ModelError("AMD endpoint returned non-JSON response")
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError("AMD endpoint returned unexpected schema")

    def __repr__(self):
        # NEVER include the key.
        return "AmdCloudModel(base_url=%r, model=%r)" % (self.base_url,
                                                        self.model)
