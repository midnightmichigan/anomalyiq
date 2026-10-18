"""anomaly/backend.py — AMD Developer Cloud model backend, DeepDive-shaped.

DeepDive's engine speaks ``chat(system, user) -> str``; AmdCloudModel speaks
``complete(messages) -> str``. This module is the adapter between them, so
the explanation engine's model calls run on AMD infrastructure — the
mandatory rule for prize eligibility in AMD ACT III.

- AmdModelBackend.auto(): builds AmdCloudModel from the environment when
  AMD_BASE_URL / AMD_MODEL / AMD_API_KEY are set, else falls back to the
  deterministic StubAmd stand-in. ``mode`` honestly reports which path is
  active (``amd-cloud-live`` vs ``stub-synthetic``).
- On any ModelError the chat() call degrades to an explicit marker string
  instead of raising — the explain engine detects the marker and switches
  to its honest extractive fallback.

Stdlib only. The API key never appears in reprs or error strings.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from amdcloud.model import AmdCloudModel, StubAmd, ModelError  # noqa: E402


class AmdModelBackend:
    """DeepDive-shaped model backend (chat(system, user)) on AMD infra."""

    def __init__(self, model=None):
        self._model = model if model is not None else StubAmd()

    @classmethod
    def auto(cls):
        """Use the AMD Developer Cloud endpoint when configured, else the stub."""
        try:
            return cls(AmdCloudModel.from_env())
        except ModelError:
            return cls(StubAmd())

    @property
    def mode(self) -> str:
        return "amd-cloud-live" if isinstance(self._model, AmdCloudModel) else "stub-synthetic"

    def chat(self, system: str, user: str) -> str:
        try:
            return self._model.complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=768)
        except ModelError:
            return ("[MODEL-ERROR: AMD Developer Cloud endpoint unreachable — "
                    "using extractive evidence fallback.]")

    def __repr__(self):
        return "AmdModelBackend(mode=%s)" % self.mode
