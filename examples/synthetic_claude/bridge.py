"""File-based LLM bridge: lets an agent session answer maricl_core's LLM calls.

Each structured-output call gets a sequential id. If <root>/responses/<id>.json exists it is
validated against the requested Pydantic schema and returned; otherwise the request (system
prompt, user prompt, JSON schema) is written to <root>/requests/<id>.json and NeedResponse is
raised, which stops the run. Re-running replays all answered calls deterministically up to the
next unanswered one; a SHA-256 of every prompt guards against non-deterministic replay.
"""
from __future__ import annotations

import hashlib
import json
import os

from claude_backend import SystemMessage


class NeedResponse(BaseException):
    """Raised when the next LLM call has no recorded answer.

    Subclasses BaseException so maricl_core's `except Exception` retry blocks do not swallow it.
    """

    def __init__(self, request: dict):
        super().__init__(request["id"])
        self.request = request


class BridgeLLM:
    def __init__(self, root: str, role: str, counter: dict):
        self.root, self.role, self.counter = root, role, counter
        os.makedirs(os.path.join(root, "requests"), exist_ok=True)
        os.makedirs(os.path.join(root, "responses"), exist_ok=True)

    def with_structured_output(self, schema):
        return _BridgeRunnable(self, schema)


class _BridgeRunnable:
    def __init__(self, bridge: BridgeLLM, schema):
        self.bridge, self.schema = bridge, schema

    def invoke(self, messages):
        b = self.bridge
        rid = f"{b.counter['n']:04d}"
        b.counter["n"] += 1
        system = "\n\n".join(m.content for m in messages if isinstance(m, SystemMessage))
        user = "\n\n".join(m.content for m in messages if not isinstance(m, SystemMessage))
        digest = hashlib.sha256(f"{self.schema.__name__}\n{system}\n{user}".encode()).hexdigest()
        request = {"id": rid, "schema": self.schema.__name__, "role": b.role,
                   "json_schema": self.schema.model_json_schema(),
                   "system": system, "user": user, "sha256": digest}
        req_path = os.path.join(b.root, "requests", f"{rid}.json")
        resp_path = os.path.join(b.root, "responses", f"{rid}.json")
        if os.path.exists(req_path):
            if json.load(open(req_path))["sha256"] != digest:
                raise RuntimeError(f"non-deterministic replay at request {rid}")
        else:
            with open(req_path, "w") as fh:
                json.dump(request, fh, indent=2)
        if os.path.exists(resp_path):
            return self.schema.model_validate(json.load(open(resp_path)))
        raise NeedResponse(request)
