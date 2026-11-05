"""Claude backend for the released MARICL core (maricl_core.py).

maricl_core expects LangChain-style LLM objects:
    llm.with_structured_output(Schema).invoke([SystemMessage(...), HumanMessage(...)]) -> Schema
This module provides that interface on top of the official Anthropic SDK
(client.messages.parse with a Pydantic output format), so maricl_core.py runs unmodified.

Every call is appended to a JSONL log (role, schema, tokens, latency) for cost accounting.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from typing import Any, List

import anthropic
from pydantic import BaseModel


# ---- message shims (stand-ins for langchain_core.messages, which maricl_core imports) ----
class _Message:
    def __init__(self, content: str):
        self.content = content


class SystemMessage(_Message):
    pass


class HumanMessage(_Message):
    pass


def install_import_shims() -> None:
    """Let `import maricl_core` succeed without LangChain/OpenAI packages installed.

    maricl_core imports ChatOpenAI only as a type annotation and uses SystemMessage /
    HumanMessage only as containers with a `.content` attribute.
    """
    # maricl_core.py lives at the repository root, two levels above this example.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    lc_openai = types.ModuleType("langchain_openai")
    lc_openai.ChatOpenAI = object
    lc_core = types.ModuleType("langchain_core")
    lc_msgs = types.ModuleType("langchain_core.messages")
    lc_msgs.SystemMessage = SystemMessage
    lc_msgs.HumanMessage = HumanMessage
    lc_core.messages = lc_msgs
    sys.modules.setdefault("langchain_openai", lc_openai)
    sys.modules.setdefault("langchain_core", lc_core)
    sys.modules.setdefault("langchain_core.messages", lc_msgs)
    # maricl_core allow-lists ast.Num, a deprecated alias of ast.Constant removed in Python 3.14.
    import ast
    if not hasattr(ast, "Num"):
        ast.Num = ast.Constant


class _CallLog:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self.lock, open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


class _StructuredRunnable:
    def __init__(self, client: anthropic.Anthropic, model: str, effort: str,
                 schema: type[BaseModel], role: str, log: _CallLog, max_tokens: int):
        self.client, self.model, self.effort = client, model, effort
        self.schema, self.role, self.log, self.max_tokens = schema, role, log, max_tokens

    def invoke(self, messages: List[Any]) -> BaseModel:
        system = "\n\n".join(m.content for m in messages if isinstance(m, SystemMessage))
        user = "\n\n".join(m.content for m in messages if not isinstance(m, SystemMessage))
        t0 = time.time()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=self.schema,
            output_config={"effort": self.effort},
        )
        usage = response.usage
        self.log.write({
            "role": self.role, "schema": self.schema.__name__, "model": response.model,
            "stop_reason": response.stop_reason, "seconds": round(time.time() - t0, 2),
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        })
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the {self.schema.__name__} request")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{self.schema.__name__} response hit max_tokens")
        if response.parsed_output is None:
            raise RuntimeError(f"no parsed {self.schema.__name__} in response")
        return response.parsed_output


class ClaudeLLM:
    """Drop-in for the `llm` / `llm_creative` objects that maricl_core.configure_llms expects."""

    def __init__(self, model: str, effort: str, log_path: str, role: str,
                 max_tokens: int = 16000, max_retries: int = 6):
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model, self.effort, self.role, self.max_tokens = model, effort, role, max_tokens
        self.log = _CallLog(log_path)

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredRunnable:
        return _StructuredRunnable(self.client, self.model, self.effort, schema,
                                   self.role, self.log, self.max_tokens)
