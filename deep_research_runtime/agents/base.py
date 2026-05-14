"""Shared context and utilities for all agent classes.

Uses LiteLLM for multi-provider model access. All LLM calls go through
litellm.acompletion() which auto-adapts to the configured provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..settings import Settings
from ..storage import TaskRegistryStore, now_iso


logger = logging.getLogger("DeepResearchMCP")


def dedupe_preserving_order(
    values: List[Any],
    *,
    limit: Optional[int] = None,
    collapse_whitespace: bool = False,
) -> List[str]:
    """Case-insensitive, order-preserving dedupe.

    Single replacement for the five near-identical helpers that previously
    lived in ``planner._dedupe_text_values`` / ``_dedupe_limited``,
    ``researcher._dedupe_text_values``, ``quality._dedupe_text_items`` and
    ``query_reform._dedupe_preserving_order``.

    Parameters
    ----------
    values:
        Iterable of items to clean and dedupe. Each is coerced to ``str``
        and stripped; empty results are dropped.
    limit:
        If set and > 0, the result is truncated to this many entries.
    collapse_whitespace:
        When ``True``, internal runs of whitespace are collapsed to a single
        space (matches the old ``quality._dedupe_text_items`` behavior).
    """
    seen: set = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if collapse_whitespace:
            text = re.sub(r"\s+", " ", text)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit and limit > 0 and len(out) >= limit:
            break
    return out


@dataclass
class AgentContext:
    """Dependency container injected into every agent.

    Instead of inheriting from multiple Mixin classes, each agent receives
    this context object in its constructor, making dependencies explicit
    and enabling isolated unit testing.
    """

    settings: Settings
    store: TaskRegistryStore

    # Worker interaction (embedding + reranking via subprocess)
    worker_caller: Callable[..., Awaitable[List[Dict[str, Any]]]]

    # Observability hooks
    log_task: Callable[..., None]
    record_timing: Callable[..., float]
    save_probe: Callable[..., str]


# ── Shared helper functions (previously scattered across Mixins) ──


def sanitize_path_name(value: str, fallback: str = "task") -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned[:80] or fallback


def infer_user_language(*texts: str) -> str:
    sample = " ".join(str(text or "") for text in texts)
    return "zh" if re.search(r"[\u4e00-\u9fff]", sample) else "en"


def language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"


def pick_text(language: str, zh_text: str, en_text: str) -> str:
    return zh_text if language == "zh" else en_text


def robust_json_parse(content: str) -> Any:
    if not content:
        return None
    candidate = content.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0]
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


# ── LiteLLM-based LLM calling ──


def _ensure_litellm_env(ctx: AgentContext) -> None:
    """Set LiteLLM environment variables from Settings if not already set.

    LiteLLM reads provider API keys from environment variables automatically.
    We bridge from our Settings to the env vars LiteLLM expects.
    """
    # DeepSeek: LiteLLM reads DEEPSEEK_API_KEY
    if ctx.settings.llm_api_key and not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = ctx.settings.llm_api_key


async def call_llm_text(
    ctx: AgentContext,
    prompt: str,
    *,
    role: str = "default",
    task_id: str = "",
    topic: str = "",
    stage: str = "llm",
    name: str = "call_llm_text",
) -> str:
    """Call LLM via LiteLLM and return text response.

    Args:
        role: Agent role for model selection (planner/researcher/writer/reviewer).
              Uses ctx.settings.get_model_for_role(role) to pick model.
    """
    import litellm

    _ensure_litellm_env(ctx)
    model = ctx.settings.get_model_for_role(role)
    started_at = time.perf_counter()

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_base=ctx.settings.llm_api_base if "deepseek" in model.lower() else None,
        )
        content = (response.choices[0].message.content or "").strip()
        if task_id:
            ctx.record_timing(task_id, topic, stage, name, started_at, prompt_chars=len(prompt), output_chars=len(content), llm_model=model)
        return content
    except Exception as exc:
        if task_id:
            ctx.record_timing(task_id, topic, stage, name, started_at, level="error", prompt_chars=len(prompt), error=str(exc), llm_model=model)
        raise


async def call_llm_json(
    ctx: AgentContext,
    prompt: str,
    *,
    role: str = "default",
    task_id: str = "",
    topic: str = "",
    stage: str = "llm",
    name: str = "call_llm_json",
) -> Any:
    """Call LLM via LiteLLM with JSON response format.

    Args:
        role: Agent role for model selection.
    """
    import litellm

    _ensure_litellm_env(ctx)
    model = ctx.settings.get_model_for_role(role)
    started_at = time.perf_counter()

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            api_base=ctx.settings.llm_api_base if "deepseek" in model.lower() else None,
        )
        content = response.choices[0].message.content or ""
        parsed = robust_json_parse(content)
        if task_id:
            ctx.record_timing(task_id, topic, stage, name, started_at, prompt_chars=len(prompt), output_chars=len(content), llm_model=model)
        return parsed
    except Exception as exc:
        if task_id:
            ctx.record_timing(task_id, topic, stage, name, started_at, level="error", prompt_chars=len(prompt), error=str(exc), llm_model=model)
        raise
