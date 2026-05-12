"""DeepResearchService: the main orchestrator using explicit agents for research logic.

All research logic is delegated to explicit Agent classes (planner, researcher, reviewer, writer).
ToolsMixin keeps the external MCP tool surface and RuntimeMixin manages low-level process lifecycle.
LLM access is via LiteLLM (no more AsyncOpenAI client management here).
"""

import asyncio
import logging
import multiprocessing as mp
import os
import queue
from typing import Any, Dict, Optional

import aiosqlite
from langsmith import Client
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .agents.base import AgentContext
from .agents.planner import PlannerAgent
from .agents.researcher import ResearcherAgent
from .agents.reviewer import ReviewerAgent
from .agents.writer import WriterAgent
from .graph import build_graph
from .runtime import RuntimeMixin
from .search_service import SearchService
from .settings import Settings
from .storage import TaskRegistryStore
from .tools import ToolsMixin


class DeepResearchService(ToolsMixin, RuntimeMixin):
    """Orchestrates the deep research workflow.

    Key architectural change: Previously inherited from 7 Mixins.
    Now uses composition with 4 Agent classes injected via AgentContext.
    Only ToolsMixin (external API surface) and RuntimeMixin (process lifecycle)
    remain as inherited classes.

    LLM access: Delegated to LiteLLM via agents/base.py. No direct OpenAI
    client management — LiteLLM handles provider routing automatically.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._configure_langsmith_env()
        self.store = TaskRegistryStore(self.settings)

        # ── Worker State ──
        self._langsmith_client: Optional[Client] = self._build_langsmith_client()
        self._mp_context = mp.get_context("spawn")
        self._model_worker_process: Optional[mp.Process] = None
        self._worker_job_queue: Optional[mp.Queue] = None
        self._worker_result_queue: Optional[mp.Queue] = None
        self._worker_manager: Optional[Any] = None
        self._worker_state: Optional[Any] = None
        self._worker_result_buffer: Dict[str, Dict[str, Any]] = {}
        self._worker_queue_empty = queue.Empty
        self._model_worker_ready = False
        self._model_worker_error: Optional[str] = None
        # Sliding-window restart history (timestamps of recent restarts). The
        # runtime layer consults this to avoid an infinite restart loop when
        # the worker is fundamentally broken (e.g. corrupt model weights).
        self._worker_restart_history: list = []
        self._task_handlers: Dict[str, logging.Handler] = {}
        self._task_loggers: Dict[str, logging.Logger] = {}
        self._active_background_tasks: Dict[str, asyncio.Task] = {}
        self._worker_start_lock = asyncio.Lock()
        self._worker_result_lock = asyncio.Lock()
        self._model_worker_log_path: Optional[str] = None
        self._graph_init_lock = asyncio.Lock()

        # ── Filesystem ──
        self.ensure_report_dir()
        self._graph_checkpoint_path = self.get_graph_checkpoint_path()
        self._graph_checkpoint_connection: Optional[aiosqlite.Connection] = None
        self.graph_checkpointer: Optional[AsyncSqliteSaver] = None
        self.app_engine = None
        self.search_service = SearchService(self.settings)

        # ── Agent Composition ──
        # LiteLLM handles LLM calls directly — no client factory needed.
        self._agent_context = AgentContext(
            settings=self.settings,
            store=self.store,
            worker_caller=self.call_model_worker,
            log_task=self.log_task,
            record_timing=self.record_timing,
            save_probe=self.save_probe,
        )
        self.planner = PlannerAgent(self._agent_context)
        self.researcher = ResearcherAgent(self._agent_context)
        self.reviewer = ReviewerAgent(self._agent_context)
        self.writer = WriterAgent(self._agent_context)

    # ── LangSmith ──

    def _configure_langsmith_env(self) -> None:
        """Propagate LangSmith config to env vars consumed by langsmith/@traceable.

        The langsmith client library reads LANGSMITH_API_KEY / LANGCHAIN_PROJECT
        from os.environ at decoration time, so we must set them here. To avoid
        clobbering values the user explicitly set in their MCP client config,
        we only fill in keys that are NOT already present in the environment.
        """
        if not self.settings.langsmith_api_key:
            return
        env_defaults = {
            "LANGSMITH_API_KEY": self.settings.langsmith_api_key,
            "LANGCHAIN_PROJECT": self.settings.langsmith_project,
            "LANGSMITH_ENDPOINT": self.settings.langsmith_endpoint,
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_TRACING_V2": "true",
        }
        for key, value in env_defaults.items():
            os.environ.setdefault(key, value)
        self.settings.langsmith_tracing = True

    def _build_langsmith_client(self) -> Optional[Client]:
        if not self.settings.langsmith_api_key:
            return None
        return Client(
            api_key=self.settings.langsmith_api_key,
            api_url=self.settings.langsmith_endpoint,
        )

    # ── Graph Lifecycle ──

    async def ensure_graph_ready(self) -> None:
        if self.app_engine is not None and self.graph_checkpointer is not None:
            return
        async with self._graph_init_lock:
            if self.app_engine is not None and self.graph_checkpointer is not None:
                return
            if self._graph_checkpoint_connection is None:
                self._graph_checkpoint_connection = await aiosqlite.connect(str(self._graph_checkpoint_path))
            self.graph_checkpointer = AsyncSqliteSaver(self._graph_checkpoint_connection)
            self.app_engine = build_graph(self)

    # ── Shutdown ──

    async def shutdown(self) -> None:
        for task in list(self._active_background_tasks.values()):
            if task.done():
                continue
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        self._active_background_tasks.clear()

        self.stop_model_backend()

        self.graph_checkpointer = None
        self.app_engine = None
        if self._graph_checkpoint_connection is not None:
            try:
                await self._graph_checkpoint_connection.close()
            except Exception:
                pass
            self._graph_checkpoint_connection = None

        if getattr(self, "store", None) is not None:
            try:
                await self.store.close()
            except Exception:
                pass
