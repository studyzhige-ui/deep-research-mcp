import asyncio
import json
import logging
import os
import queue
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import httpx

from .agents.base import robust_json_parse
from .langsmith_utils import trace_worker
from .storage import now_iso


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
os.environ["TQDM_DISABLE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
for noisy_logger in ("httpx", "httpcore", "openai", "faiss"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger("DeepResearchMCP")


class RuntimeMixin:
    def classify_error(self, exc: Exception) -> str:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return "timeout"
        if isinstance(exc, aiohttp.ClientError):
            return "network"
        if isinstance(exc, httpx.HTTPError):
            return "network"
        if isinstance(exc, RuntimeError):
            message = str(exc).lower()
            if "redis" in message:
                return "redis"
            if "worker" in message or "model" in message:
                return "worker"
        return "internal"

    def ensure_report_dir(self) -> Path:
        path = Path(self.settings.report_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _sanitize_path_name(value: str, fallback: str = "task") -> str:
        cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", value or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
        return cleaned[:80] or fallback

    @staticmethod
    def infer_user_language(*texts: str) -> str:
        sample = " ".join(str(text or "") for text in texts)
        return "zh" if re.search(r"[\u4e00-\u9fff]", sample) else "en"

    @staticmethod
    def language_name(language: str) -> str:
        return "Chinese" if language == "zh" else "English"

    @staticmethod
    def pick_text(language: str, zh_text: str, en_text: str) -> str:
        return zh_text if language == "zh" else en_text

    def get_task_output_dir(self, task_id: str, topic: str = "") -> Path:
        root = self.ensure_report_dir()
        topic_part = self._sanitize_path_name(topic, fallback="task")
        task_dir = root / f"{topic_part}_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def get_task_trace_path(self, task_id: str, topic: str = "") -> Path:
        task_dir = self.get_task_output_dir(task_id, topic)
        return task_dir / f"DeepResearch_Trace_{task_id}.jsonl"

    def get_worker_runtime_dir(self) -> Path:
        runtime_dir = self.ensure_report_dir() / "_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir

    def get_worker_bootstrap_log_path(self) -> Path:
        runtime_dir = self.get_worker_runtime_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return runtime_dir / f"DeepResearch_WorkerBootstrap_{timestamp}.log"

    def get_graph_checkpoint_path(self) -> Path:
        runtime_dir = self.get_worker_runtime_dir()
        return runtime_dir / "DeepResearch_GraphCheckpoints.sqlite"

    def prepare_startup(self) -> None:
        self.ensure_report_dir()
        # Print a one-shot config self-check so users see immediately which engines
        # are active and which are skipped due to missing API keys. Blocking errors
        # are still allowed to fall through to start_model_backend, which decides
        # whether the failure is fatal.
        try:
            self.settings.validate_and_report(emit=True)
        except Exception:
            # Validation must never prevent startup — it's purely informational.
            pass
        if not self.start_model_backend(stage="startup"):
            raise RuntimeError(self._model_worker_error or "Model worker failed to start during MCP startup.")

    def setup_task_logger(self, task_id: str, topic: str = "") -> str:
        task_dir = self.get_task_output_dir(task_id, topic)
        if task_id in self._task_handlers and task_id in self._task_loggers:
            return str(task_dir / f"DeepResearch_Process_{task_id}.log")
        log_path = task_dir / f"DeepResearch_Process_{task_id}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        task_logger = logging.getLogger(f"DeepResearchMCP.task.{task_id}")
        task_logger.setLevel(logging.INFO)
        task_logger.propagate = False
        task_logger.handlers.clear()
        task_logger.addHandler(handler)
        self._task_handlers[task_id] = handler
        self._task_loggers[task_id] = task_logger
        return str(log_path)

    def log_task(self, task_id: str, message: str, level: str = "info", **context: Any) -> None:
        task_logger = self._task_loggers.get(task_id)
        suffix = f" | {json.dumps(context, ensure_ascii=False)}" if context else ""
        rendered = f"{message}{suffix}"
        target = level if level in {"debug", "info", "warning", "error"} else "info"
        if task_logger:
            getattr(task_logger, target)(rendered)
        getattr(logger, target)(f"[task={task_id}] {rendered}")

    def save_probe(self, task_id: str, topic: str, stage: str, name: str, payload: Any) -> str:
        trace_path = self.get_task_trace_path(task_id, topic)
        if not getattr(self.settings, "debug_trace", False):
            return str(trace_path)
        entry = {
            "timestamp": now_iso(),
            "task_id": task_id,
            "stage": stage,
            "name": name,
            "payload": payload,
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return str(trace_path)

    def record_timing(
        self,
        task_id: str,
        topic: str,
        stage: str,
        name: str,
        started_at: float,
        level: str = "info",
        **extra: Any,
    ) -> float:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        payload = {"elapsed_ms": elapsed_ms, **extra}
        self.log_task(task_id, f"Timing: {name}", level=level, stage=stage, **payload)
        if getattr(self.settings, "debug_trace", False):
            self.save_probe(task_id, topic, stage, f"{name}_timing", payload)
        return elapsed_ms

    async def validate_runtime_dependencies(self) -> List[str]:
        issues = []
        if not self.settings.llm_api_key:
            issues.append("Missing DEEP_RESEARCH_LLM_API_KEY.")
        if not Path(self.settings.embedder_path).exists():
            issues.append(f"Embedding model path does not exist: {self.settings.embedder_path}")
        if not Path(self.settings.reranker_path).exists():
            issues.append(f"Reranker model path does not exist: {self.settings.reranker_path}")
        if not await self.store.ping():
            issues.append("Task registry is unavailable.")
        return issues

    async def get_worker_health(self) -> Dict[str, Any]:
        status = ""
        current_job = ""
        raw_heartbeat = ""
        if self._worker_state is not None:
            try:
                status = str(self._worker_state.get("status") or "")
                current_job = str(self._worker_state.get("job_id") or "")
                raw_heartbeat = str(self._worker_state.get("heartbeat") or "")
            except Exception:
                status = ""
                current_job = ""
                raw_heartbeat = ""

        heartbeat_age = None
        stale = False
        if raw_heartbeat:
            try:
                heartbeat_age = max(0.0, time.time() - float(raw_heartbeat))
                stale = heartbeat_age > self.settings.worker_stale_after
            except Exception:
                heartbeat_age = None
                stale = True
        else:
            stale = bool(self._model_worker_process and self._model_worker_process.is_alive())

        return {
            "status": status or "UNKNOWN",
            "current_job": current_job or "",
            "heartbeat_age_sec": heartbeat_age,
            "stale": stale,
            "process_alive": bool(self._model_worker_process and self._model_worker_process.is_alive()),
        }

    async def runtime_summary(self) -> Dict[str, Any]:
        issues = await self.validate_runtime_dependencies()
        worker = await self.get_worker_health()
        worker_state = "ready" if self._model_worker_ready else str(worker["status"] or "unavailable").lower()
        return {
            "healthy": not issues,
            "issues": issues,
            "worker_ready": self._model_worker_ready,
            "worker_state": worker_state,
            "worker_error": self._model_worker_error,
            "active_task_count": len(self._active_background_tasks),
            "report_dir": self.settings.report_dir,
            "search_backend": "tavily" if self.settings.tavily_api_key else "duckduckgo+readability",
            "llm_model": self.settings.llm_model,
        }

    def stop_model_backend(self) -> None:
        process = self._model_worker_process
        if not process:
            self._model_worker_ready = False
            self._model_worker_log_path = None
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        self._model_worker_process = None
        if self._worker_job_queue is not None:
            try:
                self._worker_job_queue.close()
                self._worker_job_queue.join_thread()
            except Exception:
                pass
        if self._worker_result_queue is not None:
            try:
                self._worker_result_queue.close()
                self._worker_result_queue.join_thread()
            except Exception:
                pass
        if self._worker_manager is not None:
            try:
                self._worker_manager.shutdown()
            except Exception:
                pass
        self._worker_job_queue = None
        self._worker_result_queue = None
        self._worker_manager = None
        self._worker_state = None
        self._worker_result_buffer.clear()
        self._model_worker_ready = False
        self._model_worker_log_path = None

    def _check_restart_budget(self, task_id: str = "", stage: str = "runtime") -> bool:
        """Return True if a worker restart is permitted by the cooldown policy.

        We keep a sliding window of recent restart timestamps. If we've already
        restarted ``worker_max_restart_attempts`` times within
        ``worker_restart_window_sec``, we refuse to try again until older
        entries fall outside the window. This prevents an infinite restart
        loop when the worker is fundamentally broken (e.g. model file gone)
        which would otherwise burn CPU and spam logs.
        """
        window = max(1, int(self.settings.worker_restart_window_sec))
        max_attempts = max(1, int(self.settings.worker_max_restart_attempts))
        now = time.time()
        history = getattr(self, "_worker_restart_history", None)
        if history is None:
            self._worker_restart_history = []
            history = self._worker_restart_history
        # Drop entries outside the window.
        cutoff = now - window
        history[:] = [ts for ts in history if ts >= cutoff]
        if len(history) >= max_attempts:
            wait_for = max(1, int(history[0] + window - now))
            self._model_worker_error = (
                f"worker restart budget exhausted ({len(history)} restarts in last {window}s); "
                f"refusing to restart for another ~{wait_for}s"
            )
            if task_id:
                self.log_task(
                    task_id,
                    "Worker restart budget exhausted.",
                    level="error",
                    stage=stage,
                    restart_count=len(history),
                    window_sec=window,
                )
            return False
        history.append(now)
        return True

    def start_model_backend(self, task_id: str = "", topic: str = "", stage: str = "runtime") -> bool:
        startup_started_at = time.perf_counter()
        if self._model_worker_process and self._model_worker_process.is_alive() and self._model_worker_ready:
            if task_id:
                self.log_task(task_id, "Worker startup skipped because backend is already ready.", stage=stage)
            return True

        # Gate any actual launch behind the sliding-window restart budget.
        if not self._check_restart_budget(task_id=task_id, stage=stage):
            return False

        logger.info("Starting model worker process...")
        if task_id:
            self.log_task(task_id, "Starting model worker process.", stage=stage)

        try:
            from .worker import model_worker_process

            if self._model_worker_process and self._model_worker_process.is_alive() and not self._model_worker_ready:
                if task_id:
                    self.log_task(task_id, "Existing worker process is alive but not ready; restarting before launch.", level="warning", stage=stage, worker_pid=self._model_worker_process.pid)
                self.stop_model_backend()
            elif self._model_worker_process and not self._model_worker_process.is_alive():
                self._model_worker_process = None

            worker_log_path = self.get_worker_bootstrap_log_path()
            self._model_worker_log_path = str(worker_log_path)
            self._worker_manager = self._mp_context.Manager()
            # Bound the inbound job queue so a runaway producer cannot OOM
            # the host. ``maxsize=0`` means unbounded (legacy behavior); any
            # positive value applies backpressure: ``Queue.put`` blocks once
            # the queue is full, naturally throttling submitters. We do the
            # actual put on a worker thread (``asyncio.to_thread``) so the
            # event loop is never blocked by this.
            job_queue_max = max(0, int(self.settings.worker_job_queue_max_size))
            self._worker_job_queue = self._mp_context.Queue(maxsize=job_queue_max)
            self._worker_result_queue = self._mp_context.Queue()
            self._worker_state = self._worker_manager.dict(
                {"status": "STARTING", "job_id": "", "heartbeat": 0.0, "error": ""}
            )
            self._model_worker_process = self._mp_context.Process(
                target=model_worker_process,
                args=(
                    self.settings.as_dict(),
                    self._worker_job_queue,
                    self._worker_result_queue,
                    self._worker_state,
                    str(worker_log_path),
                ),
                daemon=True,
            )
            self._model_worker_process.start()

            if task_id:
                self.log_task(
                    task_id,
                    "Worker process launched.",
                    stage=stage,
                    worker_pid=self._model_worker_process.pid,
                    worker_log_path=str(worker_log_path),
                )

            last_status = ""
            for _ in range(self.settings.worker_start_timeout):
                status = str(self._worker_state.get("status") or "") if self._worker_state is not None else ""
                if task_id and status and status != last_status:
                    last_status = status
                    self.log_task(task_id, "Worker startup status changed.", stage=stage, worker_status=status)
                if status == "READY":
                    self._model_worker_ready = True
                    self._model_worker_error = None
                    if task_id:
                        self.log_task(task_id, "Worker became ready.", stage=stage, elapsed_ms=round((time.perf_counter() - startup_started_at) * 1000, 2))
                    return True
                if status and status.startswith("ERROR:"):
                    self._model_worker_ready = False
                    self._model_worker_error = status
                    self.stop_model_backend()
                    if task_id:
                        self.log_task(task_id, "Worker startup failed.", level="error", stage=stage, worker_status=status, elapsed_ms=round((time.perf_counter() - startup_started_at) * 1000, 2), worker_log_path=self._model_worker_log_path or "")
                    return False
                time.sleep(1)
        except Exception as exc:
            self._model_worker_error = str(exc)
            self._model_worker_ready = False
            self.stop_model_backend()
            if task_id:
                self.log_task(task_id, "Worker startup crashed.", level="error", stage=stage, error=str(exc), elapsed_ms=round((time.perf_counter() - startup_started_at) * 1000, 2), worker_log_path=self._model_worker_log_path or "")
            return False

        self._model_worker_error = "Model worker startup timed out."
        self._model_worker_ready = False
        self.stop_model_backend()
        if task_id:
            self.log_task(task_id, "Worker startup timed out.", level="error", stage=stage, worker_status=last_status or "UNKNOWN", elapsed_ms=round((time.perf_counter() - startup_started_at) * 1000, 2), worker_log_path=self._model_worker_log_path or "")
        return False

    def _drain_worker_result_queue(self) -> None:
        if self._worker_result_queue is None:
            return
        while True:
            try:
                item = self._worker_result_queue.get_nowait()
            except queue.Empty:
                return
            except Exception:
                return
            if isinstance(item, dict):
                job_id = str(item.get("job_id") or "").strip()
                if job_id:
                    self._worker_result_buffer[job_id] = item

    async def ensure_model_backend_ready(self, task_id: str = "", topic: str = "", stage: str = "runtime") -> bool:
        async with self._worker_start_lock:
            worker = await self.get_worker_health()
            if worker["process_alive"] and self._model_worker_ready and not worker["stale"]:
                if task_id:
                    self.log_task(task_id, "Worker already ready.", stage=stage, worker_status=worker["status"])
                return True
            if worker["stale"] and worker["process_alive"]:
                self._model_worker_error = "worker heartbeat stale"
                if task_id:
                    self.log_task(task_id, "Worker marked stale, restarting backend.", level="warning", stage=stage, worker_status=worker["status"])
                self.stop_model_backend()
            self._model_worker_ready = False
            return await asyncio.to_thread(self.start_model_backend, task_id, topic, stage)

    @trace_worker(name="call_model_worker")
    async def call_model_worker(
        self,
        queries: List[str],
        documents: List[Dict[str, Any]],
        *,
        task_id: str = "",
        topic: str = "",
        stage: str = "researcher",
    ) -> List[Dict[str, Any]]:
        if not await self.ensure_model_backend_ready(task_id=task_id, topic=topic, stage=stage):
            raise RuntimeError(self._model_worker_error or "worker unavailable")
        max_attempts = max(1, self.settings.worker_job_retry_attempts)
        query_preview = ", ".join(str(item) for item in queries[:2])[:300]
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            job_identifier = uuid.uuid4().hex
            if task_id:
                self.log_task(
                    task_id,
                    "Submitting worker job.",
                    stage=stage,
                    job_id=job_identifier,
                    attempt=attempt,
                    query_preview=query_preview,
                    document_count=len(documents),
                )

            if self._worker_job_queue is None:
                raise RuntimeError("worker queue is unavailable")
            await asyncio.to_thread(
                self._worker_job_queue.put,
                {"job_id": job_identifier, "queries": queries, "documents": documents},
            )

            waited = 0.0
            try:
                while waited < self.settings.worker_result_timeout:
                    async with self._worker_result_lock:
                        self._drain_worker_result_queue()
                        payload = self._worker_result_buffer.pop(job_identifier, None)

                    if payload:

                        if not isinstance(payload, dict):
                            raise RuntimeError("worker returned a malformed result payload")

                        status = str(payload.get("status") or "").strip().upper()
                        if status == "OK":
                            evidence = payload.get("evidence", [])
                            if task_id:
                                self.log_task(task_id, "Worker job returned evidence.", stage=stage, job_id=job_identifier, attempt=attempt, evidence_count=len(evidence) if isinstance(evidence, list) else -1)
                            return evidence
                        if status == "EMPTY":
                            return []

                        error_message = str(payload.get("error") or "worker returned an unknown error").strip()
                        traceback_text = str(payload.get("traceback") or "").strip()
                        if traceback_text:
                            error_message = f"{error_message}\n{traceback_text[-1600:]}"
                        raise RuntimeError(error_message)

                    worker = await self.get_worker_health()
                    if not worker["process_alive"] or worker["stale"]:
                        self._model_worker_ready = False
                        self._model_worker_error = (
                            "worker exited while processing job"
                            if not worker["process_alive"]
                            else "worker heartbeat stale while processing job"
                        )
                        raise RuntimeError(self._model_worker_error)

                    await asyncio.sleep(0.5)
                    waited += 0.5

                raise TimeoutError(f"Model worker timed out for job {job_identifier}")
            except Exception as exc:
                last_error = exc
                if task_id:
                    self.log_task(
                        task_id,
                        "Worker job attempt failed.",
                        level="warning" if attempt < max_attempts else "error",
                        stage=stage,
                        job_id=job_identifier,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=str(exc),
                    )
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(0.5)

        if last_error:
            raise last_error
        raise RuntimeError("worker job failed without an explicit error")

    @staticmethod
    def parse_worker_result_payload(content: str) -> Any:
        if not content:
            return None
        try:
            return json.loads(content)
        except Exception:
            return robust_json_parse(content)
