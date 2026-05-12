import asyncio
import json
import time
import traceback
import uuid
from pathlib import Path

import aiohttp
from langsmith import tracing_context

from .agents.base import call_llm_json, infer_user_language
from .langsmith_utils import trace_chain, trace_tool
from .models import ResearchState
from .storage import now_iso


#: Maximum accepted length (in characters) for a research topic or follow-up
#: question. Keeps prompt size bounded and prevents trivial DoS by sending an
#: enormous string to the LLM.
MAX_TOPIC_LENGTH = 2000
MAX_FOLLOW_UP_LENGTH = 2000
MAX_BACKGROUND_LENGTH = 4000


def _validate_text_input(value: str, *, field: str, max_length: int) -> str:
    """Normalize and validate a user-supplied free-text field.

    Returns the trimmed string. Raises ValueError when the input is empty,
    not a string, or exceeds ``max_length``.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} is required and cannot be empty")
    if len(text) > max_length:
        raise ValueError(
            f"{field} is too long ({len(text)} chars). Maximum allowed: {max_length}."
        )
    return text


class ToolsMixin:
    @staticmethod
    def _graph_config(task_id: str) -> dict:
        return {"configurable": {"thread_id": task_id}}

    async def _get_graph_state_values(self, task_id: str) -> dict:
        try:
            await self.ensure_graph_ready()
            snapshot = await asyncio.wait_for(self.app_engine.aget_state(self._graph_config(task_id)), timeout=5.0)
        except Exception:
            return {}
        values = getattr(snapshot, "values", None)
        return values if isinstance(values, dict) else {}

    @trace_tool(name="draft_research_plan")
    async def tool_draft_plan(self, topic: str, background_intent: str) -> str:
        try:
            topic = _validate_text_input(topic, field="topic", max_length=MAX_TOPIC_LENGTH)
            background_intent = _validate_text_input(
                background_intent or "(none)",
                field="background_intent",
                max_length=MAX_BACKGROUND_LENGTH,
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        task_id = str(uuid.uuid4())[:8]
        output_language = infer_user_language(topic, background_intent)
        self.ensure_report_dir()
        runtime_issues = await self.validate_runtime_dependencies()
        planner_started_at = time.perf_counter()
        try:
            seed_queries = await self.planner.generate_seed_queries(
                topic=topic,
                background_intent=background_intent,
                output_language=output_language,
                task_id=task_id,
            )
            verticals = self.search_service.infer_verticals(f"{topic} {background_intent}")
            timeout = aiohttp.ClientTimeout(total=max(20, self.settings.page_fetch_timeout * max(1, len(seed_queries))))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                reconnaissance_docs = await self.search_service.reconnaissance(
                    session,
                    seed_queries,
                    verticals=verticals,
                    max_results_per_query=self.settings.draft_max_results_per_query,
                )
            reconnaissance = self.planner.build_reconnaissance_result(
                seed_queries=seed_queries,
                documents=reconnaissance_docs,
            )
            execution_plan = await self.planner.draft_execution_plan(
                topic=topic,
                background_intent=background_intent,
                reconnaissance=reconnaissance,
                output_language=output_language,
                task_id=task_id,
            )
        except Exception as exc:
            self.log_task(task_id, "Execution-plan draft failed, using fallback plan.", level="warning", stage="draft", error=str(exc))
            seed_queries = self.planner._fallback_seed_queries(topic, background_intent)
            reconnaissance = self.planner.build_reconnaissance_result(seed_queries=seed_queries, documents=[])
            execution_plan = self.planner.normalize_execution_plan(
                {},
                topic=topic,
                background_intent=background_intent,
                reconnaissance=reconnaissance,
                output_language=output_language,
            )
        self.record_timing(
            task_id,
            topic,
            "draft",
            "planner_total",
            planner_started_at,
            seed_query_count=len(seed_queries),
            reconnaissance_document_count=len(reconnaissance.get("documents", [])),
            query_strategy_count=len(execution_plan.get("query_strategy", [])),
        )
        plan_text = self.planner.render_execution_plan(execution_plan, reconnaissance, output_language=output_language)
        log_path = self.setup_task_logger(task_id, topic)
        trace_path = self.get_task_trace_path(task_id, topic)
        await self.store.save_draft(task_id, topic, plan_text)
        await self.store.save_task_meta(
            task_id,
            {
                "task_id": task_id,
                "topic": topic,
                "background_intent": background_intent,
                "approved_plan": plan_text,
                "execution_plan": execution_plan,
                "reconnaissance": reconnaissance,
                "plan_data": {},
                "output_language": output_language,
                "lifecycle": "draft",
                "created_at": now_iso(),
                "log_path": log_path,
                "trace_path": str(trace_path),
                "graph_thread_id": task_id,
                "graph_checkpoint_path": str(getattr(self, "_graph_checkpoint_path", "")),
                "task_registry_path": str(getattr(self.store, "registry_path", "")),
                "runtime_issues": runtime_issues,
            },
        )
        await self.store.append_task_event(task_id, "draft", "Research plan draft created.", runtime_issues=runtime_issues)
        await self.store.set_status(task_id, "Waiting for plan approval.", lifecycle="draft", stage="draft")
        self.save_probe(task_id, topic, "draft", "execution_plan", execution_plan)
        self.save_probe(task_id, topic, "draft", "reconnaissance", reconnaissance)
        self.save_probe(task_id, topic, "draft", "runtime_issues", runtime_issues)
        self.log_task(
            task_id,
            "Execution-plan draft created.",
            stage="draft",
            topic=topic,
            runtime_issue_count=len(runtime_issues),
            query_strategy_count=len(execution_plan.get("query_strategy", [])),
        )

        lines = [f"task_id: `{task_id}`", "", plan_text]
        if runtime_issues:
            lines.extend(["", "runtime_issues:"])
            lines.extend(f"- {issue}" for issue in runtime_issues)
        lines.extend(["", "Wait for explicit user approval before starting the research task."])
        return "\n".join(lines)

    @trace_tool(name="start_research_task")
    async def tool_execute_plan(self, task_id: str, user_feedback: str = "approve") -> str:
        meta = await self.store.load_task_meta(task_id)
        if not meta:
            return f"Task `{task_id}` was not found."
        if task_id in self._active_background_tasks:
            return f"Task `{task_id}` is already running in the background."
        draft = await self.store.load_draft(task_id)
        if not draft:
            return f"Draft for task `{task_id}` was not found."

        topic = str(meta.get("topic") or draft.get("topic") or task_id)
        approved_plan = str(meta.get("approved_plan") or draft.get("plan") or "")
        plan_data = meta.get("plan_data") if isinstance(meta.get("plan_data"), dict) else {}
        execution_plan = meta.get("execution_plan") if isinstance(meta.get("execution_plan"), dict) else {}
        reconnaissance = meta.get("reconnaissance") if isinstance(meta.get("reconnaissance"), dict) else {}
        output_language = str(meta.get("output_language") or infer_user_language(topic, user_feedback))
        lifecycle = str(meta.get("lifecycle") or "")
        if lifecycle in {"completed", "cancelled"}:
            return f"Task `{task_id}` is already `{lifecycle}`."

        log_path = self.setup_task_logger(task_id, topic)
        trace_path = self.get_task_trace_path(task_id, topic)
        await self.store.save_task_meta(
            task_id,
            {
                "topic": topic,
                "approved_plan": approved_plan,
                "plan_data": plan_data,
                "execution_plan": execution_plan,
                "reconnaissance": reconnaissance,
                "user_feedback": user_feedback,
                "output_language": output_language,
                "lifecycle": "running",
                "stage": "startup",
                "started_at": now_iso(),
                "log_path": log_path,
                "trace_path": str(trace_path),
                "graph_thread_id": task_id,
                "graph_checkpoint_path": str(getattr(self, "_graph_checkpoint_path", "")),
                "task_registry_path": str(getattr(self.store, "registry_path", "")),
            },
        )
        await self.store.append_task_event(task_id, "startup", "Task approved and queued for background execution.", user_feedback=user_feedback)
        await self.store.set_status(task_id, "Stage 0/4: task accepted and queued.", lifecycle="running", stage="startup")
        self.log_task(task_id, "Task approved and queued.", stage="startup", user_feedback=user_feedback)

        await self.ensure_graph_ready()
        graph_config = self._graph_config(task_id)

        initial_state: ResearchState = {
            "task_id": task_id,
            "topic": topic,
            "approved_plan": approved_plan,
            "plan_data": plan_data,
            "execution_plan": execution_plan,
            "reconnaissance": reconnaissance,
            "user_feedback": user_feedback,
            "sub_tasks": [],
            "knowledge_cards": [],
            "section_digests": [],
            "conflicts": [],
            "quality_review": {},
            "loop_count": 0,
            "route_to": "supervisor",
            "final_report": "",
            "section_results": [],
        }

        async def consume_graph() -> None:
            with self._langsmith_graph_context(task_id=task_id, topic=topic):
                await self._run_graph_stream(initial_state, graph_config)

        async def run_graph() -> None:
            graph_started_at = time.perf_counter()
            try:
                await asyncio.wait_for(consume_graph(), timeout=self.settings.task_execution_timeout)
            except asyncio.CancelledError:
                self.log_task(task_id, "Background task cancelled.", level="warning", stage="cancelled")
                await self.store.save_task_meta(task_id, {"lifecycle": "cancelled", "stage": "cancelled", "cancelled_at": now_iso()})
                await self.store.append_task_event(task_id, "cancelled", "Task execution cancelled.", level="warning")
                await self.store.set_status(task_id, "Task was cancelled.", lifecycle="cancelled", stage="cancelled")
                raise
            except Exception as exc:
                error_code = self.classify_error(exc)
                error_message = str(exc).strip() or exc.__class__.__name__
                # Capture the full traceback so post-mortem debugging doesn't
                # require re-running the failing scenario. We bound the size
                # because Python tracebacks can balloon when LLM payloads or
                # large dicts are involved, and a 10MB traceback would bloat
                # the sqlite registry.
                traceback_text = traceback.format_exc()
                if len(traceback_text) > 16000:
                    traceback_text = traceback_text[:8000] + "\n... [truncated] ...\n" + traceback_text[-8000:]
                exception_type = f"{exc.__class__.__module__}.{exc.__class__.__name__}"
                self.log_task(task_id, "Background task failed.", level="error", stage="failed", error_code=error_code, error=error_message)
                self.save_probe(
                    task_id, topic, "task", "task_error",
                    {
                        "error": error_message,
                        "error_code": error_code,
                        "exception_type": exception_type,
                        "traceback": traceback_text,
                    },
                )
                await self.store.save_task_meta(
                    task_id,
                    {
                        "lifecycle": "failed",
                        "stage": "failed",
                        "error": error_message,
                        "error_code": error_code,
                        "exception_type": exception_type,
                        "failed_at": now_iso(),
                    },
                )
                await self.store.append_task_event(
                    task_id, "failed", "Task execution failed.",
                    level="error",
                    error=error_message,
                    error_code=error_code,
                    exception_type=exception_type,
                    traceback=traceback_text,
                )
                await self.store.set_status(task_id, f"Task failed: {error_message}", lifecycle="failed", stage="failed", error=error_message, error_code=error_code)
            else:
                final_meta = await self.store.load_task_meta(task_id)
                if str(final_meta.get("lifecycle") or "") != "completed":
                    await self.store.save_task_meta(task_id, {"lifecycle": "completed", "stage": "completed", "completed_at": now_iso()})
                    await self.store.append_task_event(task_id, "completed", "Task execution finished.")
                    await self.store.set_status(task_id, "Task completed.", lifecycle="completed", stage="completed")
                self.log_task(task_id, "Background task finished.", stage="completed")
            finally:
                self.record_timing(task_id, topic, "task", "task_total", graph_started_at)
                self._active_background_tasks.pop(task_id, None)

        self._active_background_tasks[task_id] = asyncio.create_task(run_graph(), name=f"deep_research_{task_id}")
        return (
            "Research task handed off successfully.\n"
            f"task_id: `{task_id}`\n"
            "Use `get_research_status` to check progress later."
        )

    @trace_tool(name="get_research_status")
    async def tool_check_status(self, task_id: str) -> str:
        meta = await self.store.load_task_meta(task_id)
        if not meta:
            return f"Task `{task_id}` was not found."
        graph_state = await self._get_graph_state_values(task_id)
        status = await self.store.get_status(task_id) or str(meta.get("status_message") or "No status yet.")
        sub_tasks = meta.get("sub_tasks") if isinstance(meta.get("sub_tasks"), list) else []
        total = len(sub_tasks)
        finished = sum(1 for item in sub_tasks if str(item.get("status") or "") in {"completed", "failed"})
        success = sum(1 for item in sub_tasks if str(item.get("status") or "") == "completed")
        failed = sum(1 for item in sub_tasks if str(item.get("status") or "") == "failed")
        lines = [
            f"task_id: `{task_id}`",
            f"topic: {meta.get('topic', '')}",
            f"status: {status}",
            f"lifecycle: {meta.get('lifecycle', 'unknown')}",
            f"stage: {meta.get('stage', 'unknown')}",
        ]
        if graph_state:
            loop_count = int(graph_state.get("loop_count") or 0)
            card_count = len(graph_state.get("knowledge_cards") or []) if isinstance(graph_state.get("knowledge_cards"), list) else 0
            route_to = str(graph_state.get("route_to") or "").strip()
            lines.append(f"graph_state: loop_count={loop_count}, knowledge_cards={card_count}, route_to={route_to or 'n/a'}")
        if total:
            lines.append(f"progress: total={total}, finished={finished}, success={success}, failed={failed}")
        for key in ("task_dir", "log_path", "trace_path", "report_path", "graph_checkpoint_path", "task_registry_path"):
            value = str(meta.get(key) or "").strip()
            if value:
                lines.append(f"{key}: {value}")
        error_code = str(meta.get("error_code") or "").strip()
        error = str(meta.get("error") or "").strip()
        if error_code:
            lines.append(f"error_code: {error_code}")
        if error:
            lines.append(f"error: {error}")

        # Real-time progress events (Improvement 3)
        progress_events = await self.store.get_progress_events(task_id, limit=15)
        if progress_events:
            lines.append("")
            lines.append("progress_timeline:")
            for evt in progress_events:
                ts = str(evt.get('timestamp', ''))[-8:]  # HH:MM:SS
                etype = evt.get('type', '')
                msg = evt.get('message', '')
                sid = evt.get('section_id', '')
                prefix = f"  [{ts}]" if ts else "  "
                if sid:
                    lines.append(f"{prefix} [{sid}] {msg}")
                else:
                    lines.append(f"{prefix} {msg}")

        return "\n".join(lines)

    @trace_tool(name="check_research_runtime")
    async def tool_check_runtime(self) -> str:
        summary = await self.runtime_summary()
        issues = summary.get("issues") if isinstance(summary.get("issues"), list) else []
        lines = [
            f"healthy: {summary.get('healthy', False)}",
            f"worker_state: {summary.get('worker_state', 'unknown')}",
            f"active_task_count: {summary.get('active_task_count', 0)}",
            f"llm_model: {summary.get('llm_model', '')}",
            f"search_backend: {summary.get('search_backend', '')}",
            f"report_dir: {summary.get('report_dir', '')}",
        ]
        last_error = str(summary.get("worker_error") or "").strip()
        if last_error:
            lines.append(f"last_error: {last_error}")
        if issues:
            lines.append("issues:")
            lines.extend(f"- {issue}" for issue in issues)
        else:
            lines.append("issues: none")
        return "\n".join(lines)

    @trace_tool(name="get_research_result")
    async def tool_get_result(self, task_id: str) -> str:
        meta = await self.store.load_task_meta(task_id)
        if not meta:
            return f"Task `{task_id}` was not found."
        graph_state = await self._get_graph_state_values(task_id)
        lines = [
            f"task_id: `{task_id}`",
            f"lifecycle: {meta.get('lifecycle', 'unknown')}",
        ]
        if graph_state:
            loop_count = int(graph_state.get("loop_count") or 0)
            final_report = str(graph_state.get("final_report") or "").strip()
            lines.append(f"graph_checkpoint: loop_count={loop_count}, final_report_ready={bool(final_report)}")
        for key in ("task_dir", "report_path", "cards_path", "sources_path", "activity_path", "metadata_path", "log_path", "trace_path", "graph_checkpoint_path", "task_registry_path"):
            value = str(meta.get(key) or "").strip()
            if value:
                lines.append(f"{key}: {value}")
        report_path = str(meta.get("report_path") or "").strip()
        if report_path and Path(report_path).exists():
            preview = Path(report_path).read_text(encoding="utf-8")[:1500].strip()
            lines.extend(["", "report_preview:", preview])
        elif str(meta.get("lifecycle") or "") != "completed":
            lines.append("result: report is not ready yet")
        quality_review = meta.get("quality_review")
        if quality_review:
            lines.extend(["", "quality_review:", json.dumps(quality_review, ensure_ascii=False, indent=2)])

        # Report version info
        versions = await self.store.list_report_versions(task_id)
        if versions:
            lines.append("")
            lines.append(f"report_versions: {len(versions)} version(s)")
            for v in versions:
                lines.append(f"  v{v['version']}: {v['change_summary']} ({v['created_at']})")

        return "\n".join(lines)

    @trace_tool(name="follow_up_research")
    async def tool_follow_up_research(self, task_id: str, follow_up_question: str) -> str:
        """Execute follow-up research on a completed task.

        Restores graph state from checkpoint, generates incremental sub-tasks
        for the follow-up question, and patches the original report.
        """
        try:
            follow_up_question = _validate_text_input(
                follow_up_question,
                field="follow_up_question",
                max_length=MAX_FOLLOW_UP_LENGTH,
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        meta = await self.store.load_task_meta(task_id)
        if not meta:
            return f"Task `{task_id}` was not found."
        lifecycle = str(meta.get("lifecycle") or "")
        if lifecycle != "completed":
            return f"Task `{task_id}` is not completed (lifecycle={lifecycle}). Follow-up requires a completed task."
        if task_id in self._active_background_tasks:
            return f"Task `{task_id}` already has a running background job."

        # Restore state from checkpoint
        graph_state = await self._get_graph_state_values(task_id)
        if not graph_state:
            return f"Cannot restore graph state for task `{task_id}`. Checkpoint may be missing."

        topic = str(meta.get("topic") or task_id)
        existing_cards = graph_state.get("knowledge_cards", [])
        existing_plan = graph_state.get("plan_data", {})
        output_language = str(meta.get("output_language") or infer_user_language(topic, follow_up_question))

        # Generate incremental plan via Planner
        follow_up_plan = await self.planner.plan_follow_up(
            question=follow_up_question,
            existing_plan=existing_plan,
            existing_cards=existing_cards,
            task_id=task_id,
            topic=topic,
            output_language=output_language,
        )

        if not follow_up_plan or not follow_up_plan.get("new_sub_tasks"):
            return f"Planner could not generate follow-up tasks for: {follow_up_question}"

        log_path = self.setup_task_logger(task_id, topic)
        await self.store.save_task_meta(task_id, {
            "lifecycle": "running",
            "stage": "follow_up",
            "follow_up_question": follow_up_question,
            "follow_up_at": now_iso(),
        })
        await self.store.append_task_event(task_id, "follow_up", f"Follow-up: {follow_up_question}")
        await self.store.append_progress_event(
            task_id, "follow_up_start",
            message=f"Starting follow-up research: {follow_up_question}",
        )

        # Build incremental state — reuse existing cards, inject new sub_tasks
        await self.ensure_graph_ready()
        graph_config = self._graph_config(task_id)

        incremental_state: ResearchState = {
            **graph_state,
            "sub_tasks": follow_up_plan["new_sub_tasks"],
            "plan_data": follow_up_plan.get("updated_plan", existing_plan),
            "loop_count": 0,
            "route_to": "dispatch_sections",
            "section_results": [],
            "final_report": "",
            "cards_before_loop": len(existing_cards),
            "previous_coverage": 0.0,
        }

        async def consume_follow_up() -> None:
            with self._langsmith_graph_context(task_id=task_id, topic=topic):
                await self._run_graph_stream(incremental_state, graph_config)

        async def run_follow_up() -> None:
            graph_started_at = time.perf_counter()
            try:
                await asyncio.wait_for(consume_follow_up(), timeout=self.settings.task_execution_timeout)
            except asyncio.CancelledError:
                await self.store.set_status(task_id, "Follow-up cancelled.", lifecycle="cancelled")
                raise
            except Exception as exc:
                error_msg = str(exc).strip() or exc.__class__.__name__
                # Same traceback-capture treatment as the primary task path
                # so follow-ups are equally debuggable after the fact.
                traceback_text = traceback.format_exc()
                if len(traceback_text) > 16000:
                    traceback_text = traceback_text[:8000] + "\n... [truncated] ...\n" + traceback_text[-8000:]
                exception_type = f"{exc.__class__.__module__}.{exc.__class__.__name__}"
                self.log_task(task_id, "Follow-up failed.", level="error", stage="failed", error=error_msg)
                await self.store.save_task_meta(
                    task_id,
                    {
                        "lifecycle": "failed",
                        "error": error_msg,
                        "exception_type": exception_type,
                        "failed_at": now_iso(),
                    },
                )
                await self.store.append_task_event(
                    task_id, "failed", "Follow-up execution failed.",
                    level="error",
                    error=error_msg,
                    exception_type=exception_type,
                    traceback=traceback_text,
                )
                await self.store.set_status(task_id, f"Follow-up failed: {error_msg}", lifecycle="failed")
            else:
                final_meta = await self.store.load_task_meta(task_id)
                if str(final_meta.get("lifecycle") or "") != "completed":
                    await self.store.save_task_meta(task_id, {"lifecycle": "completed", "stage": "completed", "completed_at": now_iso()})
                    await self.store.set_status(task_id, "Follow-up completed.", lifecycle="completed")
            finally:
                self.record_timing(task_id, topic, "task", "follow_up_total", graph_started_at)
                self._active_background_tasks.pop(task_id, None)

        self._active_background_tasks[task_id] = asyncio.create_task(run_follow_up(), name=f"follow_up_{task_id}")
        return (
            "Follow-up research started.\n"
            f"task_id: `{task_id}`\n"
            f"question: {follow_up_question}\n"
            f"new_tasks: {len(follow_up_plan['new_sub_tasks'])}\n"
            "Use `get_research_status` to check progress."
        )

    @trace_tool(name="compare_report_versions")
    async def tool_compare_versions(self, task_id: str, version_a: int = 0, version_b: int = 0) -> str:
        """Compare two versions of a research report.

        version_a and version_b default to the two most recent versions.
        Returns a section-by-section diff summary.
        """
        versions = await self.store.list_report_versions(task_id)
        if len(versions) < 2:
            return f"Task `{task_id}` has {len(versions)} version(s). Need at least 2 to compare."

        if version_a <= 0 or version_b <= 0:
            version_a = versions[-2]["version"]
            version_b = versions[-1]["version"]

        va = await self.store.load_report_version(task_id, version_a)
        vb = await self.store.load_report_version(task_id, version_b)
        if not va or not vb:
            return f"Could not load version {version_a} or {version_b}."

        # Parse sections from both versions
        import re
        def parse_sections(content: str) -> dict:
            sections = {}
            parts = re.split(r'^(#{1,2}\s+.+)$', content, flags=re.MULTILINE)
            current_heading = "preamble"
            for part in parts:
                if re.match(r'^#{1,2}\s+', part):
                    current_heading = part.strip()
                else:
                    sections[current_heading] = part.strip()
            return sections

        sa = parse_sections(va["content"])
        sb = parse_sections(vb["content"])
        all_headings = list(dict.fromkeys(list(sa.keys()) + list(sb.keys())))

        lines = [
            f"Version comparison: v{version_a} vs v{version_b}",
            f"v{version_a}: {va.get('change_summary', '')} ({va.get('created_at', '')})",
            f"v{version_b}: {vb.get('change_summary', '')} ({vb.get('created_at', '')})",
            "",
        ]
        for heading in all_headings:
            a_text = sa.get(heading, "")
            b_text = sb.get(heading, "")
            if heading == "preamble" and not a_text and not b_text:
                continue
            if a_text == b_text:
                lines.append(f"  {heading}: [unchanged]")
            elif not a_text:
                lines.append(f"  {heading}: [NEW in v{version_b}] (+{len(b_text)} chars)")
            elif not b_text:
                lines.append(f"  {heading}: [REMOVED in v{version_b}]")
            else:
                diff_chars = len(b_text) - len(a_text)
                sign = "+" if diff_chars > 0 else ""
                lines.append(f"  {heading}: [MODIFIED] ({sign}{diff_chars} chars)")

        return "\n".join(lines)

    @trace_chain(name="graph_background_execution")
    async def _run_graph_stream(self, initial_state: ResearchState, graph_config: dict) -> None:
        async for _ in self.app_engine.astream(initial_state, config=graph_config):
            pass

    def _langsmith_graph_context(self, task_id: str, topic: str):
        metadata = {
            "task_id": task_id,
            "topic": topic,
            "component": "graph_execution",
        }
        return tracing_context(
            project_name=self.settings.langsmith_project,
            enabled=self.settings.langsmith_tracing,
            client=self._langsmith_client,
            metadata=metadata,
            tags=["deep-research", "graph"],
        )
