"""LangGraph workflow with Send()-based research-task parallelism.

Architecture:
    supervisor → dispatch_sections ──Send("section_researcher", payload)×N──→ section_researcher
                                                                                    ↓ (auto-reduce)
                                                                               reflector → [outline_builder | dispatch_sections]
                                                                                    ↓
                                                                                  writer

Changes implemented:
- Change 1: Semantic saturation-based early stopping
- Change 2: Track-level parallel research via Send() API (MAP-REDUCE)
- Change 5: Knowledge cache integration
- Change 7: Degraded task awareness in routing
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Union

import aiohttp
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents.base import AgentContext, infer_user_language, sanitize_path_name
from .knowledge_cache import KnowledgeCache
from .langsmith_utils import trace_chain
from .models import KnowledgeCard, ResearchState, SectionDigest, SectionResearchInput, SubTask


def build_graph(service: "DeepResearchService"):
    """Build the LangGraph state machine with track-level Send() parallelism."""

    workflow = StateGraph(ResearchState)

    # ────────────────────────────────────────────────────────
    #  Node: Supervisor
    #  Runs once at graph entry. Builds executable search tasks
    #  from the approved execution strategy.
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_supervisor")
    async def node_supervisor(state: ResearchState) -> Dict[str, Any]:
        task_id = state["task_id"]
        topic = state["topic"]
        planner = service.planner

        await service.store.append_task_event(task_id, "supervisor", "Building executable research tasks from the approved strategy.")
        await service.store.set_status(task_id, "Stage 1/5: organizing executable research tasks.", lifecycle="running", stage="supervisor")
        await service.store.append_progress_event(task_id, "plan_start", message="Loading execution strategy and building research tasks...")
        service.log_task(task_id, "Supervisor started.", stage="supervisor")

        execution_plan = state.get("execution_plan") or {}
        if not isinstance(execution_plan, dict) or not execution_plan.get("query_strategy"):
            output_language = infer_user_language(state.get("topic", ""), state.get("user_feedback", ""))
            execution_plan = planner.normalize_execution_plan(
                {},
                topic=topic,
                background_intent=str(state.get("user_feedback") or state.get("approved_plan") or topic),
                reconnaissance=state.get("reconnaissance") if isinstance(state.get("reconnaissance"), dict) else {},
                output_language=output_language,
            )

        sub_tasks = planner.build_research_subtasks(execution_plan)
        sub_tasks = await planner.prepare_subtasks_for_search(sub_tasks, task_id=task_id, topic=topic, stage="supervisor")

        service.save_probe(task_id, topic, "supervisor", "execution_plan", execution_plan)
        service.save_probe(task_id, topic, "supervisor", "search_sub_tasks", sub_tasks)
        await service.store.append_progress_event(
            task_id, "plan_ready",
            message=f"Research strategy ready: {len(execution_plan.get('query_strategy', []))} tracks, {len(sub_tasks)} search tasks.",
            section_count=len(execution_plan.get('query_strategy', [])), task_count=len(sub_tasks),
        )
        service.log_task(task_id, "Supervisor prepared executable research tasks.", stage="supervisor",
                         query_strategy_count=len(execution_plan.get("query_strategy", [])), sub_task_count=len(sub_tasks))
        await service.store.save_task_meta(task_id, {"sub_tasks": sub_tasks, "execution_plan": execution_plan, "stage": "supervisor"})

        return {
            "sub_tasks": sub_tasks,
            "execution_plan": execution_plan,
            "cards_before_loop": 0,
            "previous_coverage": 0.0,
            "section_results": [],
        }

    # ────────────────────────────────────────────────────────
    #  Node: Dispatch Research Tasks (Fan-Out via Send)
    #  Groups pending sub-tasks by research track and ensures
    #  the model worker is ready, then returns Send() objects.
    #  This node itself returns nothing to state — the Send()
    #  conditional edge handles the fan-out.
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_dispatch_sections")
    async def node_dispatch_sections(state: ResearchState) -> Dict[str, Any]:
        task_id = state["task_id"]
        topic = state["topic"]

        # Ensure model worker is ready before fanning out
        worker_started = time.perf_counter()
        if not await service.ensure_model_backend_ready(task_id=task_id, topic=topic, stage="researcher"):
            error = service._model_worker_error or "worker unavailable"
            service.record_timing(task_id, topic, "researcher", "worker_ready_gate", worker_started, level="error", error=error)
            raise RuntimeError(error)
        service.record_timing(task_id, topic, "researcher", "worker_ready_gate", worker_started)

        pending_tasks = [t for t in state.get("sub_tasks", []) if t.get("status") == "pending"]
        total = len(pending_tasks)

        await service.store.append_task_event(task_id, "researcher", f"Dispatching {total} search tasks across research tracks.")
        await service.store.set_status(task_id, f"Stage 2/5: collecting evidence for {total} search tasks.", lifecycle="running", stage="researcher")
        service.log_task(task_id, "Dispatch started.", stage="researcher", pending_task_count=total)

        # Record how many cards exist before this research cycle
        existing_cards = state.get("knowledge_cards", [])
        return {
            "cards_before_loop": len(existing_cards),
            "section_results": [],  # Reset for this cycle
        }

    # ────────────────────────────────────────────────────────
    #  Conditional Edge: Fan-Out to Research Workers
    #  Groups pending tasks by track id and emits one
    #  Send("section_researcher", payload) per track.
    # ────────────────────────────────────────────────────────
    def fan_out_sections(state: ResearchState) -> Sequence[Send]:
        """Return a list of Send() objects, one per section with pending tasks."""
        pending_tasks = [t for t in state.get("sub_tasks", []) if t.get("status") == "pending"]
        if not pending_tasks:
            # No pending tasks — go straight to reflector
            return [Send("collect_results", state)]

        # Group by section_id
        by_section: Dict[str, List[SubTask]] = defaultdict(list)
        for task in pending_tasks:
            sid = str(task.get("section_id") or "UNASSIGNED")
            by_section[sid].append(task)

        existing_card_count = len(state.get("knowledge_cards", []))
        task_id = state["task_id"]
        topic = state["topic"]

        sends = []
        for section_id, tasks in by_section.items():
            section_title = str(tasks[0].get("section_title") or section_id)
            payload: SectionResearchInput = {
                "task_id": task_id,
                "topic": topic,
                "section_id": section_id,
                "section_title": section_title,
                "pending_tasks": tasks,
                "existing_card_count": existing_card_count,
            }
            sends.append(Send("section_researcher", payload))

        service.log_task(task_id, f"Fanning out to {len(sends)} section researchers.", stage="researcher",
                         sections=[sid for sid in by_section.keys()])
        return sends

    # ────────────────────────────────────────────────────────
    #  Node: Section Researcher (Send Target)
    #  Runs in parallel for each section. Receives a
    #  SectionResearchInput payload via Send().
    #  Returns knowledge_cards and sub_tasks updates that
    #  are auto-reduced into the parent state.
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_section_researcher")
    async def node_section_researcher(state: SectionResearchInput) -> Dict[str, Any]:
        task_id = state["task_id"]
        topic = state["topic"]
        section_id = state["section_id"]
        section_title = state.get("section_title", section_id)
        pending_tasks = state.get("pending_tasks", [])
        existing_card_count = state.get("existing_card_count", 0)

        section_started = time.perf_counter()
        await service.store.append_progress_event(
            task_id, "section_start", section_id=section_id, status="researching",
            message=f"Researching: {section_title} ({len(pending_tasks)} queries)",
        )
        service.log_task(task_id, f"Section researcher started: {section_title}",
                         stage="researcher", section_id=section_id, task_count=len(pending_tasks))

        # Build a per-section knowledge cache for dedup within this section
        cache = KnowledgeCache(similarity_threshold=service.settings.knowledge_cache_similarity_threshold)

        result = await service.researcher.research_section(
            task_id=task_id,
            topic=topic,
            section_id=section_id,
            pending_tasks=pending_tasks,
            existing_cards=[],  # Each section starts fresh; global dedup happens at reduce
            knowledge_cache=cache,
        )

        cards_output = result.get("cards", [])
        task_updates = result.get("task_updates", {})

        # Build updated sub_tasks for this section only
        updated_tasks: List[SubTask] = []
        for task in pending_tasks:
            key = f"{task.get('intent', '')}::{task.get('query', '')}"
            updated_tasks.append(task_updates.get(key, task))

        # Progress stats for this section
        success = sum(1 for t in task_updates.values() if t.get("status") == "completed")
        failed = sum(1 for t in task_updates.values() if t.get("status") == "failed")
        degraded = sum(1 for t in task_updates.values() if t.get("status") == "degraded")

        service.record_timing(task_id, topic, "researcher", f"{section_id}_section_research", section_started,
                              section_title=section_title, cards=len(cards_output),
                              success=success, failed=failed, degraded=degraded)
        await service.store.append_progress_event(
            task_id, "section_done", section_id=section_id, status="done",
            message=f"Done: {section_title} ({len(cards_output)} cards, {success} ok, {failed} fail, {degraded} degraded)",
            cards=len(cards_output), success=success, failed=failed, degraded=degraded,
        )
        service.log_task(task_id, f"Section researcher finished: {section_title}",
                         stage="researcher", section_id=section_id,
                         cards=len(cards_output), success=success, failed=failed, degraded=degraded)

        # Return updates — reducers on knowledge_cards (operator.add) and
        # sub_tasks (_merge_sub_tasks) handle the parallel merge automatically
        return {
            "knowledge_cards": cards_output,
            "sub_tasks": updated_tasks,
            "section_results": [{
                "section_id": section_id,
                "section_title": section_title,
                "card_count": len(cards_output),
                "success": success,
                "failed": failed,
                "degraded": degraded,
            }],
        }

    # ────────────────────────────────────────────────────────
    #  Node: Collect Results (Reduce)
    #  Runs after ALL section_researcher Send() nodes complete.
    #  Aggregates stats, updates loop count, and persists metadata.
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_collect_results")
    async def node_collect_results(state: ResearchState) -> Dict[str, Any]:
        task_id = state["task_id"]
        topic = state["topic"]

        section_results = state.get("section_results", [])
        cards = state.get("knowledge_cards", [])
        sub_tasks = state.get("sub_tasks", [])

        # Aggregate stats from all sections
        total_cards = sum(r.get("card_count", 0) for r in section_results)
        total_success = sum(r.get("success", 0) for r in section_results)
        total_failed = sum(r.get("failed", 0) for r in section_results)
        total_degraded = sum(r.get("degraded", 0) for r in section_results)

        # Global dedup pass on all cards via KnowledgeCache
        global_cache = KnowledgeCache(similarity_threshold=service.settings.knowledge_cache_similarity_threshold)
        deduped_cards = global_cache.add_cards(cards)
        dedup_removed = len(cards) - len(deduped_cards)

        await service.store.save_task_meta(task_id, {
            "sub_tasks": sub_tasks,
            "knowledge_card_count": len(deduped_cards),
            "stage": "researcher",
            "research_progress": {
                "sections_researched": len(section_results),
                "total_cards": total_cards,
                "dedup_removed": dedup_removed,
                "success": total_success,
                "failed": total_failed,
                "degraded": total_degraded,
            },
        })
        service.log_task(task_id, "All section researchers completed. Results collected.", stage="researcher",
                         sections=len(section_results), total_cards=total_cards,
                         deduped=len(deduped_cards), removed=dedup_removed,
                         success=total_success, failed=total_failed, degraded=total_degraded)

        loop_count = state.get("loop_count", 0) + 1
        return {
            "knowledge_cards": deduped_cards,
            "loop_count": loop_count,
            "section_results": [],  # Clear for next cycle
        }

    # ────────────────────────────────────────────────────────
    #  Node: Reflector
    #  Reviews section quality, computes saturation, and
    #  decides whether to loop back or proceed to writer.
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_reflector")
    async def node_reflector(state: ResearchState) -> Dict[str, Any]:
        task_id = state["task_id"]
        topic = state["topic"]
        reviewer = service.reviewer
        reflector_started = time.perf_counter()

        await service.store.append_task_event(task_id, "reflector", "Reviewing section quality and computing saturation.")
        await service.store.set_status(task_id, "Stage 3/5: reviewing evidence coverage and overall coherence.", lifecycle="running", stage="reflector")
        await service.store.append_progress_event(task_id, "reflector_start", message="Reviewing research quality and saturation...")
        service.log_task(task_id, "Reflector started.", stage="reflector")

        cards = state.get("knowledge_cards", [])
        execution_plan = state.get("execution_plan") or {}
        plan_data = state.get("plan_data") or {}
        if isinstance(plan_data, dict) and isinstance(plan_data.get("sections"), list) and plan_data.get("sections"):
            sections = plan_data.get("sections")
        else:
            sections = service.planner.research_tracks_as_sections(execution_plan if isinstance(execution_plan, dict) else {})
        grouped = service.researcher.group_cards_by_section(cards)
        grouped_map = {str(g.get("section_id") or ""): g for g in grouped}

        existing = {(t.get("section_id"), t.get("query")) for t in state.get("sub_tasks", [])}
        follow_up_tasks: List[SubTask] = []
        section_reviews: List[Dict[str, Any]] = []
        section_digests: List[SectionDigest] = []
        all_missing: List[str] = []

        review_semaphore = asyncio.Semaphore(max(1, service.settings.reflector_review_concurrency))
        stagger = max(0.0, float(service.settings.reflector_review_stagger_seconds))

        async def review_one(index: int, section: Dict) -> Dict[str, Any]:
            if stagger > 0:
                await asyncio.sleep(max(0, (index - 1)) * stagger)
            async with review_semaphore:
                section_id = str(section.get("section_id") or f"S{index:02d}")
                title = str(section.get("title") or f"Section {index}")
                section_cards = (grouped_map.get(section_id) or {}).get("cards", [])
                rule_review = reviewer.rule_based_section_review(section_id, title, section, section_cards)
                llm_review = await reviewer.llm_section_review(
                    task_id=task_id, topic=topic, section_id=section_id,
                    title=title, section=section, section_cards=section_cards, rule_review=rule_review,
                )
                merged_review = reviewer.merge_section_review(section, rule_review, llm_review)
                digest = reviewer.build_section_digest(section, section_cards, merged_review)
                service.record_timing(task_id, topic, "reflector", f"{section_id}_review", reflector_started,
                                      section_title=title, card_count=len(section_cards), final_is_enough=bool(merged_review.get("is_enough")))
                return {"index": index, "section": section, "section_id": section_id, "title": title,
                        "review": merged_review, "digest": digest}

        outputs = await asyncio.gather(*(review_one(i, s) for i, s in enumerate(sections, 1)))

        for out in outputs:
            review = out["review"]
            section_reviews.append(review)
            section_digests.append(out["digest"])
            all_missing.extend(review.get("missing_questions", []))

            for req in review.get("follow_up_requests", []):
                if isinstance(req, dict):
                    query = str(req.get("query") or "").strip()
                    gap_type = str(req.get("gap_type") or "").strip()
                    evidence_goal = str(req.get("evidence_goal") or "").strip()
                else:
                    query = str(req).strip()
                    gap_type, evidence_goal = "", ""
                key = (out["section_id"], query)
                if not query or key in existing:
                    continue
                existing.add(key)
                follow_up_tasks.append({
                    "query": query, "intent": out["title"],
                    "section_id": out["section_id"], "section_title": out["title"],
                    "section_goal": str(out["section"].get("purpose") or ""),
                    "section_order": out["index"],
                    "gap_type": gap_type, "evidence_goal": evidence_goal,
                    "status": "pending",
                })

        # ── Semantic Saturation Check (Change 1) ──
        previous_coverage = float(state.get("previous_coverage", 0.0) or 0.0)
        cards_before = int(state.get("cards_before_loop", 0) or 0)
        cards_after = len(cards)

        saturation, current_coverage = reviewer.compute_saturation_score(
            section_reviews, previous_coverage, cards_before, cards_after
        )
        should_stop, stop_reason = reviewer.should_stop_early(
            saturation, state.get("loop_count", 0), bool(follow_up_tasks)
        )

        # Build quality review
        avg = lambda key: round(sum(r.get(key, 0.0) for r in section_reviews) / max(1, len(section_reviews)), 2)
        quality_review = {
            "is_enough": should_stop or not follow_up_tasks,
            "reason": stop_reason or "Proceeding to writer.",
            "missing_sections": [r.get("section_title", "") for r in section_reviews if not r.get("is_enough")],
            "quality_dimensions": {
                "coverage": avg("coverage_score"), "evidence_count": avg("evidence_count_score"),
                "source_diversity": avg("source_diversity_score"),
                "saturation_score": saturation, "current_coverage": current_coverage,
            },
            "section_reviews": section_reviews,
            "final_response_focus": [str((execution_plan or {}).get("user_goal") or topic).strip()],
            "future_outlook_focus": [str((execution_plan or {}).get("output_language") or plan_data.get("output_language") or "en") == "zh"
                                     and "总结主要趋势、未解决问题，以及未来 1 到 3 年的合理判断。"
                                     or "Summarize major trends, unresolved issues, and a reasonable 1-3 year outlook."],
        }

        # Degraded task assessment (Change 7)
        degraded_info = reviewer.assess_degraded_impact(state.get("sub_tasks", []), sections)

        save_meta = {"quality_review": quality_review, "section_digests": section_digests, "stage": "reflector",
                     "saturation_score": saturation, "degraded_impact": degraded_info}

        if not should_stop and follow_up_tasks:
            follow_up_tasks = await service.planner.prepare_subtasks_for_search(follow_up_tasks, task_id=task_id, topic=topic, stage="reflector")
            # Follow-up tasks are appended via the reducer — just return the new ones
            await service.store.save_task_meta(task_id, dict(save_meta, sub_tasks=state.get("sub_tasks", []) + follow_up_tasks))
            service.record_timing(task_id, topic, "reflector", "reflector_total", reflector_started,
                                  follow_up_count=len(follow_up_tasks), route_to="dispatch_sections", saturation=saturation)
            service.log_task(task_id, "Reflector: another research round.", stage="reflector",
                             follow_ups=len(follow_up_tasks), saturation=saturation)
            return {
                "route_to": "dispatch_sections",
                "sub_tasks": follow_up_tasks,  # Reducer merges these into existing
                "quality_review": quality_review,
                "section_digests": section_digests,
                "conflicts": list(dict.fromkeys(all_missing)),
                "previous_coverage": current_coverage,
                "saturation_score": saturation,
            }

        await service.store.save_task_meta(task_id, save_meta)
        service.record_timing(task_id, topic, "reflector", "reflector_total", reflector_started,
                              follow_up_count=0, route_to="outline_builder", saturation=saturation)
        service.log_task(task_id, "Reflector routed to evidence outline builder.", stage="reflector",
                         reason=stop_reason, saturation=saturation)
        return {
            "route_to": "outline_builder",
            "quality_review": quality_review,
            "section_digests": section_digests,
            "conflicts": list(dict.fromkeys(all_missing)),
            "previous_coverage": current_coverage,
            "saturation_score": saturation,
            "early_stop_reason": stop_reason,
        }

    # ────────────────────────────────────────────────────────
    #  Node: Outline Builder
    #  Creates the final report outline after evidence collection.
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_outline_builder")
    async def node_outline_builder(state: ResearchState) -> Dict[str, Any]:
        task_id = state["task_id"]
        topic = state["topic"]
        execution_plan = state.get("execution_plan") if isinstance(state.get("execution_plan"), dict) else {}
        output_language = str(execution_plan.get("output_language") or infer_user_language(topic, state.get("approved_plan", "")))
        await service.store.append_progress_event(task_id, "outline_start", message="Building evidence-based report outline...")
        outline = await service.planner.build_evidence_outline(
            topic=topic,
            execution_plan=execution_plan,
            section_digests=state.get("section_digests", []) if isinstance(state.get("section_digests"), list) else [],
            output_language=output_language,
            task_id=task_id,
        )
        plan_data = {
            "task_type": execution_plan.get("task_type", "deep_research"),
            "user_goal": outline.get("user_goal") or execution_plan.get("user_goal", topic),
            "report_title": outline.get("report_title") or topic,
            "sections": outline.get("sections", []),
            "final_response_goal": execution_plan.get("user_goal", topic),
            "planner_notes": outline.get("outline_notes", ""),
            "output_language": output_language,
        }
        await service.store.save_task_meta(task_id, {"evidence_outline": outline, "plan_data": plan_data, "stage": "outline_builder"})
        service.save_probe(task_id, topic, "outline_builder", "evidence_outline", outline)
        return {"evidence_outline": outline, "plan_data": plan_data}

    # ────────────────────────────────────────────────────────
    #  Node: Writer
    # ────────────────────────────────────────────────────────
    @trace_chain(name="node_writer")
    async def node_writer(state: ResearchState) -> Dict[str, Any]:
        task_id = state["task_id"]
        await service.store.append_progress_event(task_id, "writer_start", message="Writing final report...")
        result = await service.writer.write_report(state, service.get_task_output_dir)
        await service.store.append_progress_event(task_id, "writer_done", message="Report completed.")
        # Save report as version 1 (or next version for follow-ups)
        final_report = result.get("final_report", "")
        if final_report:
            version = await service.store.save_report_version(task_id, final_report, "Initial research")
            await service.store.append_progress_event(
                task_id, "report_saved", message=f"Report saved as version {version}.",
            )
        return result

    # ────────────────────────────────────────────────────────
    #  Graph Assembly
    #
    #  supervisor → dispatch_sections ─→ [Send×N] section_researcher
    #                                         ↓ (auto-reduce)
    #                                    collect_results
    #                                         ↓
    #                                    reflector ─→ outline_builder ─→ writer ─→ END
    #                                       ↑
    #                                       └── loop ── dispatch_sections
    # ────────────────────────────────────────────────────────
    workflow.add_node("supervisor", node_supervisor)
    workflow.add_node("dispatch_sections", node_dispatch_sections)
    workflow.add_node("section_researcher", node_section_researcher)
    workflow.add_node("collect_results", node_collect_results)
    workflow.add_node("reflector", node_reflector)
    workflow.add_node("outline_builder", node_outline_builder)
    workflow.add_node("writer", node_writer)

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "dispatch_sections")

    # Fan-out: dispatch_sections → [Send×N] section_researcher
    workflow.add_conditional_edges("dispatch_sections", fan_out_sections, ["section_researcher", "collect_results"])

    # All section_researcher Send() nodes auto-converge here
    workflow.add_edge("section_researcher", "collect_results")

    # Reduce → Reflector
    workflow.add_edge("collect_results", "reflector")

    # Reflector routes: loop back to dispatch_sections or build an evidence-based outline before writing.
    workflow.add_conditional_edges(
        "reflector",
        lambda state: state.get("route_to", "writer"),
        {"outline_builder": "outline_builder", "writer": "writer", "dispatch_sections": "dispatch_sections"},
    )
    workflow.add_edge("outline_builder", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=getattr(service, "graph_checkpointer", None))
