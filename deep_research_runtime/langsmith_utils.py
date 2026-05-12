from __future__ import annotations

from typing import Any, Dict, List

from langsmith import traceable


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, dict, str)) else 0


def _trim_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _compact_sub_tasks(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "query": _trim_text(item.get("query"), 120),
                "section_id": _trim_text(item.get("section_id"), 24),
                "status": _trim_text(item.get("status"), 32),
            }
        )
    return result


def summarize_research_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {"state_type": type(state).__name__}
    plan_data = state.get("plan_data") if isinstance(state.get("plan_data"), dict) else {}
    sections = plan_data.get("sections") if isinstance(plan_data.get("sections"), list) else []
    return {
        "task_id": _trim_text(state.get("task_id"), 32),
        "topic": _trim_text(state.get("topic"), 120),
        "loop_count": int(state.get("loop_count") or 0),
        "route_to": _trim_text(state.get("route_to"), 32),
        "sub_task_count": _safe_len(state.get("sub_tasks")),
        "knowledge_card_count": _safe_len(state.get("knowledge_cards")),
        "section_digest_count": _safe_len(state.get("section_digests")),
        "section_ids": [str(item.get("section_id") or "") for item in sections[:6] if isinstance(item, dict)],
        "sub_tasks_preview": _compact_sub_tasks(state.get("sub_tasks")),
        "final_report_ready": bool(str(state.get("final_report") or "").strip()),
    }


def summarize_worker_documents(documents: Any) -> List[Dict[str, Any]]:
    if not isinstance(documents, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in documents[:5]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        result.append(
            {
                "url": _trim_text(item.get("url"), 200),
                "title": _trim_text(item.get("title"), 120),
                "content_chars": len(content),
                "published_time": _trim_text(item.get("published_time"), 48),
            }
        )
    return result


def process_tool_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "self"}


def process_tool_outputs(output: Any) -> Any:
    if isinstance(output, str):
        return {"text_preview": _trim_text(output, 1200), "text_chars": len(output)}
    return output


def process_graph_node_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {key: value for key, value in inputs.items() if key != "self"}
    if "state" in filtered:
        filtered["state"] = summarize_research_state(filtered["state"])
    if "initial_state" in filtered:
        filtered["initial_state"] = summarize_research_state(filtered["initial_state"])
    if "graph_config" in filtered and isinstance(filtered["graph_config"], dict):
        filtered["graph_config"] = {"configurable": filtered["graph_config"].get("configurable", {})}
    return filtered


def process_graph_node_outputs(output: Any) -> Any:
    if not isinstance(output, dict):
        return output
    summary: Dict[str, Any] = {}
    for key, value in output.items():
        if key == "knowledge_cards":
            summary[key] = {"count": _safe_len(value)}
        elif key == "sub_tasks":
            summary[key] = {"count": _safe_len(value), "preview": _compact_sub_tasks(value)}
        elif key == "section_digests":
            summary[key] = {"count": _safe_len(value)}
        elif key == "quality_review" and isinstance(value, dict):
            summary[key] = {
                "is_enough": bool(value.get("is_enough")),
                "missing_sections": value.get("missing_sections", []),
                "quality_dimensions": value.get("quality_dimensions", {}),
            }
        elif key == "final_report":
            text = str(value or "")
            summary[key] = {"chars": len(text), "preview": _trim_text(text, 800)}
        else:
            summary[key] = value
    return summary


def process_worker_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {key: value for key, value in inputs.items() if key != "self"}
    filtered["queries"] = [_trim_text(item, 160) for item in filtered.get("queries", [])[:5]]
    filtered["documents"] = summarize_worker_documents(filtered.get("documents"))
    return filtered


def process_worker_outputs(output: Any) -> Any:
    if isinstance(output, list):
        return {
            "evidence_count": len(output),
            "preview": [
                {
                    "url": _trim_text(item.get("url"), 200),
                    "title": _trim_text(item.get("title"), 120),
                    "score": item.get("score"),
                    "excerpt_preview": _trim_text(item.get("excerpt"), 280),
                }
                for item in output[:5]
                if isinstance(item, dict)
            ],
        }
    return output


def trace_tool(*, name: str):
    return traceable(
        name=name,
        run_type="tool",
        process_inputs=process_tool_inputs,
        process_outputs=process_tool_outputs,
    )


def trace_chain(*, name: str):
    return traceable(
        name=name,
        run_type="chain",
        process_inputs=process_graph_node_inputs,
        process_outputs=process_graph_node_outputs,
    )


def trace_worker(*, name: str):
    return traceable(
        name=name,
        run_type="retriever",
        process_inputs=process_worker_inputs,
        process_outputs=process_worker_outputs,
    )
