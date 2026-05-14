import multiprocessing as mp
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from deep_research_runtime.service import DeepResearchService


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


service = DeepResearchService()
mcp = FastMCP(service.settings.mcp_name)


@mcp.tool()
async def check_research_runtime() -> str:
    """Check whether the deep research runtime is basically available."""
    return await service.tool_check_runtime()


@mcp.tool()
async def draft_research_plan(topic: str, background_intent: str) -> str:
    """Draft a research plan and return a task id plus the proposed outline."""
    return await service.tool_draft_plan(topic, background_intent)


@mcp.tool()
async def start_research_task(task_id: str, plan_adjustments: str = "") -> str:
    """Start the approved research task in the background only after the user explicitly approves the plan.

    ``plan_adjustments`` is an optional free-text note from the user — any
    extra constraints, focus shifts, or clarifications attached to the
    approval. Pass an empty string when the user approves the plan as-is.
    The note is fed back to the planner if the execution plan needs to be
    re-normalized at startup.
    """
    return await service.tool_execute_plan(task_id, plan_adjustments)


@mcp.tool()
async def get_research_status(task_id: str) -> str:
    """Return the current status, progress, real-time progress timeline, and local output paths for a task."""
    return await service.tool_check_status(task_id)


@mcp.tool()
async def get_research_result(task_id: str) -> str:
    """Return the final artifact paths, report preview, and version history for a task."""
    return await service.tool_get_result(task_id)


@mcp.tool()
async def follow_up_research(task_id: str, follow_up_question: str) -> str:
    """Run incremental follow-up research on a completed task. Expands the existing report with new evidence."""
    return await service.tool_follow_up_research(task_id, follow_up_question)


@mcp.tool()
async def cancel_research_task(task_id: str) -> str:
    """Cancel an in-progress research task. No-op when the task is already
    completed, failed, or cancelled. Returns a status message either way."""
    return await service.tool_cancel(task_id)


@mcp.tool()
async def list_research_tasks(limit: int = 20, lifecycle: str = "") -> str:
    """List recent research tasks (most-recently updated first).

    ``limit`` is clamped to [1, 100]. ``lifecycle`` filters by lifecycle
    value (``draft`` / ``running`` / ``completed`` / ``failed`` /
    ``cancelled``); empty string returns all.
    """
    return await service.tool_list_tasks(limit, lifecycle)


@mcp.tool()
async def compare_report_versions(task_id: str, version_a: int = 0, version_b: int = 0) -> str:
    """Compare two versions of a research report.

    Report versions are 1-indexed. Passing ``0`` (the default) for either
    argument means "auto-select"; when both are ``0`` the comparison runs
    against the two most recent versions. Returns a section-level diff
    summary (unchanged / new / removed / modified with char delta).
    """
    return await service.tool_compare_versions(task_id, version_a, version_b)


def main() -> int:
    """Entrypoint used by the console script registered in pyproject.toml.

    Delegates to :mod:`deep_research_runtime.cli`, which handles the
    ``init`` / ``doctor`` subcommands. When no subcommand is given it starts
    the MCP server over stdio (the default mode for end users).
    """
    from deep_research_runtime.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    mp.freeze_support()
    # If the user passed a subcommand (``init`` / ``doctor`` / ``--help``),
    # let the CLI router handle it without touching the heavy MCP runtime.
    if len(sys.argv) > 1:
        raise SystemExit(main())
    service.prepare_startup()
    mcp.run()
