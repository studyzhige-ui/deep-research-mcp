"""Command-line entry points for the Deep Research MCP server.

This module exposes two top-level commands:

* ``deep-research-mcp`` (no args) — start the MCP server over stdio. This is
  what end users wire into Claude Code / Claude Desktop / Codex / Cursor.

* ``deep-research-mcp init`` — interactive setup wizard. Asks the user a few
  questions and prints a ready-to-paste MCP client config snippet. The wizard
  intentionally does NOT write API keys to disk — it only emits text that the
  user can copy into their own client config.

* ``deep-research-mcp doctor`` — non-destructive config self-check. Loads
  :class:`Settings` from the current environment and prints the same report
  emitted at server startup, so users can debug their config without booting
  the worker subprocess.

The wizard is intentionally dependency-light: it uses only ``input()`` and the
standard library so it works in any terminal without prompting the user to
install extras like rich/click.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
#  Wizard internals
# ──────────────────────────────────────────────────────────────────────

# Each tuple is: (env_var, prompt, help_text, required)
SEARCH_ENGINES: List[Tuple[str, str, str, bool]] = [
    (
        "TAVILY_API_KEY",
        "Tavily API key",
        "AI-optimized web search with full-text content. Free tier: 1000 requests/month. "
        "Get one at https://tavily.com — strongly recommended as your primary engine.",
        True,
    ),
    (
        "EXA_API_KEY",
        "Exa API key (optional)",
        "Neural semantic search, useful for technical / long-form content. "
        "Get one at https://exa.ai. Skip with Enter.",
        False,
    ),
    (
        "SERPER_API_KEY",
        "Serper API key (optional)",
        "Google Search wrapper, useful for fresh news. https://serper.dev. Skip with Enter.",
        False,
    ),
    (
        "BOCHA_API_KEY",
        "Bocha API key (optional)",
        "Chinese-language semantic search. Skip with Enter if you don't search Chinese sources.",
        False,
    ),
]

LLM_PROVIDERS: List[Tuple[str, str, str]] = [
    ("deepseek", "deepseek/deepseek-v4-flash", "DEEPSEEK_API_KEY"),
    ("openai", "openai/gpt-4.1-mini", "OPENAI_API_KEY"),
    ("anthropic", "anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
    ("gemini", "gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
]


def _prompt(text: str, *, default: str = "", secret: bool = False) -> str:
    """Wrapper around ``input()`` with a default value and trim.

    ``secret`` is currently a no-op (we don't echo masked input) — it's a
    placeholder in case we later want to integrate ``getpass``. We keep the
    raw API key visible during the wizard because the user is the one typing
    it on their own machine; masking would only hurt copy/paste accuracy.
    """
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"  {text}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def _print_header(text: str) -> None:
    bar = "─" * max(8, len(text) + 4)
    print(f"\n{bar}\n  {text}\n{bar}")


def _emit_client_snippets(env: Dict[str, str]) -> None:
    """Print copy-paste-ready MCP client config blocks for the major clients."""
    print("\nCopy ONE of the following blocks into your MCP client config.\n")

    json_env = json.dumps(env, indent=6).replace("\n", "\n      ")
    claude_block = (
        '{\n'
        '  "mcpServers": {\n'
        '    "deep-research": {\n'
        '      "command": "deep-research-mcp",\n'
        f'      "env": {json_env}\n'
        '    }\n'
        '  }\n'
        '}'
    )

    print("── Claude Desktop / Claude Code (claude_desktop_config.json or ~/.claude.json) ──")
    print(claude_block)

    # TOML for Codex
    print("\n── Codex (~/.codex/config.toml) ──")
    print("[mcp_servers.deep-research]")
    print('command = "deep-research-mcp"')
    print('args = []')
    print("env = {")
    items = [f'  {k} = {json.dumps(v)}' for k, v in env.items()]
    print(",\n".join(items))
    print("}")

    # Cursor
    print("\n── Cursor (~/.cursor/mcp.json) ──")
    print(claude_block)

    print("\nRestart your MCP client after saving and the `deep-research` tools should appear.")


def run_init_wizard() -> int:
    """Interactive wizard. Returns a Unix-style exit code (0 = success)."""
    _print_header("Deep Research MCP — setup wizard")
    print(
        "This wizard helps you build a config snippet for your MCP client.\n"
        "It does NOT write API keys to disk — you paste the output into your\n"
        "own client config file. Press Enter to skip any optional field."
    )

    env: Dict[str, str] = {}

    # 1) LLM provider
    _print_header("1. Language model")
    print("Pick the LLM provider you want this server to call:")
    for idx, (name, default_model, _) in enumerate(LLM_PROVIDERS, start=1):
        print(f"    {idx}) {name:<10} (default model: {default_model})")
    choice = _prompt("Choice (1-4)", default="1")
    try:
        provider_idx = max(1, min(len(LLM_PROVIDERS), int(choice))) - 1
    except ValueError:
        provider_idx = 0
    provider_name, default_model, provider_env_var = LLM_PROVIDERS[provider_idx]

    model = _prompt("Model identifier", default=default_model)
    env["DEEP_RESEARCH_LLM_MODEL"] = model

    api_key = _prompt(f"{provider_env_var} (required)", secret=True)
    if api_key:
        env[provider_env_var] = api_key
        # Also mirror to the generic DEEP_RESEARCH_LLM_API_KEY for clarity.
        env["DEEP_RESEARCH_LLM_API_KEY"] = api_key
    else:
        print(f"  ! No {provider_env_var} provided — you must add one before running real tasks.")

    # 2) Search engines
    _print_header("2. Search engines")
    print("Configure at least one search engine. Tavily is strongly recommended.")
    configured_any_search = False
    for env_var, label, help_text, required in SEARCH_ENGINES:
        print(f"\n  • {help_text}")
        value = _prompt(label, secret=True)
        if value:
            env[env_var] = value
            configured_any_search = True
        elif required:
            print(f"    ! Skipped — without {env_var} you'll only have DuckDuckGo as a fallback.")

    if any(env.get(k) for k in ("EXA_API_KEY", "SERPER_API_KEY", "BOCHA_API_KEY")):
        env["DEEP_RESEARCH_MULTI_SOURCE"] = "1"

    if not configured_any_search:
        # Make sure DDG is on as the last-resort fallback.
        env["DEEP_RESEARCH_DUCKDUCKGO_FALLBACK"] = "1"

    # 3) Optional: LangSmith
    _print_header("3. Observability (optional)")
    langsmith_key = _prompt("LangSmith API key (Enter to skip)", secret=True)
    if langsmith_key:
        env["LANGSMITH_API_KEY"] = langsmith_key
        env["LANGCHAIN_PROJECT"] = _prompt("LangSmith project", default="DeepResearch")

    # 4) Output
    _print_header("4. Config snippets")
    _emit_client_snippets(env)
    return 0


# ──────────────────────────────────────────────────────────────────────
#  Doctor — non-destructive config check
# ──────────────────────────────────────────────────────────────────────


def run_doctor() -> int:
    """Run a configuration self-check without starting the worker."""
    from .settings import Settings

    settings = Settings()
    report = settings.validate_and_report(emit=True)
    if report["blocking_errors"]:
        return 1
    return 0


# ──────────────────────────────────────────────────────────────────────
#  Prune — opt-in maintenance command
# ──────────────────────────────────────────────────────────────────────


def run_prune(extra_args: List[str]) -> int:
    """Delete checkpoints and registry rows for terminal tasks older than N days.

    Usage::

        deep-research-mcp prune [--days N] [--dry-run]

    Defaults come from ``settings.checkpoint_retention_days`` (env var
    ``DEEP_RESEARCH_CHECKPOINT_RETENTION_DAYS``, default 30).
    """
    from .maintenance import prune
    from .settings import Settings

    days: Optional[int] = None
    dry_run = False
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg in {"--dry-run", "-n"}:
            dry_run = True
        elif arg == "--days" and i + 1 < len(extra_args):
            try:
                days = int(extra_args[i + 1])
            except ValueError:
                print(f"--days expects an integer, got {extra_args[i + 1]!r}", file=sys.stderr)
                return 2
            i += 1
        elif arg.startswith("--days="):
            try:
                days = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"--days expects an integer", file=sys.stderr)
                return 2
        else:
            print(f"Unknown argument: {arg!r}. Try `deep-research-mcp help`.", file=sys.stderr)
            return 2
        i += 1

    settings = Settings()
    result = prune(settings, retention_days=days, dry_run=dry_run)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    candidates = result["candidates"]
    if dry_run:
        print(
            f"\n[dry-run] Would prune {len(candidates)} task(s). "
            f"Re-run without --dry-run to actually delete.",
            file=sys.stderr,
        )
    else:
        ckpt_total = sum(result["checkpoint_deleted"].values())
        reg_total = sum(result["registry_deleted"].values())
        print(
            f"\nPruned {len(candidates)} task(s); removed "
            f"{ckpt_total} checkpoint row(s) and {reg_total} registry row(s).",
            file=sys.stderr,
        )
    return 0


# ──────────────────────────────────────────────────────────────────────
#  Server entrypoint dispatch
# ──────────────────────────────────────────────────────────────────────


def _start_server() -> int:
    """Lazy-import the MCP server so `init`/`doctor` don't pay startup cost."""
    import multiprocessing as mp

    # Late import — module-level imports inside deep_research_mcp.py do heavy
    # work (model worker setup, langgraph initialisation). The wizard should
    # remain instant.
    from deep_research_mcp import mcp, service

    mp.freeze_support()
    service.prepare_startup()
    mcp.run()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Top-level CLI dispatch.

    Subcommands:
        (none)     start the MCP server
        init       run the interactive setup wizard
        doctor     print a config self-check report
        --help     show usage
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        return _start_server()

    cmd = args[0]
    if cmd in {"-h", "--help", "help"}:
        print(
            "Usage: deep-research-mcp [COMMAND] [OPTIONS]\n"
            "\n"
            "Commands:\n"
            "  (none)                       Start the MCP server over stdio.\n"
            "  init                         Interactive wizard to build a client config snippet.\n"
            "  doctor                       Print a non-destructive config self-check.\n"
            "  prune [--days N] [--dry-run] Delete checkpoints + registry rows for\n"
            "                               completed/failed tasks older than N days.\n"
            "                               Default N comes from\n"
            "                               DEEP_RESEARCH_CHECKPOINT_RETENTION_DAYS (=30).\n"
            "  help                         Show this message.\n"
        )
        return 0
    if cmd == "init":
        return run_init_wizard()
    if cmd == "doctor":
        return run_doctor()
    if cmd == "prune":
        return run_prune(args[1:])

    print(f"Unknown command: {cmd!r}. Try `deep-research-mcp --help`.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
