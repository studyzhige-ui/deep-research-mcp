# Deep Research MCP

An autonomous deep-research engine packaged as a [Model Context Protocol](https://modelcontextprotocol.io) server.
Give it a topic; it plans, searches, deduplicates evidence, writes a structured Markdown report with verifiable citations, and remembers enough to do incremental follow-up research later.

Built on **LangGraph** (map-reduce parallel research), **LiteLLM** (any LLM provider), and **FastMCP** (works with Claude Desktop, Claude Code, Cursor, Codex, and any MCP-compatible host).

---

## 5-minute quickstart

### 1. Install

```bash
# From a clone of this repository
pip install -e .

# Optional: local embedder + reranker (used for evidence dedup/rerank).
# Skip this if you don't have a GPU or don't want the worker subprocess.
pip install -e ".[local-models]"

# Optional: DuckDuckGo fallback + PDF extraction
pip install -e ".[extras]"
```

### 2. Generate your client config

Run the interactive wizard — it asks for your API keys and prints a JSON/TOML snippet ready to paste into your MCP client config. **Keys are never written to disk by the wizard**, only shown on screen.

```bash
deep-research-mcp init
```

You will be asked for:

| Field | Required? | Where to get it |
|---|---|---|
| LLM provider + API key | yes | DeepSeek / OpenAI / Anthropic / Gemini etc. |
| Tavily API key | strongly recommended | https://tavily.com (free tier 1000/mo) |
| Exa / Serper / Bocha | optional | for richer multi-source search |
| LangSmith API key | optional | for tracing |

### 3. Wire the server into your MCP client

Paste the snippet the wizard produced into your client config. Example for **Claude Desktop / Claude Code** (`~/.claude.json` or platform-specific path):

```json
{
  "mcpServers": {
    "deep-research": {
      "command": "deep-research-mcp",
      "env": {
        "DEEPSEEK_API_KEY": "YOUR_DEEPSEEK_API_KEY",
        "DEEP_RESEARCH_LLM_MODEL": "deepseek/deepseek-v4-flash",
        "TAVILY_API_KEY": "YOUR_TAVILY_API_KEY"
      }
    }
  }
}
```

For **Codex** (`~/.codex/config.toml`):

```toml
[mcp_servers.deep-research]
command = "deep-research-mcp"
env = { DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY", TAVILY_API_KEY = "YOUR_TAVILY_API_KEY" }
```

For **Cursor** (`~/.cursor/mcp.json`): same JSON shape as Claude.

Restart your MCP client. The seven `deep-research` tools should appear in the tool picker.

### 4. Verify

```bash
# Confirm your env vars are picked up correctly (no server start, no LLM calls).
deep-research-mcp doctor
```

You should see a summary like:

```
[deep-research] [OK] LLM model=deepseek/deepseek-v4-flash (key via DEEP_RESEARCH_LLM_API_KEY)
[deep-research] [OK] Search engines active: tavily
[deep-research] [SKIP] exa: EXA_API_KEY not set
```

---

## How users interact with it

From inside any MCP-compatible chat client:

1. **Draft a plan**
   `draft_research_plan(topic, background_intent)` → returns a task id + proposed outline.
2. **Approve and run**
   `start_research_task(task_id)` → kicks off background execution.
3. **Check progress**
   `get_research_status(task_id)` → real-time timeline of section_start / section_done / retry events.
4. **Get the report**
   `get_research_result(task_id)` → final Markdown path + preview + version history.
5. **Iterate**
   `follow_up_research(task_id, question)` → incremental research on a completed report.
6. **Compare**
   `compare_report_versions(task_id, va, vb)` → section-by-section diff.

---

## Configuration reference

All settings are read from environment variables — pass them via your MCP client's `env` field.

### Required

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / etc. | LLM provider key. LiteLLM auto-detects based on `DEEP_RESEARCH_LLM_MODEL`. |
| `TAVILY_API_KEY` | Primary search engine. Without this you'll fall back to DuckDuckGo only. |

### LLM tuning

| Variable | Default | Description |
|---|---|---|
| `DEEP_RESEARCH_LLM_MODEL` | `deepseek/deepseek-v4-flash` | LiteLLM model identifier. |
| `DEEP_RESEARCH_PLANNER_MODEL` | (same as above) | Override per agent role. |
| `DEEP_RESEARCH_RESEARCHER_MODEL` | (same) | |
| `DEEP_RESEARCH_WRITER_MODEL` | (same) | |
| `DEEP_RESEARCH_REVIEWER_MODEL` | (same) | |

### Search

| Variable | Default | Description |
|---|---|---|
| `TAVILY_API_KEY` | — | Primary engine. |
| `EXA_API_KEY` | — | Neural semantic search. |
| `SERPER_API_KEY` | — | Google Search wrapper. |
| `BOCHA_API_KEY` | — | Chinese-language semantic search. |
| `JINA_API_KEY` | — | Jina Reader for JS-heavy pages (works without key in degraded mode). |
| `DEEP_RESEARCH_MULTI_SOURCE` | `0` | Set to `1` to aggregate every configured engine instead of Tavily-only. |
| `DEEP_RESEARCH_DUCKDUCKGO_FALLBACK` | `1` | Free fallback when nothing else is configured. |

### Vertical sources

| Variable | Default | Description |
|---|---|---|
| `DEEP_RESEARCH_ACADEMIC_SEARCH` | `0` | Enable Semantic Scholar. |
| `DEEP_RESEARCH_ARXIV_SEARCH` | `0` | Enable arXiv (requires academic search on). |
| `DEEP_RESEARCH_PUBMED_SEARCH` | `0` | Enable PubMed (requires academic search on). |

### Observability (optional)

| Variable | Default | Description |
|---|---|---|
| `LANGSMITH_API_KEY` | — | Enables LangSmith tracing. |
| `LANGCHAIN_PROJECT` | `DeepResearch` | Project name in the LangSmith dashboard. |

### Output & runtime

| Variable | Default | Description |
|---|---|---|
| `DEEP_RESEARCH_REPORT_DIR` | `~/Desktop/DeepResearch` | Where final reports are saved. |
| `DEEP_RESEARCH_EMBEDDER_PATH` | `~/.cache/deep_research/models/bge-small-zh-v1.5` | Local embedder. Downloaded on first use if absent. |
| `DEEP_RESEARCH_RERANKER_PATH` | `~/.cache/deep_research/models/bge-reranker-base` | Local reranker. |

---

## Architecture (one paragraph)

The server is a LangGraph state machine: `supervisor → dispatch_sections → section_researcher × N → collect_results → reflector → outline_builder → writer`. Each section runs in parallel via LangGraph's `Send()` API. Evidence collected by researchers is stored as `KnowledgeCard` objects, deduplicated both by URL (exact) and by claim similarity (semantic, threshold 0.85). The reflector decides whether to loop for more evidence or stop based on a saturation metric. Output is structured Markdown with numbered citations linking back to the source catalog. Full architecture details live in [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Server starts but reports `No search engine is available` | No keys set, DuckDuckGo disabled | Set `TAVILY_API_KEY` or `DEEP_RESEARCH_DUCKDUCKGO_FALLBACK=1` |
| `Model worker failed to start` | Missing `sentence-transformers` / `torch` | `pip install -e ".[local-models]"` |
| LLM calls fail with "API key not found" | Provider key not set in client `env` | Add `DEEPSEEK_API_KEY` (or your provider's key) to client config |
| Reports save somewhere unexpected | `DEEP_RESEARCH_REPORT_DIR` defaults to `~/Desktop/DeepResearch` | Override with the env var |

Run `deep-research-mcp doctor` whenever something looks off — it prints exactly which engines are active and why others were skipped.

---

## Security notes

- **API keys live in the user's MCP client config, never in this repository.** The maintainers ship no embedded keys; if a fork ever does, treat them as already-leaked.
- The MCP server uses stdio transport by default — no network listener is exposed. Don't run it behind an HTTP transport on a public network without adding authentication first.
- Reports and checkpoints are stored as plaintext under `DEEP_RESEARCH_REPORT_DIR`. Don't point this at a synced folder if your research topics are sensitive.

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

See [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) for architecture deep-dives and [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md) for design rationale.
