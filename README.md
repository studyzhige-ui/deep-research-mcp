# Deep Research MCP

Deep Research MCP is a Model Context Protocol server for autonomous, citation-driven research.

It turns a broad research topic into a planned workflow, searches for evidence, deduplicates sources, writes a structured Markdown report, and supports follow-up research against previous reports.

## Features

- MCP server over stdio for Claude Desktop, Claude Code, Cursor, Codex, and other MCP-compatible clients
- Plan-first workflow: draft a research plan before executing the full task
- Parallel section research with LangGraph
- LLM provider abstraction through LiteLLM
- Search integrations for Tavily, Exa, Serper, Bocha, Bing, Google Custom Search, SerpAPI, Searx, DuckDuckGo, Jina Reader, Semantic Scholar, arXiv, and PubMed Central
- Evidence normalization, deduplication, citation tracking, and report version comparison
- Local SQLite checkpoints and Markdown report output
- Optional local embedder and reranker support

## MCP Tools

The server exposes these tools:

| Tool | Purpose |
|---|---|
| `check_research_runtime` | Check whether the runtime and configuration are usable. |
| `draft_research_plan` | Create a research plan and return a task ID. |
| `start_research_task` | Start an approved research task. |
| `get_research_status` | Inspect progress and task state. |
| `get_research_result` | Return final report paths, preview, and version history. |
| `follow_up_research` | Extend a completed report with additional research. |
| `compare_report_versions` | Compare two report versions. |

## Installation

Clone the repository and install it into a Python environment:

```bash
git clone https://github.com/studyzhige-ui/deep-research-mcp.git
cd deep-research-mcp
pip install -e .
```

Optional extras:

```bash
# Local embedding / reranking worker
pip install -e ".[local-models]"

# DuckDuckGo fallback and PDF extraction
pip install -e ".[extras]"

# Development tools
pip install -e ".[dev]"
```

Verify the command is available:

```bash
deep-research-mcp doctor
```

## MCP Client Configuration

Add the server to your MCP client's configuration. The exact file location depends on the client, but the server entry has this shape:

```json
{
  "mcpServers": {
    "deep-research": {
      "command": "deep-research-mcp",
      "args": [],
      "env": {
        "DEEP_RESEARCH_LLM_MODEL": "deepseek/deepseek-chat",
        "DEEPSEEK_API_KEY": "YOUR_DEEPSEEK_API_KEY",
        "TAVILY_API_KEY": "YOUR_TAVILY_API_KEY"
      }
    }
  }
}
```

For TOML-based clients:

```toml
[mcp_servers.deep-research]
command = "deep-research-mcp"
args = []

[mcp_servers.deep-research.env]
DEEP_RESEARCH_LLM_MODEL = "deepseek/deepseek-chat"
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"
```

Restart the MCP client after changing its configuration.

## Configuration

Configuration is provided through environment variables. Do not commit API keys to this repository.

### LLM

| Variable | Description |
|---|---|
| `DEEP_RESEARCH_LLM_MODEL` | LiteLLM model identifier, such as `deepseek/deepseek-chat`, `openai/gpt-4.1-mini`, or `anthropic/claude-3-5-sonnet`. |
| `DEEP_RESEARCH_LLM_API_KEY` | Optional generic LLM API key. Provider-specific variables are also supported by LiteLLM. |
| `DEEP_RESEARCH_LLM_BASE_URL` | Optional custom API base URL. |
| `DEEP_RESEARCH_PLANNER_MODEL` | Optional model override for planning. |
| `DEEP_RESEARCH_RESEARCHER_MODEL` | Optional model override for research. |
| `DEEP_RESEARCH_WRITER_MODEL` | Optional model override for writing. |
| `DEEP_RESEARCH_REVIEWER_MODEL` | Optional model override for review. |

Provider-specific keys include `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and any other variable supported by LiteLLM for the selected model.

### Search

| Variable | Description |
|---|---|
| `TAVILY_API_KEY` | Primary web search provider. |
| `EXA_API_KEY` | Optional Exa neural search provider. |
| `SERPER_API_KEY` | Optional Serper search provider. |
| `BOCHA_API_KEY` | Optional Bocha search provider. |
| `JINA_API_KEY` | Optional Jina Reader key for page extraction. |
| `SERPAPI_API_KEY` | Optional SerpAPI provider. |
| `BING_API_KEY` | Optional Bing Web Search provider. |
| `GOOGLE_API_KEY` | Optional Google Custom Search API key. |
| `GOOGLE_CSE_ID` | Google Custom Search engine ID. |
| `SEARX_BASE_URL` | Optional Searx/SearxNG instance URL. |
| `DEEP_RESEARCH_MULTI_SOURCE` | Set to `1` to aggregate all configured search providers. |
| `DEEP_RESEARCH_DUCKDUCKGO_FALLBACK` | Set to `1` to use DuckDuckGo when available. |

### Academic Sources

| Variable | Description |
|---|---|
| `DEEP_RESEARCH_ACADEMIC_SEARCH` | Enable academic search routing. |
| `DEEP_RESEARCH_ARXIV_SEARCH` | Enable arXiv search. |
| `DEEP_RESEARCH_PUBMED_SEARCH` | Enable PubMed Central search. |

### Runtime

| Variable | Description |
|---|---|
| `DEEP_RESEARCH_REPORT_DIR` | Directory for generated reports and checkpoints. |
| `DEEP_RESEARCH_EMBEDDER_PATH` | Local embedding model path or model name. |
| `DEEP_RESEARCH_RERANKER_PATH` | Local reranker model path or model name. |
| `DEEP_RESEARCH_TASK_EXECUTION_TIMEOUT` | Maximum task runtime in seconds. |
| `DEEP_RESEARCH_DEBUG_TRACE` | Set to `1` for additional debug output. |

### Observability

| Variable | Description |
|---|---|
| `LANGSMITH_API_KEY` | Optional LangSmith tracing key. |
| `LANGCHAIN_PROJECT` | LangSmith project name. |

## Usage

Inside an MCP-compatible client, a typical workflow is:

1. Call `draft_research_plan` with a topic and background intent.
2. Review the proposed plan.
3. Call `start_research_task` with the returned task ID.
4. Poll `get_research_status`.
5. Call `get_research_result` when the task completes.
6. Use `follow_up_research` or `compare_report_versions` when needed.

Example user prompt:

```text
Use deep-research to draft a plan for researching recent progress in RISC-V adoption in data centers.
```

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

The main entry point is `deep_research_mcp.py`. Runtime code lives in `deep_research_runtime/`, and tests live in `tests/`.

## Security

- API keys must be provided by the user through environment variables or MCP client configuration.
- This repository does not include API keys, local credentials, private configuration files, generated checkpoints, or generated reports.
- The default MCP transport is stdio. Do not expose the server over a public network without adding an authentication layer.
- Generated reports and checkpoints may contain sensitive research content. Store `DEEP_RESEARCH_REPORT_DIR` accordingly.

## License

MIT
