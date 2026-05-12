"""Centralized configuration for Deep Research Engine.

Supports:
- Multi-model via LiteLLM (per-role model assignment)
- Multi-source search (Tavily, Exa, Serper, Bocha + plugins)
- All settings configurable via environment variables
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _default_report_dir() -> str:
    return str(Path.home() / "Desktop" / "DeepResearch")


def _load_nanobot_config() -> Dict[str, Any]:
    config_path = Path.home() / ".nanobot" / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


_NANOBOT_CONFIG = _load_nanobot_config()


def _deepseek_api_key_fallback() -> str:
    providers = _NANOBOT_CONFIG.get("providers", {})
    deepseek = providers.get("deepseek", {}) if isinstance(providers, dict) else {}
    return str(deepseek.get("apiKey") or "")


def _deepseek_api_base_fallback() -> str:
    providers = _NANOBOT_CONFIG.get("providers", {})
    deepseek = providers.get("deepseek", {}) if isinstance(providers, dict) else {}
    return str(deepseek.get("apiBase") or "https://api.deepseek.com/v1")


def _mcp_server_env_fallback(key: str) -> str:
    tools = _NANOBOT_CONFIG.get("tools", {})
    if not isinstance(tools, dict):
        return ""
    mcp_servers = tools.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        return ""

    for server_name, server_config in mcp_servers.items():
        if not isinstance(server_config, dict):
            continue
        env = server_config.get("env", {})
        if not isinstance(env, dict):
            continue
        value = str(env.get(key) or "").strip()
        if not value:
            continue
        command = str(server_config.get("command") or "").lower()
        args = " ".join(str(item) for item in server_config.get("args", [])).lower() if isinstance(server_config.get("args"), list) else ""
        if server_name == "deep_research" or "deep_research" in command or "deep_research" in args:
            return value

    for server_config in mcp_servers.values():
        if not isinstance(server_config, dict):
            continue
        env = server_config.get("env", {})
        if not isinstance(env, dict):
            continue
        value = str(env.get(key) or "").strip()
        if value:
            return value
    return ""


# ──────────────────────────────────────────────────────────────────────
#  LiteLLM Model Name Reference (for users)
#
#  LiteLLM uses the format: "provider/model-name"
#  Below are the currently available mainstream models per provider.
#
#  ── DeepSeek ──
#    deepseek/deepseek-chat          # DeepSeek-V3 通用对话
#    deepseek/deepseek-reasoner      # DeepSeek-R1 推理模型
#    deepseek/deepseek-v4-flash      # DeepSeek-V4 Flash (当前默认)
#
#  ── OpenAI ──
#    openai/gpt-4.1                  # GPT-4.1 旗舰
#    openai/gpt-4.1-mini             # GPT-4.1 轻量
#    openai/gpt-4.1-nano             # GPT-4.1 最快
#    openai/o3                       # o3 推理模型
#    openai/o3-mini                  # o3 轻量推理
#    openai/o4-mini                  # o4 推理
#    openai/gpt-4o                   # GPT-4o 多模态
#    openai/gpt-4o-mini              # GPT-4o 轻量
#
#  ── Google Gemini ──
#    gemini/gemini-2.5-pro           # Gemini 2.5 Pro
#    gemini/gemini-2.5-flash         # Gemini 2.5 Flash (快速)
#    gemini/gemini-2.0-flash         # Gemini 2.0 Flash
#
#  ── Anthropic Claude ──
#    anthropic/claude-sonnet-4-20250514  # Claude Sonnet 4
#    anthropic/claude-3-7-sonnet     # Claude 3.7 Sonnet
#    anthropic/claude-3-5-haiku      # Claude 3.5 Haiku (快速)
#    anthropic/claude-3-5-sonnet     # Claude 3.5 Sonnet
#
#  ── MiniMax ──
#    minimax/MiniMax-Text-01         # MiniMax 文本旗舰
#    minimax/abab7-chat              # abab7 对话
#
#  ── 智谱 GLM ──
#    zhipu/glm-4-plus                # GLM-4 Plus
#    zhipu/glm-4-flash               # GLM-4 Flash (免费)
#    zhipu/glm-4-long                # GLM-4 长上下文
#    zhipu/glm-4                     # GLM-4 标准
#
#  ── 小米 MiLM ──
#    xiaomi/milm-1                   # MiLM-1 旗舰
#
#  Note: Set API keys via environment variables:
#    DEEPSEEK_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
#    ANTHROPIC_API_KEY, MINIMAX_API_KEY, ZHIPUAI_API_KEY, etc.
#    LiteLLM reads these automatically per provider.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Settings:
    # ── Identity ──
    mcp_name: str = os.environ.get("DEEP_RESEARCH_MCP_NAME", "LangGraph_DeepResearch_Engine")
    report_dir: str = os.environ.get("DEEP_RESEARCH_REPORT_DIR", _default_report_dir())

    # ── LangSmith Observability ──
    langsmith_api_key: str = os.environ.get("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.environ.get("LANGCHAIN_PROJECT", "DeepResearch")
    langsmith_endpoint: str = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_tracing: bool = os.environ.get("LANGSMITH_TRACING_V2", os.environ.get("LANGSMITH_TRACING", "0")) in {"1", "true", "True"}

    # ────────────────────────────────────────────────────
    #  LLM — Multi-Model via LiteLLM
    #
    #  llm_model: 默认模型 (所有未指定角色的调用使用此模型)
    #  planner_model / researcher_model / writer_model / reviewer_model:
    #    按角色分模型，留空则降级到 llm_model。
    #
    #  DeepSeek 需要设置 DEEPSEEK_API_KEY 和 api_base:
    #    LiteLLM 会自动读取 DEEPSEEK_API_KEY 环境变量。
    #    对于自定义 base_url，使用 llm_api_base 配置。
    # ────────────────────────────────────────────────────
    llm_api_key: str = os.environ.get("DEEP_RESEARCH_LLM_API_KEY", _mcp_server_env_fallback("DEEP_RESEARCH_LLM_API_KEY") or _deepseek_api_key_fallback())
    llm_api_base: str = os.environ.get("DEEP_RESEARCH_LLM_BASE_URL", _deepseek_api_base_fallback())
    llm_model: str = os.environ.get("DEEP_RESEARCH_LLM_MODEL", "deepseek/deepseek-v4-flash")

    # Per-role model overrides (leave empty to use llm_model)
    planner_model: str = os.environ.get("DEEP_RESEARCH_PLANNER_MODEL", "")
    researcher_model: str = os.environ.get("DEEP_RESEARCH_RESEARCHER_MODEL", "")
    writer_model: str = os.environ.get("DEEP_RESEARCH_WRITER_MODEL", "")
    reviewer_model: str = os.environ.get("DEEP_RESEARCH_REVIEWER_MODEL", "")

    # ────────────────────────────────────────────────────
    #  Search — Multi-Source Configuration
    #
    #  Layer 1: Search Discovery (找到相关 URL)
    #    Tavily  — AI 专用搜索，自带全文内容 (主力)
    #    Exa AI  — 神经网络语义搜索，深度研究 (补充)
    #    Serper  — Google 搜索 API，实时新闻 (补充)
    #    Bocha   — 中文语义排序，结构化卡片 (中文补充)
    #
    #  Layer 2: Content Extraction (把 URL 变成干净文本)
    #    Jina Reader — 处理 JS 渲染页面、反爬
    #    Readability — 本地 HTML 清洗 (已内置)
    #
    #  Layer 3: Vertical Plugins (垂直领域增强)
    #    可注册的插件系统，按 section 关键词自动触发
    #    内置: Academic (Semantic Scholar + arXiv)
    #
    #  enable_multi_source_search: 是否启用多源搜索
    #    false = 仅使用 Tavily (默认，对用户最简单)
    #    true  = 聚合所有已配置 API key 的搜索引擎
    # ────────────────────────────────────────────────────
    # API keys are read from environment variables first, then from the optional
    # ~/.nanobot/config.json file (kept for backward compatibility). They MUST NOT
    # be hardcoded here — distributing a public MCP server with embedded keys would
    # leak the maintainer's quota to anyone who installs the package.
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", _mcp_server_env_fallback("TAVILY_API_KEY"))
    exa_api_key: str = os.environ.get("EXA_API_KEY", _mcp_server_env_fallback("EXA_API_KEY"))
    serper_api_key: str = os.environ.get("SERPER_API_KEY", _mcp_server_env_fallback("SERPER_API_KEY"))
    bocha_api_key: str = os.environ.get("BOCHA_API_KEY", _mcp_server_env_fallback("BOCHA_API_KEY"))
    jina_api_key: str = os.environ.get("JINA_API_KEY", _mcp_server_env_fallback("JINA_API_KEY") or "")
    serpapi_api_key: str = os.environ.get("SERPAPI_API_KEY", _mcp_server_env_fallback("SERPAPI_API_KEY") or "")
    bing_api_key: str = os.environ.get("BING_API_KEY", _mcp_server_env_fallback("BING_API_KEY") or "")
    google_api_key: str = os.environ.get("GOOGLE_API_KEY", _mcp_server_env_fallback("GOOGLE_API_KEY") or "")
    google_cse_id: str = os.environ.get("GOOGLE_CSE_ID", _mcp_server_env_fallback("GOOGLE_CSE_ID") or "")
    searx_base_url: str = os.environ.get("SEARX_BASE_URL", _mcp_server_env_fallback("SEARX_BASE_URL") or "")

    # Multi-source toggle (user decides whether to use multiple search APIs)
    enable_multi_source_search: bool = os.environ.get("DEEP_RESEARCH_MULTI_SOURCE", "0") in {"1", "true", "True"}
    enable_duckduckgo_fallback: bool = os.environ.get("DEEP_RESEARCH_DUCKDUCKGO_FALLBACK", "1") in {"1", "true", "True"}
    use_jina_reader_without_key: bool = os.environ.get("DEEP_RESEARCH_USE_JINA_WITHOUT_KEY", "1") in {"1", "true", "True"}
    draft_max_queries: int = int(os.environ.get("DEEP_RESEARCH_DRAFT_MAX_QUERIES", "3"))
    draft_max_results_per_query: int = int(os.environ.get("DEEP_RESEARCH_DRAFT_MAX_RESULTS_PER_QUERY", "3"))

    # Vertical plugin toggles
    enable_academic_search: bool = os.environ.get("DEEP_RESEARCH_ACADEMIC_SEARCH", "0") in {"1", "true", "True"}
    enable_arxiv_search: bool = os.environ.get("DEEP_RESEARCH_ARXIV_SEARCH", "0") in {"1", "true", "True"}
    enable_pubmed_search: bool = os.environ.get("DEEP_RESEARCH_PUBMED_SEARCH", "0") in {"1", "true", "True"}
    enable_serpapi_search: bool = os.environ.get("DEEP_RESEARCH_SERPAPI_SEARCH", "0") in {"1", "true", "True"}
    enable_bing_search: bool = os.environ.get("DEEP_RESEARCH_BING_SEARCH", "0") in {"1", "true", "True"}
    enable_google_search: bool = os.environ.get("DEEP_RESEARCH_GOOGLE_SEARCH", "0") in {"1", "true", "True"}
    enable_searx_search: bool = os.environ.get("DEEP_RESEARCH_SEARX_SEARCH", "0") in {"1", "true", "True"}

    # ── Local Models ──
    # Default location is cross-platform. Override with DEEP_RESEARCH_EMBEDDER_PATH /
    # DEEP_RESEARCH_RERANKER_PATH if you have local model weights elsewhere.
    # When the directory does not exist, sentence-transformers will try to download
    # the model by name from Hugging Face on first run.
    embedder_path: str = os.environ.get(
        "DEEP_RESEARCH_EMBEDDER_PATH",
        str(Path.home() / ".cache" / "deep_research" / "models" / "bge-small-zh-v1.5"),
    )
    reranker_path: str = os.environ.get(
        "DEEP_RESEARCH_RERANKER_PATH",
        str(Path.home() / ".cache" / "deep_research" / "models" / "bge-reranker-base"),
    )

    # ── Search Tuning ──
    search_concurrency: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_CONCURRENCY", "5"))
    search_max_results: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_MAX_RESULTS", "5"))
    search_rewrite_query_count: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_REWRITE_QUERY_COUNT", "4"))
    max_queries_per_intent: int = int(os.environ.get("DEEP_RESEARCH_MAX_QUERIES_PER_INTENT", "3"))

    # ── Adaptive Search Profiles (per section priority) ──
    search_profile_high_max_results: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_HIGH_MAX_RESULTS", "8"))
    search_profile_high_max_queries: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_HIGH_MAX_QUERIES", "5"))
    search_profile_high_rewrite_count: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_HIGH_REWRITE_COUNT", "6"))
    search_profile_medium_max_results: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_MEDIUM_MAX_RESULTS", "5"))
    search_profile_medium_max_queries: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_MEDIUM_MAX_QUERIES", "3"))
    search_profile_medium_rewrite_count: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_MEDIUM_REWRITE_COUNT", "4"))
    search_profile_low_max_results: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_LOW_MAX_RESULTS", "3"))
    search_profile_low_max_queries: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_LOW_MAX_QUERIES", "2"))
    search_profile_low_rewrite_count: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_PROFILE_LOW_REWRITE_COUNT", "2"))

    # ── Reflector / Loop Control ──
    reflector_review_concurrency: int = int(os.environ.get("DEEP_RESEARCH_REFLECTOR_REVIEW_CONCURRENCY", "3"))
    reflector_review_stagger_seconds: float = float(os.environ.get("DEEP_RESEARCH_REFLECTOR_REVIEW_STAGGER_SECONDS", "3"))
    max_reflection_loops: int = int(os.environ.get("DEEP_RESEARCH_MAX_REFLECTION_LOOPS", "3"))

    # ── Adaptive Loop / Semantic Saturation ──
    saturation_threshold: float = float(os.environ.get("DEEP_RESEARCH_SATURATION_THRESHOLD", "0.85"))
    min_loops_before_early_stop: int = int(os.environ.get("DEEP_RESEARCH_MIN_LOOPS", "1"))

    # ── Page Cleaning ──
    page_cleaning_concurrency: int = int(os.environ.get("DEEP_RESEARCH_PAGE_CLEANING_CONCURRENCY", "4"))
    page_fetch_timeout: int = int(os.environ.get("DEEP_RESEARCH_PAGE_FETCH_TIMEOUT", "20"))
    page_cleaned_max_chars: int = int(os.environ.get("DEEP_RESEARCH_PAGE_CLEANED_MAX_CHARS", "20000"))

    # ── Worker ──
    worker_start_timeout: int = int(os.environ.get("DEEP_RESEARCH_WORKER_START_TIMEOUT", "180"))
    worker_result_timeout: int = int(os.environ.get("DEEP_RESEARCH_WORKER_RESULT_TIMEOUT", "600"))
    worker_job_retry_attempts: int = int(os.environ.get("DEEP_RESEARCH_WORKER_JOB_RETRY_ATTEMPTS", "2"))
    worker_heartbeat_interval: int = int(os.environ.get("DEEP_RESEARCH_WORKER_HEARTBEAT_INTERVAL", "5"))
    worker_stale_after: int = int(os.environ.get("DEEP_RESEARCH_WORKER_STALE_AFTER", "300"))
    # Maintenance: retain at most this many days of completed/failed task
    # checkpoints. The pruner is opt-in (CLI command), never automatic, so
    # this only affects users who run `deep-research-mcp prune`.
    checkpoint_retention_days: int = int(os.environ.get("DEEP_RESEARCH_CHECKPOINT_RETENTION_DAYS", "30"))

    # ── Grounded citation generation (Improvement ④) ──
    # When enabled, the writer's per-section call requests STRUCTURED JSON
    # output where each paragraph names which evidence_ids back it. The
    # closed-set evidence catalog is passed in the prompt, so the LLM can't
    # invent citation numbers — they're filtered out at parse time. Failed
    # JSON parses fall back to the legacy free-form prompt path; we never
    # block report generation on a writer that misbehaves.
    enable_grounded_citations: bool = os.environ.get("DEEP_RESEARCH_GROUNDED_CITATIONS", "1") in {"1", "true", "True"}

    # ── Cross-source conflict detection (Improvement ④, prior commit) ──
    # Detects when two cards in the same section make incompatible claims
    # and instructs the writer to surface the disagreement instead of
    # silently picking one. Costs one LLM call per section that has at
    # least one candidate pair; sections without disagreement signal
    # never trigger a call.
    enable_conflict_detection: bool = os.environ.get("DEEP_RESEARCH_CONFLICT_DETECTION", "1") in {"1", "true", "True"}
    conflict_min_cards: int = int(os.environ.get("DEEP_RESEARCH_CONFLICT_MIN_CARDS", "3"))
    conflict_max_pairs_per_section: int = int(os.environ.get("DEEP_RESEARCH_CONFLICT_MAX_PAIRS_PER_SECTION", "6"))

    # ── Recency weighting (Improvement ③) ──
    # When a task's time_scope marks it as time-sensitive, multiply each
    # document's quality score by an age-decay factor. Older sources still
    # rank against the floor (0.3) so they aren't excluded outright — they
    # just no longer outweigh a fresh primary source by virtue of having
    # more body text.
    recency_weighting_enabled: bool = os.environ.get("DEEP_RESEARCH_RECENCY_WEIGHTING", "1") in {"1", "true", "True"}
    recency_half_life_recent_months: int = int(os.environ.get("DEEP_RESEARCH_RECENCY_HALF_LIFE_RECENT_MONTHS", "6"))
    recency_half_life_current_months: int = int(os.environ.get("DEEP_RESEARCH_RECENCY_HALF_LIFE_CURRENT_MONTHS", "18"))
    recency_half_life_default_months: int = int(os.environ.get("DEEP_RESEARCH_RECENCY_HALF_LIFE_DEFAULT_MONTHS", "36"))
    # Bound the in-memory mp.Queue feeding the worker so a runaway producer
    # cannot OOM the process. 0 disables the limit (legacy behavior). The
    # default of 64 is enough for ~tens of concurrent research tasks while
    # still capping memory at ~tens of MB of queued embedding inputs.
    worker_job_queue_max_size: int = int(os.environ.get("DEEP_RESEARCH_WORKER_QUEUE_MAX_SIZE", "64"))
    # Restart cooldown — if the worker process dies repeatedly within a
    # short window we stop trying instead of looping forever.
    worker_max_restart_attempts: int = int(os.environ.get("DEEP_RESEARCH_WORKER_MAX_RESTARTS", "5"))
    worker_restart_window_sec: int = int(os.environ.get("DEEP_RESEARCH_WORKER_RESTART_WINDOW_SEC", "300"))

    # ── Error Recovery ──
    search_retry_attempts: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_RETRY_ATTEMPTS", "2"))
    # Circuit breaker: after this many consecutive failures, skip the retriever
    # for `search_circuit_breaker_cooldown_sec` seconds so a flaky engine
    # doesn't drag down every other research task. 0 disables the breaker.
    search_circuit_breaker_threshold: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_CB_THRESHOLD", "3"))
    search_circuit_breaker_cooldown_sec: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_CB_COOLDOWN_SEC", "60"))
    # Per-engine rate limit. The semaphore is created at retriever-construction
    # time and limits in-flight requests to the same engine. 0 = unlimited.
    search_engine_rate_limit: int = int(os.environ.get("DEEP_RESEARCH_SEARCH_RATE_LIMIT", "4"))
    search_retry_backoff_base: float = float(os.environ.get("DEEP_RESEARCH_SEARCH_RETRY_BACKOFF_BASE", "1.5"))
    enable_query_reformulation: bool = os.environ.get("DEEP_RESEARCH_ENABLE_QUERY_REFORMULATION", "1") in {"1", "true", "True"}
    # Number of strategy-diverse reformulations to request in a single LLM
    # call when a query returns zero results. Capped at the number of
    # strategies defined in query_reform.py (currently 5). The default of 3
    # covers SIMPLIFY/SYNONYMS/DECOMPOSE — the three highest-recall paths.
    max_reformulation_attempts: int = int(os.environ.get("DEEP_RESEARCH_MAX_REFORMULATION_ATTEMPTS", "3"))

    # ── Task ──
    task_execution_timeout: int = int(os.environ.get("DEEP_RESEARCH_TASK_EXECUTION_TIMEOUT", "5400"))
    debug_trace: bool = os.environ.get("DEEP_RESEARCH_DEBUG_TRACE", "0") in {"1", "true", "True"}

    # ── Writer ──
    section_digest_max_cards_per_section: int = int(os.environ.get("DEEP_RESEARCH_SECTION_DIGEST_MAX_CARDS_PER_SECTION", "24"))
    section_digest_excerpt_chars: int = int(os.environ.get("DEEP_RESEARCH_SECTION_DIGEST_EXCERPT_CHARS", "180"))
    writer_max_cards_per_section: int = int(os.environ.get("DEEP_RESEARCH_WRITER_MAX_CARDS_PER_SECTION", "8"))
    writer_raw_cards_per_section: int = int(os.environ.get("DEEP_RESEARCH_WRITER_RAW_CARDS_PER_SECTION", "6"))
    writer_max_quotes_per_section: int = int(os.environ.get("DEEP_RESEARCH_WRITER_MAX_QUOTES_PER_SECTION", "3"))
    writer_outline_summary_chars: int = int(os.environ.get("DEEP_RESEARCH_WRITER_OUTLINE_SUMMARY_CHARS", "320"))
    writer_excerpt_chars: int = int(os.environ.get("DEEP_RESEARCH_WRITER_EXCERPT_CHARS", "220"))
    enable_citation_verification: bool = os.environ.get("DEEP_RESEARCH_ENABLE_CITATION_VERIFICATION", "1") in {"1", "true", "True"}

    # ── Knowledge Cache ──
    knowledge_cache_similarity_threshold: float = float(os.environ.get("DEEP_RESEARCH_KNOWLEDGE_CACHE_SIMILARITY", "0.85"))

    # ── Helper Methods ──

    def get_model_for_role(self, role: str) -> str:
        """Return the LiteLLM model string for a given agent role.

        Falls back to llm_model if no role-specific model is configured.
        Supported roles: planner, researcher, writer, reviewer.
        """
        role_attr = f"{role}_model"
        role_model = getattr(self, role_attr, "")
        return role_model or self.llm_model

    def get_search_profile(self, priority: str) -> Dict[str, int]:
        """Return search parameters scaled by section priority."""
        profiles = {
            "high": {
                "max_results": self.search_profile_high_max_results,
                "max_queries": self.search_profile_high_max_queries,
                "rewrite_count": self.search_profile_high_rewrite_count,
            },
            "medium": {
                "max_results": self.search_profile_medium_max_results,
                "max_queries": self.search_profile_medium_max_queries,
                "rewrite_count": self.search_profile_medium_rewrite_count,
            },
            "low": {
                "max_results": self.search_profile_low_max_results,
                "max_queries": self.search_profile_low_max_queries,
                "rewrite_count": self.search_profile_low_rewrite_count,
            },
        }
        return profiles.get(priority, profiles["medium"])

    def get_active_search_engines(self) -> List[str]:
        """Return list of search engines with configured API keys.

        When enable_multi_source_search is False, only returns ['tavily'].
        When True, returns all engines that have a valid API key.
        """
        if not self.enable_multi_source_search:
            return ["tavily"] if self.tavily_api_key else []

        engines = []
        if self.tavily_api_key:
            engines.append("tavily")
        if self.exa_api_key:
            engines.append("exa")
        if self.serper_api_key:
            engines.append("serper")
        if self.bocha_api_key:
            engines.append("bocha")
        if self.enable_serpapi_search and self.serpapi_api_key:
            engines.append("serpapi")
        if self.enable_bing_search and self.bing_api_key:
            engines.append("bing")
        if self.enable_google_search and self.google_api_key and self.google_cse_id:
            engines.append("google")
        if self.enable_searx_search and self.searx_base_url:
            engines.append("searx")
        return engines

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ──────────────────────────────────────────────────────────────────────
    #  Configuration self-check
    #
    #  Called once at startup to surface configuration problems early, before
    #  the user submits their first research task. Prints a human-readable
    #  summary to stderr and returns the same data structurally so that
    #  programmatic callers (init wizard, tests) can inspect it.
    # ──────────────────────────────────────────────────────────────────────

    def validate_and_report(self, *, emit: bool = True) -> Dict[str, Any]:
        """Inspect the loaded settings and report which features are available.

        Returns a dict with the following shape::

            {
                "llm": {"configured": bool, "model": str, "via": str},
                "search_engines": {"active": [...], "skipped": {name: reason}},
                "vertical": {"academic": bool, "arxiv": bool, "pubmed": bool},
                "langsmith": bool,
                "blocking_errors": [str, ...],
                "warnings": [str, ...],
            }

        ``blocking_errors`` is non-empty when the server cannot reasonably run
        (e.g. no LLM key and no search engines). Callers may choose to abort.
        """
        report: Dict[str, Any] = {
            "llm": {},
            "search_engines": {"active": [], "skipped": {}},
            "vertical": {},
            "langsmith": bool(self.langsmith_api_key),
            "blocking_errors": [],
            "warnings": [],
        }

        # LLM
        # LiteLLM also reads provider-specific env vars (DEEPSEEK_API_KEY, OPENAI_API_KEY...)
        # so a missing llm_api_key isn't necessarily fatal — flag it as a warning instead.
        llm_via = "DEEP_RESEARCH_LLM_API_KEY" if self.llm_api_key else "(none — relying on LiteLLM provider env var)"
        report["llm"] = {
            "configured": bool(self.llm_api_key),
            "model": self.llm_model,
            "via": llm_via,
        }
        if not self.llm_api_key:
            report["warnings"].append(
                "DEEP_RESEARCH_LLM_API_KEY not set — LiteLLM will fall back to provider env vars "
                "(e.g. DEEPSEEK_API_KEY, OPENAI_API_KEY). Ensure one of them is configured."
            )

        # Search engines
        active = self.get_active_search_engines()
        report["search_engines"]["active"] = active

        skipped: Dict[str, str] = {}
        if not self.tavily_api_key:
            skipped["tavily"] = "TAVILY_API_KEY not set"
        if self.enable_multi_source_search:
            if not self.exa_api_key:
                skipped["exa"] = "EXA_API_KEY not set"
            if not self.serper_api_key:
                skipped["serper"] = "SERPER_API_KEY not set"
            if not self.bocha_api_key:
                skipped["bocha"] = "BOCHA_API_KEY not set"
        report["search_engines"]["skipped"] = skipped

        if not active and not self.enable_duckduckgo_fallback:
            report["blocking_errors"].append(
                "No search engine is available. Set TAVILY_API_KEY (recommended) or "
                "enable DEEP_RESEARCH_DUCKDUCKGO_FALLBACK=1 as a free fallback."
            )

        # Vertical sources
        report["vertical"] = {
            "academic": self.enable_academic_search,
            "arxiv": self.enable_arxiv_search,
            "pubmed": self.enable_pubmed_search,
        }

        if emit:
            self._emit_validation_report(report)

        return report

    @staticmethod
    def _emit_validation_report(report: Dict[str, Any]) -> None:
        import sys as _sys

        def _line(msg: str) -> None:
            print(f"[deep-research] {msg}", file=_sys.stderr, flush=True)

        llm = report["llm"]
        marker = "OK" if llm["configured"] else "WARN"
        _line(f"[{marker}] LLM model={llm['model']} (key via {llm['via']})")

        active = report["search_engines"]["active"]
        if active:
            _line(f"[OK] Search engines active: {', '.join(active)}")
        else:
            _line("[WARN] No primary search engines configured.")

        skipped = report["search_engines"]["skipped"]
        for name, reason in skipped.items():
            _line(f"[SKIP] {name}: {reason}")

        vertical_on = [k for k, v in report["vertical"].items() if v]
        if vertical_on:
            _line(f"[OK] Vertical search enabled: {', '.join(vertical_on)}")

        if report["langsmith"]:
            _line("[OK] LangSmith tracing enabled")

        for warn in report["warnings"]:
            _line(f"[WARN] {warn}")
        for err in report["blocking_errors"]:
            _line(f"[ERROR] {err}")
