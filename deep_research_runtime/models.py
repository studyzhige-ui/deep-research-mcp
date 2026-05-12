import operator
from typing import Annotated, Any, Dict, List, TypedDict


def _merge_sub_tasks(left: List["SubTask"], right: List["SubTask"]) -> List["SubTask"]:
    """Reducer for sub_tasks: merge by (intent, query) key.

    When parallel section_researcher nodes return their own task updates,
    this reducer merges them into the canonical list, preferring completed
    or degraded status over pending.
    """
    if not right:
        return left
    if not left:
        return right
    index = {}
    for task in left:
        key = (task.get("intent", ""), task.get("query", ""))
        index[key] = task
    for task in right:
        key = (task.get("intent", ""), task.get("query", ""))
        existing = index.get(key)
        if existing is None:
            # New task not in left — append
            index[key] = task
        elif task.get("status") in ("completed", "degraded", "failed") and existing.get("status") == "pending":
            # Right has a resolved status — take it
            index[key] = task
    return list(index.values())


def _merge_section_results(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer for section_results: simple append-merge from parallel Send() nodes."""
    return (left or []) + (right or [])


class SubTask(TypedDict, total=False):
    query: str
    intent: str
    section_id: str
    section_title: str
    section_goal: str
    section_order: int
    gap_type: str
    search_goal: str
    evidence_goal: str
    required_source_types: List[str]
    required_evidence_types: List[str]
    rewritten_queries: List[str]
    time_scope: str
    search_profile: Dict[str, object]
    status: str  # "pending" | "completed" | "failed" | "degraded"
    error: str
    result_count: int
    searched_urls: List[str]
    # Error recovery fields
    retry_count: int
    last_error: str
    degradation_reason: str
    source_strategy: List[str]
    source_types: List[str]
    extraction_fields: List[str]


class SectionPlan(TypedDict, total=False):
    section_id: str
    title: str
    purpose: str
    priority: str
    questions: List[str]
    query_hints: List[str]
    depends_on: List[str]
    evidence_digest_ids: List[str]
    evidence_requirements: List[str]


class Document(TypedDict, total=False):
    document_id: str
    url: str
    title: str
    content: str
    raw_content: str
    source_name: str
    source_layer: str  # "general" | "vertical"
    source_kind: str  # "web" | "paper" | "pdf" | "repo" | "news" | "mcp" | "custom"
    published_time: str
    authors: List[str]
    venue: str
    year: int | None
    doi: str
    pdf_url: str
    metadata: Dict[str, Any]
    score: float


class ReconnaissanceResult(TypedDict, total=False):
    seed_queries: List[str]
    documents: List[Document]
    source_landscape: List[str]
    ambiguities: List[str]
    suggested_strategy: List[str]


class ResearchExecutionPlan(TypedDict, total=False):
    task_type: str
    user_goal: str
    scope: str
    source_strategy: List[Dict[str, Any]]
    query_strategy: List[Dict[str, Any]]
    screening_rules: List[str]
    extraction_schema: List[str]
    quality_rules: List[str]
    expected_deliverable: str
    output_language: str
    reconnaissance_summary: str


class EvidenceOutline(TypedDict, total=False):
    report_title: str
    user_goal: str
    sections: List[SectionPlan]
    outline_notes: str


class KnowledgeCard(TypedDict, total=False):
    unit_id: str
    section_id: str
    claim: str
    evidence_summary: str
    exact_excerpt: str
    evidence_id: str
    source: str
    source_title: str
    source_type: str
    claim_type: str
    time_scope: str
    entities: List[str]
    stance: str
    evidence_strength: str
    evidence_score: float
    confidence: str


class SectionDigestItem(TypedDict, total=False):
    item_id: str
    claim: str
    evidence_summary: str
    exact_excerpt: str
    confidence: str
    source_title: str
    source_url: str
    reference_numbers: List[int]


class SectionDigest(TypedDict, total=False):
    section_id: str
    title: str
    purpose: str
    questions: List[str]
    coverage_score: float
    evidence_count_score: float
    source_diversity_score: float
    is_enough: bool
    review_reason: str
    missing_questions: List[str]
    key_claims: List[str]
    items: List[SectionDigestItem]


class SectionResearchInput(TypedDict, total=False):
    """Payload sent to each section_researcher via Send().

    Contains only the data needed for one section's research,
    keeping the parallel nodes lightweight and isolated.
    """
    task_id: str
    topic: str
    section_id: str
    section_title: str
    pending_tasks: List[SubTask]
    existing_card_count: int


class ResearchState(TypedDict, total=False):
    task_id: str
    topic: str
    approved_plan: str
    plan_data: Dict[str, object]
    execution_plan: ResearchExecutionPlan
    reconnaissance: ReconnaissanceResult
    evidence_outline: EvidenceOutline
    user_feedback: str
    sub_tasks: Annotated[List[SubTask], _merge_sub_tasks]
    knowledge_cards: Annotated[List[KnowledgeCard], operator.add]
    section_digests: List[SectionDigest]
    conflicts: List[str]
    # Cross-source conflicts surfaced by conflict_detector.py.
    # Sparse map ``{section_id: [ConflictRecord, ...]}`` — sections without
    # detected conflicts are absent rather than mapped to an empty list, so
    # the writer's "if section in section_conflicts" check is meaningful.
    section_conflicts: Dict[str, List[Dict[str, Any]]]
    # Citation grounding audit produced by writer.py while it renders each
    # section via the structured-JSON path. Aggregates: total citations
    # emitted, citation IDs the LLM tried to invent (filtered out),
    # numeric / quote grounding failures, and a per-section breakdown.
    # When the writer falls back to the legacy free-form path for a
    # section (e.g. malformed LLM JSON), the per-section entry records
    # ``"writer_fallback": true`` so callers can see why audit signals
    # are missing for that section.
    citation_audit: Dict[str, Any]
    quality_review: Dict[str, object]
    loop_count: int
    route_to: str
    final_report: str
    # Adaptive loop / semantic saturation fields
    previous_coverage: float
    cards_before_loop: int
    saturation_score: float
    early_stop_reason: str
    # Section-level parallel research (Send API)
    section_results: Annotated[List[Dict[str, Any]], _merge_section_results]
