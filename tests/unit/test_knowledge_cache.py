"""Tests for KnowledgeCache deduplication semantics.

The cache is on the hot path of every research run — if dedup is too
aggressive we lose evidence; if it's too loose we waste LLM tokens digesting
the same fact many times.
"""

from __future__ import annotations


def _card(unit_id: str, claim: str, source: str, section: str = "S1") -> dict:
    return {
        "unit_id": unit_id,
        "claim": claim,
        "source": source,
        "section_id": section,
        "evidence_summary": "",
        "evidence_ids": [],
    }


def test_exact_duplicate_is_rejected():
    from deep_research_runtime.knowledge_cache import KnowledgeCache

    cache = KnowledgeCache()
    accepted_1 = cache.add_cards([_card("u1", "Claim A", "http://example.com")])
    accepted_2 = cache.add_cards([_card("u2", "Claim A", "http://example.com")])
    assert len(accepted_1) == 1
    assert accepted_2 == [], "Identical claim+source pair must dedup"


def test_claim_hash_is_case_insensitive():
    from deep_research_runtime.knowledge_cache import KnowledgeCache

    cache = KnowledgeCache()
    cache.add_cards([_card("u1", "Quantum supremacy", "http://x")])
    duped = cache.add_cards([_card("u2", "QUANTUM SUPREMACY", "http://x")])
    assert duped == [], "Same claim with different case should be deduped"


def test_different_source_is_kept():
    """Same claim from two different sources is corroborating evidence, not dup."""
    from deep_research_runtime.knowledge_cache import KnowledgeCache

    cache = KnowledgeCache()
    cache.add_cards([_card("u1", "Claim A", "http://source1.com")])
    accepted = cache.add_cards([_card("u2", "Claim A", "http://source2.com")])
    assert len(accepted) == 1


def test_section_index_groups_correctly():
    from deep_research_runtime.knowledge_cache import KnowledgeCache

    cache = KnowledgeCache()
    cache.add_cards([
        _card("u1", "Fact 1", "http://a", section="alpha"),
        _card("u2", "Fact 2", "http://b", section="alpha"),
        _card("u3", "Fact 3", "http://c", section="beta"),
    ])
    alpha = cache.get_cards_for_section("alpha")
    beta = cache.get_cards_for_section("beta")
    assert {c["unit_id"] for c in alpha} == {"u1", "u2"}
    assert {c["unit_id"] for c in beta} == {"u3"}


def test_coverage_stats_count_unique_urls():
    from deep_research_runtime.knowledge_cache import KnowledgeCache

    cache = KnowledgeCache()
    cache.add_cards([
        _card("u1", "F1", "http://a"),
        _card("u2", "F2", "http://a"),  # same URL, different claim
        _card("u3", "F3", "http://b"),
    ])
    stats = cache.get_coverage_stats()
    assert stats["total_cards"] == 3
    assert stats["unique_urls"] == 2
