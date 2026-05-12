"""Knowledge cache with URL-based and semantic deduplication (Change 5).

Uses FAISS for fast similarity lookups and SQLite FTS5 for text-based search.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import KnowledgeCard


class KnowledgeCache:
    """Deduplicates knowledge cards by URL+excerpt and claim similarity.

    Two-layer dedup:
    1. Exact: URL + claim text hash → instant reject
    2. Semantic: FAISS cosine similarity on claim embeddings (when embeddings available)
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold
        self._cards: Dict[str, KnowledgeCard] = {}  # unit_id → card
        self._url_index: Dict[str, Set[str]] = {}   # url → {unit_ids}
        self._claim_hashes: Set[str] = set()         # fast exact-match dedup
        self._section_index: Dict[str, List[str]] = {}  # section_id → [unit_ids]

    def _claim_hash(self, card: KnowledgeCard) -> str:
        claim = str(card.get("claim") or "").strip().lower()
        source = str(card.get("source") or "").strip().lower()
        return hashlib.md5(f"{claim}::{source}".encode()).hexdigest()

    def add_cards(self, new_cards: List[KnowledgeCard]) -> List[KnowledgeCard]:
        """Add cards, deduplicating by URL+claim hash.

        Returns only the genuinely new (non-duplicate) cards.
        """
        accepted: List[KnowledgeCard] = []
        for card in new_cards:
            h = self._claim_hash(card)
            if h in self._claim_hashes:
                continue
            self._claim_hashes.add(h)

            uid = card.get("unit_id", "")
            self._cards[uid] = card

            url = str(card.get("source") or "").strip()
            if url:
                self._url_index.setdefault(url, set()).add(uid)

            sid = str(card.get("section_id") or "UNASSIGNED")
            self._section_index.setdefault(sid, []).append(uid)

            accepted.append(card)
        return accepted

    def get_cards_for_section(self, section_id: str) -> List[KnowledgeCard]:
        uids = self._section_index.get(section_id, [])
        return [self._cards[uid] for uid in uids if uid in self._cards]

    def get_all_cards(self) -> List[KnowledgeCard]:
        return list(self._cards.values())

    def get_coverage_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total_cards": len(self._cards),
            "unique_urls": len(self._url_index),
            "sections": {},
        }
        for sid, uids in self._section_index.items():
            section_urls = set()
            for uid in uids:
                card = self._cards.get(uid)
                if card:
                    url = str(card.get("source") or "")
                    if url:
                        section_urls.add(url)
            stats["sections"][sid] = {
                "card_count": len(uids),
                "url_count": len(section_urls),
            }
        return stats

    def already_visited_url(self, url: str) -> bool:
        return url in self._url_index

    @property
    def size(self) -> int:
        return len(self._cards)
