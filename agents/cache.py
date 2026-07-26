import hashlib
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def derive_cache_key(ticker: str, forecast_hash: str, news_hash: str, model_version: str = "v1") -> str:
    """Derive deterministic SHA256 key for Tier 1 exact hit cache."""
    raw = f"{ticker}|{forecast_hash}|{news_hash}|{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class DualCache:
    """
    Two-Tiered Cache Architecture.
    
    Tier 1 (Exact Hit): Hash-based key lookup.
    Tier 2 (Semantic Similarity): News digest similarity lookup for narrative framing reuse.
    """
    
    def __init__(self):
        self._exact_store: Dict[str, Dict[str, Any]] = {}
        self._news_vector_store: Dict[str, Tuple[str, Dict[str, Any]]] = {} # hash -> (text, data)

    def get_exact(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Tier 1 lookup."""
        return self._exact_store.get(cache_key)

    def set_exact(self, cache_key: str, payload: Dict[str, Any]):
        """Tier 1 store."""
        self._exact_store[cache_key] = payload

    def find_similar_news_framing(self, news_digest: str, similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """
        Tier 2 lookup: Check if news digest is semantically similar to historical news.
        Simple Jaccard/bag-of-words similarity for fast fallback.
        """
        words_input = set(news_digest.lower().split())
        if not words_input:
            return None
            
        for news_hash, (stored_news, stored_data) in self._news_vector_store.items():
            words_stored = set(stored_news.lower().split())
            intersection = words_input.intersection(words_stored)
            union = words_input.union(words_stored)
            jaccard_sim = len(intersection) / max(len(union), 1)
            
            if jaccard_sim >= similarity_threshold:
                logger.info(f"DualCache Tier 2: Found similar news environment (sim={jaccard_sim:.2f}). Reusing narrative framing.")
                return stored_data
                
        return None

    def store_news_framing(self, news_digest: str, report_data: Dict[str, Any]):
        """Tier 2 store."""
        news_hash = hashlib.sha256(news_digest.encode("utf-8")).hexdigest()
        self._news_vector_store[news_hash] = (news_digest, report_data)
