"""BM25 retrieval model using rank_bm25."""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi


def _score_query(args):
    """Score a single query against the BM25 index (for parallel execution)."""
    bm25, query, top_k = args
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(-scores)[:top_k]
    top_scores = -np.sort(-scores)[:top_k]
    if len(top_indices) < top_k:
        padding_idx = np.full(top_k - len(top_indices), -1, dtype=np.int64)
        top_indices = np.concatenate([top_indices, padding_idx])
        padding_scores = np.zeros(top_k - len(top_scores), dtype=np.float32)
        top_scores = np.concatenate([top_scores, padding_scores])
    return top_indices, top_scores


class BM25Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.bm25 = None
        self.corpus_pubkeys = None

    @property
    def slug(self):
        return "bm25"

    def index_corpus(self, corpus_texts, corpus_pubkeys, cache_dir=None, batch_size=32):
        """Build BM25 index from tokenized corpus."""
        print("  Tokenizing corpus for BM25...")
        # Use map for faster tokenization (lazy generator → less memory overhead)
        tokenized_corpus = list(map(str.split, corpus_texts))
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_pubkeys = list(corpus_pubkeys)
        print(f"  BM25 index built ({len(corpus_pubkeys)} documents)")

    def retrieve(self, query_texts, top_k, cache_dir=None, lang=None, batch_size=32):
        """Score all documents for each query in parallel and return top-k indices and scores."""
        n_queries = len(query_texts)
        results = np.zeros((n_queries, top_k), dtype=np.int64)
        score_results = np.zeros((n_queries, top_k), dtype=np.float32)

        # Parallel scoring using threads (BM25 is CPU-bound but releases GIL
        # during numpy operations; even without that, parallelism helps on
        # multi-core systems)
        args_list = [(self.bm25, query, top_k) for query in query_texts]

        n_workers = min(16, len(query_texts))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i, (top_indices, top_scores) in enumerate(pool.map(_score_query, args_list)):
                results[i] = top_indices
                score_results[i] = top_scores
                if (i + 1) % max(1, n_queries // 10) == 0 or i + 1 == n_queries:
                    print(f"    BM25: {i + 1}/{n_queries} queries")

        return results, score_results
