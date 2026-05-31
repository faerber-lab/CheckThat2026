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


def prepare_corpus_texts(json_data_list):
    """
    Combine multiple fields from JSON documents into a single text for BM25 indexing.
    
    Args:
        json_data_list: List of dictionaries, each containing document fields
        
    Returns:
        List of combined text strings
    """
    combined_texts = []
    for doc in json_data_list:
        # Collect all relevant fields
        text_parts = [
            doc.get('title', ''),
            doc.get('abstract', ''),
            doc.get('venue', ''),
            doc.get('authors', '')
        ]
        # Join non-empty parts with spaces
        combined_text = ' '.join([part for part in text_parts if part.strip()])
        combined_texts.append(combined_text)
    return combined_texts


# Example usage:
if __name__ == "__main__":
    # Sample data (replace with your actual JSON loading)
    sample_docs = [
        {
            "pubkey": 9394,
            "title": "Hypericum perforatum and Its Ingredients Hypericin and Pseudohypericin Demonstrate an Antiviral Activity against SARS-CoV-2",
            "abstract": "For almost two years, the COVID-19 pandemic has constituted a major challenge to human health, particularly due to the lack of efficient antivirals to be used against the virus during routine treatment interventions. Multiple treatment options have been investigated for their potential inhibitory effect on SARS-CoV-2. Natural products, such as plant extracts, may be a promising option, as they have shown an antiviral activity against other viruses in the past. Here, a quantified extract of",
            "venue": "Pharmaceuticals",
            "authors": "Fakry F. Mohamed, Darisuren Anhlan, Michael Schöfbänker, Joachim Kühn, Eike R. Hrincius, Stephan Ludwig"
        },
        # Add more documents as needed
    ]
    
    # Prepare the corpus texts by combining all fields
    corpus_texts = prepare_corpus_texts(sample_docs)
    corpus_pubkeys = [doc['pubkey'] for doc in sample_docs]
    
    # Initialize and index the corpus
    model = BM25Model("bm25_all_fields")
    model.index_corpus(corpus_texts, corpus_pubkeys)
    
    # Example query
    queries = ["antiviral SARS-CoV-2 natural products"]
    results, scores = model.retrieve(queries, top_k=5)
    
    # Display results
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for j, (idx, score) in enumerate(zip(results[i], scores[i])):
            if idx != -1:
                print(f"  {j+1}. Pubkey: {model.corpus_pubkeys[idx]}, Score: {score:.4f}")