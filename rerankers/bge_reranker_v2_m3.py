"""BGE reranker (BAAI/bge-reranker-v2-m3) wrapper for the eval pipeline."""
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from sentence_transformers import CrossEncoder


class BGERerankerV2M3:
    """
    Lightweight wrapper around BAAI/bge-reranker-v2-m3.

    Uses sentence-transformers CrossEncoder for faster loading
    (single safetensors read + efficient device placement).
    """

    MAX_LENGTH = 8192

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                first_gpu = cuda_visible.split(",")[0].strip()
                device = f"cuda:{first_gpu}"
            else:
                device = "cpu"
        self.device = device

        print(f"  Loading reranker: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device, max_length=self.MAX_LENGTH, model_kwargs={"torch_dtype": torch.bfloat16, "local_files_only": True})
        print(f"  Reranker loaded.")

    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: list[int],
        batch_size: int = 256,
        top_k: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        Rerank documents for a single query.

        Returns (reranked_doc_ids, scores) sorted descending by score.
        """
        pairs = [[query, doc] for doc in documents]

        all_scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False, max_length=self.MAX_LENGTH)
        if hasattr(all_scores, 'cpu'):
            all_scores = all_scores.cpu().numpy()
        else:
            all_scores = np.asarray(all_scores)

        # Sort descending by score
        sorted_idx = np.argsort(-all_scores)
        if top_k is not None:
            sorted_idx = sorted_idx[:top_k]

        return [doc_ids[j] for j in sorted_idx], [float(all_scores[j]) for j in sorted_idx]

    def rerank_queries(
        self,
        query_texts: list[str],
        doc_texts: list[str],
        faiss_indices,       # np.ndarray (n_queries, faiss_top_k)
        reranker_top_k: int,
        batch_size: int = 1, # queries processed at a time (each has many docs)
        query_batch_size: int = 10,  # number of queries whose pairs are batched into one GPU call
        return_scores: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """
        Rerank FAISS results for all queries using batched GPU inference.

        Flattens multiple queries' query-doc pairs into one large batch for
        efficient GPU utilisation, instead of processing one query at a time.

        Args:
            query_texts: list of query strings
            doc_texts: full corpus text list (indexed by doc idx)
            faiss_indices: np.ndarray shape (n_queries, faiss_top_k)
            reranker_top_k: how many to keep per query after reranking
            batch_size: (ignored, kept for API compat)
            query_batch_size: number of queries to batch together per GPU call
            return_scores: if True, also return reranker scores

        Returns:
            list of lists of reranked corpus indices;
            if return_scores=True, also returns list of lists of scores.
        """
        import sys

        n_queries = len(query_texts)
        faiss_top_k = faiss_indices.shape[1]
        all_reranked = []
        all_scores_out = [] if return_scores else None

        report_step = max(1, n_queries // 10)

        for i in range(0, n_queries, query_batch_size):
            end = min(i + query_batch_size, n_queries)

            # Flatten: collect all (query, doc) pairs for this batch of queries
            all_pairs = []
            for j in range(i, end):
                query = query_texts[j]
                retrieved_ids = faiss_indices[j].tolist()
                for idx in retrieved_ids:
                    all_pairs.append([query, doc_texts[idx]])

            # Single GPU call for all pairs in this chunk
            flat_scores = self.model.predict(
                all_pairs,
                batch_size=32,
                show_progress_bar=False,
                max_length=self.MAX_LENGTH,
            )
            if hasattr(flat_scores, 'cpu'):
                flat_scores = flat_scores.cpu().numpy()

            # Reshape flat scores into (n_queries_in_chunk, faiss_top_k) matrix
            scores_matrix = np.asarray(flat_scores, dtype=np.float32).reshape(end - i, faiss_top_k)

            # Per-query sort descending, keep top-k
            sorted_local = np.argsort(-scores_matrix, axis=1)[:, :reranker_top_k]

            for j in range(end - i):
                orig_ids = faiss_indices[i + j]
                reranked_ids = orig_ids[sorted_local[j]].tolist()
                all_reranked.append(reranked_ids)
                if return_scores:
                    all_scores_out.append(scores_matrix[j, sorted_local[j]].tolist())

            if end % report_step == 0 or end == n_queries:
                print(f"    Reranked {end}/{n_queries} queries", flush=True)

            sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_scores:
            return all_reranked, all_scores_out
        return all_reranked
