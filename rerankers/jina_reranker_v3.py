"""Jina Reranker v3 (jinaai/jina-reranker-v3) wrapper for the eval pipeline."""
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.enable_cudnn_sdp(False)
from transformers import AutoModel


class JinaRerankerV3:
    """
    Wrapper around jinaai/jina-reranker-v3.

    Listwise document reranker (0.6B params) based on Qwen3-0.6B with a
    last-but-not-late interaction architecture. Uses the model's built-in
    rerank() method. Multilingual, supports up to 64 documents per query.
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v3", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                first_gpu = cuda_visible.split(",")[0].strip()
                device = f"cuda:{first_gpu}"
            else:
                device = "cpu"
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

        print(f"  Loading reranker: {model_name} on {device}")
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.is_cuda else torch.float32,
            "trust_remote_code": True,
            "local_files_only": True,
            "device_map": device if self.is_cuda else None,
        }
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs,
        ).eval()
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
        results = self.model.rerank(query, documents)

        # results is already sorted by relevance_score descending
        if top_k is not None:
            results = results[:top_k]

        reranked_ids = [doc_ids[r["index"]] for r in results]
        scores = [r["relevance_score"] for r in results]
        return reranked_ids, scores

    def rerank_queries(
        self,
        query_texts: list[str],
        doc_texts: list[str],
        faiss_indices,       # np.ndarray (n_queries, faiss_top_k)
        reranker_top_k: int,
        batch_size: int = 1,
        return_scores: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """
        Rerank FAISS results for all queries.

        Args:
            query_texts: list of query strings
            doc_texts: full corpus text list (indexed by doc idx)
            faiss_indices: np.ndarray shape (n_queries, faiss_top_k)
            reranker_top_k: how many to keep per query after reranking
            batch_size: number of queries to process at once
            return_scores: if True, also return reranker scores

        Returns:
            list of lists of reranked corpus indices;
            if return_scores=True, also returns list of lists of scores.
        """
        all_reranked = []
        all_scores = [] if return_scores else None

        for i in range(0, len(query_texts), batch_size):
            for j in range(i, min(i + batch_size, len(query_texts))):
                query = query_texts[j]
                retrieved_ids = faiss_indices[j].tolist()
                docs = [doc_texts[idx] for idx in retrieved_ids]

                reranked_ids, scores = self.rerank(
                    query, docs, retrieved_ids, top_k=reranker_top_k,
                )
                all_reranked.append(reranked_ids)
                if return_scores:
                    all_scores.append(scores)

            done = min(i + batch_size, len(query_texts))
            if done % max(1, len(query_texts) // 20) < batch_size or done == len(query_texts):
                print(f"    Reranked {done}/{len(query_texts)} queries")

            if self.is_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_scores:
            return all_reranked, all_scores
        return all_reranked
