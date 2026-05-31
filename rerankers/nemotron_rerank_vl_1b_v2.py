"""Nemotron Reranker (nvidia/llama-nemotron-rerank-vl-1b-v2) wrapper - Optimized."""
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoProcessor


class NemotronRerankVL1BV2:
    """Optimized wrapper for nvidia/llama-nemotron-rerank-vl-1b-v2."""

    def __init__(self, model_name: str = "nvidia/llama-nemotron-rerank-vl-1b-v2", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

        print(f"  Loading reranker: {model_name} on {device}")

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa",
                device_map=self.device if self.is_cuda else "auto",
                local_files_only=True,
            )
            .eval()
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            max_input_tiles=6,
            use_thumbnail=True,
            rerank_max_length=8192,
            local_files_only=True,
        )
        
        torch.cuda.empty_cache()
        print(f"  Reranker loaded. Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    def _prepare_batch(self, examples):
        """Prepare a batch of examples for the model."""
        batch_dict = self.processor.process_queries_documents_crossencoder(examples)
        if self.is_cuda:
            batch_dict = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_dict.items()
            }
        return batch_dict

    def _compute_scores(self, examples: list[dict], batch_size: int = 4) -> list[float]:
        """Score examples in batches."""
        all_scores = []

        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            batch_dict = self._prepare_batch(batch)

            with torch.inference_mode():
                logits = self.model(**batch_dict, return_dict=True).logits.squeeze(-1)
                all_scores.extend(logits.float().tolist())
            
            del batch_dict, logits
            torch.cuda.empty_cache()

        return all_scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: list[int],
        batch_size: int = 4,
        top_k: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """Rerank documents for a single query."""
        examples = [
            {"question": query, "doc_text": doc, "doc_image": ""}
            for doc in documents
        ]
        all_scores = self._compute_scores(examples, batch_size=batch_size)

        sorted_idx = sorted(range(len(all_scores)), key=lambda j: all_scores[j], reverse=True)
        if top_k is not None:
            sorted_idx = sorted_idx[:top_k]

        return [doc_ids[j] for j in sorted_idx], [all_scores[j] for j in sorted_idx]

    def rerank_queries(
        self,
        query_texts: list[str],
        doc_texts: list[str],
        faiss_indices,
        reranker_top_k: int,
        batch_size: int = 4,
        query_batch_size: int = 10,
        return_scores: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """Rerank queries efficiently."""
        import sys

        n_queries = len(query_texts)
        faiss_top_k = faiss_indices.shape[1]
        all_reranked = []
        all_scores_out = [] if return_scores else None

        report_step = max(1, n_queries // 10)

        for i in range(0, n_queries, query_batch_size):
            end = min(i + query_batch_size, n_queries)

            # Collect all examples for this chunk
            chunk_examples = []
            for j in range(i, end):
                query = query_texts[j]
                retrieved_ids = faiss_indices[j].tolist()
                for idx in retrieved_ids:
                    chunk_examples.append({
                        "question": query,
                        "doc_text": doc_texts[idx],
                        "doc_image": "",
                    })

            # Score all examples in scoring batches
            flat_scores = self._compute_scores(chunk_examples, batch_size=batch_size)

            scores_matrix = np.asarray(flat_scores, dtype=np.float32).reshape(end - i, faiss_top_k)
            sorted_local = np.argsort(-scores_matrix, axis=1)[:, :reranker_top_k]

            for j in range(end - i):
                orig_ids = faiss_indices[i + j]
                all_reranked.append(orig_ids[sorted_local[j]].tolist())
                if return_scores:
                    all_scores_out.append(scores_matrix[j, sorted_local[j]].tolist())

            if end % report_step == 0 or end == n_queries:
                print(f"    Reranked {end}/{n_queries} queries", flush=True)

            sys.stdout.flush()
            torch.cuda.empty_cache()

        if return_scores:
            return all_reranked, all_scores_out
        return all_reranked