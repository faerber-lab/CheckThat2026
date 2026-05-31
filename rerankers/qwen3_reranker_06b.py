"""Qwen3 Reranker (Qwen/Qwen3-Reranker-0.6B) wrapper - With Configurable Batch Sizes."""
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Reranker06B:
    """Wrapper for Qwen/Qwen3-Reranker-0.6B on A100 GPUs."""

    PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    DEFAULT_INSTRUCTION = (
        "Given a scientific claim from a social media post, "
        "retrieve the source paper that supports or is related to the claim"
    )

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

        self._assert_cuda_healthy()

        print(f"  Loading reranker: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side="left", 
            local_files_only=True
        )
        
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        
        model_kwargs = {
            "local_files_only": True,
            "device_map": self.device if self.is_cuda else None,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).eval()
        
        if not self.is_cuda:
            self.model.to(self.device)
        
        self.max_length = 8192
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        
        torch.cuda.empty_cache()
        print(f"  Reranker loaded. Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    def _assert_cuda_healthy(self):
        if not torch.cuda.is_available():
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise RuntimeError("CUDA_VISIBLE_DEVICES set but CUDA unavailable.")
            return
        try:
            test = torch.zeros(1, device=self.device)
            _ = test + 1
            torch.cuda.synchronize(self.device)
        except Exception as e:
            raise RuntimeError(f"CUDA device {self.device} failed: {e}") from e

    def _format_pair(self, query: str, doc: str, instruction: str) -> str:
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _compute_scores(self, pairs: list[str], batch_size: int = 16) -> list[float]:
        """Score pairs with the given batch size."""
        all_scores = []

        # THIS IS THE KEY FIX: Use the passed batch_size parameter
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            max_input_len = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            inputs = self.tokenizer(
                batch,
                padding=False,
                truncation=True,
                max_length=max_input_len,
                return_attention_mask=False,
            )
            
            for idx in range(len(inputs["input_ids"])):
                inputs["input_ids"][idx] = (
                    self.prefix_tokens + inputs["input_ids"][idx] + self.suffix_tokens
                )
            
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :]
                true_v = logits[:, self.token_true_id]
                false_v = logits[:, self.token_false_id]
                stacked = torch.stack([false_v, true_v], dim=1)
                scores = torch.nn.functional.softmax(stacked, dim=1)[:, 1]
            
            all_scores.extend(scores.tolist())
            del inputs, logits, true_v, false_v, stacked, scores

        return all_scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: list[int],
        batch_size: int = 1,
        top_k: int | None = None,
    ) -> tuple[list[int], list[float]]:
        pairs = [self._format_pair(query, doc, self.DEFAULT_INSTRUCTION) for doc in documents]
        all_scores = self._compute_scores(pairs, batch_size=batch_size)
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
        batch_size: int = 16,         # FIXED: Changed default from 1 to 16
        query_batch_size: int = 100,  # FIXED: Changed default from 10 to 100
        return_scores: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """Rerank queries with configurable batch sizes."""
        import sys

        n_queries = len(query_texts)
        faiss_top_k = faiss_indices.shape[1]
        all_reranked = []
        all_scores_out = [] if return_scores else None

        report_step = max(1, n_queries // 10)

        # Process queries in query_batch_size chunks
        for i in range(0, n_queries, query_batch_size):
            end = min(i + query_batch_size, n_queries)

            # Collect all pairs for this chunk
            all_pairs = []
            for j in range(i, end):
                query = query_texts[j]
                retrieved_ids = faiss_indices[j].tolist()
                for idx in retrieved_ids:
                    all_pairs.append(
                        self._format_pair(query, doc_texts[idx], self.DEFAULT_INSTRUCTION)
                    )

            # Score all pairs - THIS FORWARDS batch_size CORRECTLY
            flat_scores = self._compute_scores(all_pairs, batch_size=batch_size)

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

        if return_scores:
            return all_reranked, all_scores_out
        return all_reranked