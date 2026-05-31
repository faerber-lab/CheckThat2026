"""Qwen3 Reranker (Qwen/Qwen3-Reranker-8B) wrapper for the eval pipeline."""
import time
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Reranker8B:
    """
    Wrapper around Qwen/Qwen3-Reranker-8B.

    LLM-based reranker that scores query-document pairs using yes/no token logits.
    Requires transformers>=4.51.0.
    """

    PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think&gt;\n\n</think&gt;\n\n"
    DEFAULT_INSTRUCTION = (
        "Given a scientific claim from a social media post, "
        "retrieve the source paper that supports or is related to the claim"
    )
    MAX_BATCH_TOKENS = 24576  # Reduced from 32768 to avoid OOM on 40GB GPUs

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B", device: str = None, device_id: int = None):
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = f"cuda:{device_id}"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

        self._assert_cuda_healthy(device_id)

        print(f"  Loading reranker: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", local_files_only=True)

        print(f"  Loading model with attn_implementation=sdpa...")
        if self.is_cuda and device_id is not None:
            device_map = {"" : f"cuda:{device_id}"}
        elif self.is_cuda:
            device_map = device
        else:
            device_map = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.is_cuda else torch.float32,
            attn_implementation="sdpa",
            device_map=device_map,
            local_files_only=True,
        ).eval()
        if not self.is_cuda:
            self.model.to(self.device)
        self.max_length = 8192

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self.content_budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        print(f"  Reranker loaded.")

    def _assert_cuda_healthy(self, device_id: int | None):
        """Raise RuntimeError if CUDA is broken, so we fail fast instead of falling back to CPU."""
        if not torch.cuda.is_available():
            if os.environ.get("CUDA_VISIBLE_DEVICES") or device_id is not None:
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES is set (or device_id was passed) but "
                    "torch.cuda.is_available() is False. The CUDA driver/context is "
                    "likely corrupted from a prior crash. Kill all Python processes "
                    "and run nvidia-smi before retrying."
                )
            return
        dev = f"cuda:{device_id}" if device_id is not None else "cuda"
        try:
            test = torch.zeros(1, device=dev)
            _ = test + 1
            torch.cuda.synchronize(dev)
        except Exception as e:
            raise RuntimeError(
                f"CUDA device {dev} failed a basic health check. "
                f"The GPU driver/context is likely corrupted. Error: {e}"
            ) from e

    def _format_pair(self, query: str, doc: str, instruction: str) -> str:
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _compute_scores(self, pairs: list[str], max_batch_size: int = 16) -> list[float]:  # Reduced from 64
        """Score query-document pairs via yes/no logit probabilities.

        Pairs are sorted by token length and batched dynamically to minimize
        padding waste while staying within memory limits.
        """
        import sys

        # Tokenize all pairs (no padding yet) to get lengths
        raw_inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.content_budget,
        )
        lengths = [len(ids) + len(self.prefix_tokens) + len(self.suffix_tokens)
                   for ids in raw_inputs["input_ids"]]

        # Sort by length — short with short, long with long
        sorted_idx = np.argsort(lengths)

        all_scores = [None] * len(pairs)
        i = 0
        n_batches = 0
        report_every = max(1, len(pairs) // 20)

        while i < len(sorted_idx):
            batch_idx = []
            batch_max_len = 0

            while len(batch_idx) < max_batch_size and i + len(batch_idx) < len(sorted_idx):
                idx = sorted_idx[i + len(batch_idx)]
                seq_len = lengths[idx]
                padded_cost = (len(batch_idx) + 1) * max(seq_len, batch_max_len)

                if padded_cost > self.MAX_BATCH_TOKENS and batch_idx:
                    break

                batch_idx.append(idx)
                batch_max_len = max(batch_max_len, seq_len)

            batch_inputs = {"input_ids": [
                self.prefix_tokens + raw_inputs["input_ids"][j] + self.suffix_tokens
                for j in batch_idx
            ]}
            batch_inputs = self.tokenizer.pad(batch_inputs, padding=True, return_tensors="pt")
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            with torch.inference_mode():
                # Run backbone only, then lm_head on last token alone.
                # Avoids allocating (batch × seq_len × vocab_size) logits tensor.
                last_hidden = self.model.model(**batch_inputs).last_hidden_state[:, -1, :]
                logits = self.model.lm_head(last_hidden)
                true_v = logits[:, self.token_true_id]
                false_v = logits[:, self.token_false_id]
                stacked = torch.stack([false_v, true_v], dim=1)
                scores = torch.nn.functional.softmax(stacked, dim=1)[:, 1]

            for j, orig_j in enumerate(batch_idx):
                all_scores[orig_j] = scores[j].item()

            del last_hidden, logits, true_v, false_v, stacked, scores

            i += len(batch_idx)
            n_batches += 1
            if n_batches % report_every == 0 or i >= len(sorted_idx):
                print(f"      scored {i}/{len(pairs)} pairs ({i/len(pairs)*100:.0f}%)", end="", flush=True)
                if i < len(sorted_idx):
                    print(f"\r", end="", flush=True)
                else:
                    print(flush=True)

        return all_scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: list[int],
        batch_size: int = 16,  # Reduced from 64
        top_k: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        Rerank documents for a single query.

        Returns (reranked_doc_ids, scores) sorted descending by score.
        """
        pairs = [self._format_pair(query, doc, self.DEFAULT_INSTRUCTION) for doc in documents]
        all_scores = self._compute_scores(pairs, max_batch_size=batch_size)

        sorted_idx = sorted(range(len(all_scores)), key=lambda j: all_scores[j], reverse=True)
        if top_k is not None:
            sorted_idx = sorted_idx[:top_k]

        return [doc_ids[j] for j in sorted_idx], [all_scores[j] for j in sorted_idx]

    def rerank_queries(
        self,
        query_texts: list[str],
        doc_texts: list[str],
        faiss_indices,       # np.ndarray (n_queries, faiss_top_k)
        reranker_top_k: int,
        batch_size: int = 16,  # Reduced from 64
        query_batch_size: int = 25,  # Reduced from 100
        return_scores: bool = False,
    ) -> list[list[int]] | tuple[list[list[int]], list[list[float]]]:
        """
        Rerank FAISS results for all queries using batched GPU inference.
        """
        import sys

        n_queries = len(query_texts)
        faiss_top_k = faiss_indices.shape[1]
        all_reranked = []
        all_scores_out = [] if return_scores else None

        report_step = max(1, n_queries // 20)
        t0 = time.time()

        for i in range(0, n_queries, query_batch_size):
            end = min(i + query_batch_size, n_queries)

            all_pairs = []
            for j in range(i, end):
                query = query_texts[j]
                retrieved_ids = faiss_indices[j].tolist()
                for idx in retrieved_ids:
                    all_pairs.append(self._format_pair(query, doc_texts[idx], self.DEFAULT_INSTRUCTION))

            flat_scores = self._compute_scores(all_pairs, max_batch_size=batch_size)

            scores_matrix = np.asarray(flat_scores, dtype=np.float32).reshape(end - i, faiss_top_k)
            sorted_local = np.argsort(-scores_matrix, axis=1)[:, :reranker_top_k]

            # Clear GPU cache after each query batch
            if self.is_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            for j in range(end - i):
                orig_ids = faiss_indices[i + j]
                all_reranked.append(orig_ids[sorted_local[j]].tolist())
                if return_scores:
                    all_scores_out.append(scores_matrix[j, sorted_local[j]].tolist())

            if end % report_step == 0 or end == n_queries:
                elapsed = time.time() - t0
                rate = end / elapsed * 3600
                eta = (n_queries - end) / rate
                print(f"    Reranked {end}/{n_queries} queries ({end/n_queries*100:.0f}%) — {elapsed/60:.0f}min elapsed, ~{eta:.0f}min remaining", flush=True)
            else:
                print(f"    Reranked {end}/{n_queries} queries", flush=True)

            sys.stdout.flush()

            if self.is_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_scores:
            return all_reranked, all_scores_out
        return all_reranked
