# CheckThat2026

This code accompanies: "Scientific Claim-Source Retrieval Revisited" (CLEF 2025, Madrid, Spain).

## Setup

- GRITLM is required for embedding generation. Get it from https://github.com/ContextualAI/gritlm.git.
- Use two separate environments because GRITLM depends on older transformers while rerankers
	require newer versions.
	- GRITLM environment: install from requirements.txt
	- Reranker environment: install from requirements_reranker.txt

## Entity Reranker

The entity reranker extracts query entities with an LLM and reorders the top-k
retrieval candidates by entity overlap (higher overlap ranks higher). The merged
script supports both batch (v3) and vLLM async (v3.5) execution modes, reuses
cached entity extractions, and writes evaluation reports and reranked indices.

- Main script: entity_reranker/llm_entity_reranker_entity_count.py
- Inputs: dataset queries/collection + reranker top-k caches
- Outputs: cached entity JSON, reranked indices (.npz), evaluation reports (.md)

## Other Rerankers

- Verification reranker: /data/horse/ws/hakh140h-style_transfer/CheckThat2026/verification_reranker
- Attribution-based approach: /data/horse/ws/hakh140h-style_transfer/CheckThat2026/Signal-based/Attribution

## Evaluation

Merged evaluation scripts now live under evaluation/ with a shared query-mode flag.

- Embedding evaluation: evaluation/evaluate_embeddings.py
	- Query modes: translated_only, paired, paired_en_once
	- Use --original_dataset_dir for paired modes
	- Cache tags: default v2/v2_5 for paired modes (override with --cache_tag)
- GritLM evaluation: evaluation/evaluate_gritlm.py
	- Query modes: translated_only, paired, paired_en_once
	- Use --original_dataset_dir for paired modes
	- Cache tags: default v2/v2_5 for paired modes (override with --cache_tag)

Note: the folder name stays evaluation to avoid breaking imports across the codebase.

## Reranker Runs

Unified reranker runner lives in reranker_runs/run_reranker.py and supports only the
following models:

- BAAI/bge-reranker-v2-m3
- jinaai/jina-reranker-v3
- Qwen/Qwen3-Reranker-0.6B
- Qwen/Qwen3-Reranker-8B
- nvidia/llama-nemotron-rerank-vl-1b-v2

Example usage:

```bash
python reranker_runs/run_reranker.py \
	--reranker Qwen/Qwen3-Reranker-8B \
	--cache_dir /path/to/eval_cache \
	--dataset_dir /path/to/Dataset_translated \
	--model_path /path/to/GritLM-7B \
	--cache_tag v2_5 \
	--languages en de fr \
	--split dev \
	--save_scores
```