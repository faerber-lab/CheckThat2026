# CheckThat! 2026 — Scientific Claim–Source Retrieval

This repository contains the code, rerankers, and evaluation pipelines for the following papers:

- **Workshop paper for CheckThat! 2026**: [Claim2Source at CheckThat! 2026: Improving Multilingual Scientific Claim–Source Retrieval with Verification-based Re-Ranking](https://arxiv.org/pdf/2607.04043)
- **Best of labs paper**: [Scientific Claim–Source Retrieval Revisited: A Comparative Study of Style Transfer and Re-Ranking](https://arxiv.org/pdf/2607.15875)

## Setup

- GRITLM is required for embedding generation. Get it from https://github.com/ContextualAI/gritlm.git.
- Use two separate environments because GRITLM depends on older transformers while rerankers
	require newer versions.
	- GRITLM environment: install from requirements.txt
	- Reranker environment: install from requirements_reranker.txt

## Style Transfer for Scientific Claims (For Best of labs paper)
Applies LLM-based prompting strategies (e.g., formal rewriting, synthetic abstracts, scientific questions) to reformulate informal tweets into formal scientific representations to improve initial retrieval.

## Similarity-based re-ranking (For Both Papers)

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
## Signal-based Re-Ranking  

- **Attribution-based Approach** (`Signal-based/Attribution/`) (For Best of labs paper): This tool decomposes claims into atomic facts, evaluates their entailment using LLMs for Natural Language Inference (NLI), and calculates a support rate to re-rank candidate sources.
- **Entity Reranker** (`entity_reranker/`) (For Best of labs paper): The entity reranker extracts query entities with an LLM and reorders the top-k retrieval candidates by entity overlap (higher overlap ranks higher). The merged script supports both batch (v3) and vLLM async (v3.5) execution modes, reuses cached entity extractions, and writes evaluation reports and reranked indices.
	- Main script: `entity_reranker/llm_entity_reranker_entity_count.py`
	- Inputs: dataset queries/collection + reranker top-k caches
	- Outputs: cached entity JSON, reranked indices (.npz), evaluation reports (.md)
- **Verification Reranker** (`verification_reranker/`) (For Both Papers)

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


