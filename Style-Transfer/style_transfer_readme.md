# Style Transfer Module

This folder contains the implementation and experimental setup for **LLM-based Style Transfer** in scientific claim-source retrieval (CheckThat! 2026 Lab).
The core objective is to bridge the domain and style gap between informal social media posts (e.g., tweets/claims) and formal scientific literature (e.g., research paper abstracts) to significantly boost information retrieval performance (e.g., MRR, Hits@k).

---

## Directory Structure

```
Style-Transfer/
├── config.py                        # Centralized configuration constants (Prompt IDs)
├── process_data.py                  # Utilities for loading and preprocessing datasets (queries & corpus)
├── llm_requests.py                  # LLM inference pipeline supporting vLLM and OpenAI API backends
├── prepare_style_transfer_data.py   # Executable script for running batch style-transfer generation
├── evaluate.py                      # Evaluation framework (BM25, Dense Embeddings, FAISS indexing, MRR)
└── prompts/                         # Collection of prompt templates for style transformation
    ├── prompt_1.txt                 # Prompt 1: Minimal Cleanup & Normalization
    ├── prompt_2.txt                 # Prompt 2: Formal Scientific Claim Rewriting
    ├── prompt_3.txt                 # Prompt 3: Synthetic Abstract Generation (~150 words)
    └── prompt_4.txt                 # Prompt 4: Targeted Scientific Question Formulation
```

---

## Overview of Style Transfer Strategies (`prompts/`)

We explore four main prompting strategies to transform informal input tweets into optimized search queries:

| Prompt ID | Target Style / Format | Key Objective |
| :--- | :--- | :--- |
| **`prompt_1.txt`** | **Minimal Preprocessing** | Strips non-essential elements (hashtags, emojis, symbols) while retaining exact original phrasing and terminology. |
| **`prompt_2.txt`** | **Formal Scientific Claim** | Rewrites informal posts into structured, formal academic claim assertions using domain-specific vocabulary. |
| **`prompt_3.txt`** | **Synthetic Abstract** | Expands the query into a full-fledged concise scientific abstract (~150 words) synthesizing context, methodology, and findings without introducing hallucinated facts. |
| **`prompt_4.txt`** | **Scientific Question** | Formulates a single, highly precise scientific research question specifically targetable against candidate paper abstracts. |

---

## Key Modules & Descriptions

### 1. `config.py`
Centralizes prompt selection IDs (`QUERY_PROMPT_NUMBER`, `CORPUS_PROMPT_NUMBER`, etc.) across the style transfer and re-ranking pipelines to guarantee reproducible experiment configurations.

### 2. `process_data.py`
Handles data ingestion for both query datasets (tweets/claims) and candidate document corpora (`collection_data_process.json`), standardizing text representations across multilingual inputs (English, German, French).

### 3. `llm_requests.py`
Executes LLM inference using either local fast batch inference engines (**vLLM** via models such as `Qwen/Qwen3.5-27B` or `Qwen/Qwen3-9B`) or cloud API models (**OpenAI API** / `meta-llama/Llama-3.3-70B-Instruct`). Features multi-threaded parallel requests and automatic batching.

### 4. `prepare_style_transfer_data.py`
A CLI utility to trigger end-to-end multi-version, multi-style, and multilingual query transformation runs.

### 5. `evaluate.py`
Comprehensive evaluation pipeline that builds dense FAISS indexes or BM25 lexical search structures over transformed queries and paper abstracts, evaluating retrieval effectiveness using metrics like **Mean Reciprocal Rank (MRR)**.

---

## Usage Examples

### 1. Running Query Style Transfer
Generate rewritten queries using `vLLM` with the default Qwen model:
```bash
python prepare_style_transfer_data.py \
    --version 1 \
    --backend vllm \
    --vllm-model Qwen/Qwen3.5-27B \
    --vllm-tp 1
```

Or run via OpenAI-compatible API endpoint:
```bash
python llm_requests.py \
    --task query \
    --backend api
```

### 2. Evaluating Retrieval Metrics
Evaluate the transformed queries against the document corpus:
```bash
python evaluate.py
```
This script computes similarity scores using BM25 and dense embeddings, saving ranking metrics to `mrr_results/`.
