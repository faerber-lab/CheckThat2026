#!/usr/bin/env python3
"""
LLM entity-count reranker.

Usage:
  # Batch mode with optional parallel sharding
  python entity_reranker/llm_entity_reranker_entity_count.py \
    --engine batch \
        --dataset_dir /path/to/CT26/Dataset_translated \
        --cache_dir /path/to/GRITLM_finetune/eval_cache_gritlm_translated \
        --output_dir /path/to/GRITLM_finetune/eval_results_translated \
    --reranker Qwen/Qwen3-Reranker-8B \
    --llm_model moonshotai/Kimi-K2.6 \
    --llm_top_k 100 \
    --parallel_shards 4

    # Batch mode with OpenAI-compatible API
    python entity_reranker/llm_entity_reranker_entity_count.py \
        --engine batch \
        --api_backend openai \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url https://api.openai.com/v1 \
        --llm_model gpt-4.1 \
        --llm_top_k 100 \
        --languages en

  # vLLM async mode
  python entity_reranker/llm_entity_reranker_entity_count.py \
    --engine vllm_async \
        --dataset_dir /path/to/CT26/Dataset_translated \
        --cache_dir /path/to/GRITLM_finetune/eval_cache_gritlm_translated \
        --output_dir /path/to/GRITLM_finetune/eval_results_translated \
    --reranker Qwen/Qwen3-Reranker-8B \
    --llm_model Qwen/Qwen3.5-9B \
    --llm_top_k 10 \
    --vllm_url http://localhost:8000/v1 \
    --max_concurrent 50
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, Optional

import numpy as np
from openai import AsyncOpenAI

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from evaluation.utils import (
    load_collection,
    load_queries,
    get_model_slug,
    evaluate_mrr,
    evaluate_hit_rate,
    macro_average_results,
)


def resolve_openai_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("OPENAI_API_KEY")


def require_openai_api_key(api_key: Optional[str], context: str) -> str:
    resolved = resolve_openai_api_key(api_key)
    if not resolved:
        raise ValueError(f"{context}: --openai_api_key or OPENAI_API_KEY is required")
    return resolved


# ------------------------------------------------------------
# Query utilities
# ------------------------------------------------------------

def load_original_queries(dataset_dir: str, lang: str) -> dict:
    """Load original-language queries from the 'copy' files for de/fr."""
    copy_file = os.path.join(dataset_dir, f"{lang}_dev copy.json")
    if not os.path.isfile(copy_file):
        return {}
    with open(copy_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {q["index"]: q["text"] for q in data}
    print(f"  Loaded {len(mapping)} original {lang.upper()} queries from copy file")
    return mapping


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

ENTITY_SYSTEM_PROMPT = """You are a multilingual expert in English, French and German at analyzing whether a scientific/academic document is relevant to a query by checking entity overlap.

**Handling bilingual queries:**
The user may provide a single query (`Query: ...`) OR a pair (`Query (original language): ...` and `Query (translated to English): ...`).
- If both are given, the original is in French or German. Use it as the PRIMARY source for extracting key entities.
- The English translation is a SUPPLEMENT to help recognise cross-lingual synonyms, abbreviations, and ambiguous terms.
- Always consider entities from BOTH versions. When checking entity overlap with the document, accept matches to either the original (in French/German) or its English equivalent.

**Your task:**
Given a QUERY and a list of DOCUMENTS (each with title, abstract, venue and author), do the following:

1. Identify the key entities mentioned in the QUERY. Normalize synonyms and abbreviations. Key entities include:
   - Biomedical & chemical entities: diseases, viruses, bacteria, drugs, chemicals, genes, proteins, pathways, cell types, model organisms
   - Numerical & quantitative entities: specific percentages, counts, thresholds, time points, age ranges, dosages, measurements
   - Technical & methodological entities: algorithms, statistical methods, software/tools, protocols, imaging techniques, assays, frameworks
   - Named entities: datasets, institutions, geographic locations, people, clinical trial phases, adverse events
   - Temporal & contextual entities: specific years, durations, frequencies

2. For each DOCUMENT, identify which key entities from the QUERY appear (or are directly synonymous) in that document. Be thorough and specific.

You MUST respond with ONLY a JSON object with exactly two keys:
- "query_entities": a list of strings, the key entities you identified from the query.
- "doc_entities": a list of lists of strings. Each inner list contains the entities from the query that were found in the corresponding document (by document number order). If no entities matched, use an empty list [].

Example response:
{"query_entities": ["COVID-19", "mRNA vaccine", "phase 3 trial"], "doc_entities": [["COVID-19", "mRNA vaccine"], ["phase 3 trial"], []]}

No extra text, no explanation."""


def build_user_message(query: str, translated_query: str, papers: list[str]) -> str:
    if translated_query and translated_query != query:
        return (
            f"Query (original language):\n\"{query}\"\n\n"
            f"Query (translated to English):\n\"{translated_query}\"\n\nDocuments:\n"
            + "\n".join(papers)
        )
    return f"Query:\n\"{query}\"\n\nDocuments:\n" + "\n".join(papers)


# ------------------------------------------------------------
# Robust JSON extraction
# ------------------------------------------------------------

_parse_fail_count = 0
_PARSE_FAIL_MAX_PRINT = 5


def _extract_balanced(text: str, open_char: str = "{", close_char: str = "}") -> str | None:
    """Extract balanced JSON substring starting from the LAST occurrence of open_char."""
    last_open = text.rfind(open_char)
    if last_open == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(last_open, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[last_open:i + 1]
    return None


def _try_parse_json_obj(content: str, label: str = "") -> dict | None:
    """Multiple strategies to extract a JSON object from LLM response text."""
    global _parse_fail_count

    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    s = _extract_balanced(content, "{", "}")
    if s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and ("query_entities" in obj or "doc_entities" in obj):
                return obj
        except json.JSONDecodeError:
            pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if m:
        s = _extract_balanced(m.group(1).strip(), "{", "}")
        if s:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    first = content.find("{")
    if first >= 0:
        s = _extract_balanced(content[first:], "{", "}")
        if s:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    _parse_fail_count += 1
    if _parse_fail_count <= _PARSE_FAIL_MAX_PRINT:
        print(f"    [WARN] JSON parse failed for {label} (failure #{_parse_fail_count}):")
        print(f"    Raw content (first 500 chars): {content[:500]}")
    elif _parse_fail_count == _PARSE_FAIL_MAX_PRINT + 1:
        print(f"    [WARN] Suppressing further parse failure logs (total: {_parse_fail_count})")
    return None


# ------------------------------------------------------------
# Tokenization helpers
# ------------------------------------------------------------

TOKENIZER = None


def get_tokenizer(model_name: str = "Qwen/Qwen3.5-9B"):
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer

        TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return TOKENIZER


def count_tokens(text: str, model_name: str = "Qwen/Qwen3.5-9B") -> int:
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))


def get_tiktoken_encoder(model_name: str = "gpt-4"):
    import tiktoken

    return tiktoken.encoding_for_model(model_name)


# ------------------------------------------------------------
# Entity extraction (batch mode, v3)
# ------------------------------------------------------------

async def score_and_extract_entities_batch(
    client: AsyncOpenAI,
    model: str,
    query: str,
    translated_query: str,
    doc_texts: list[str],
    cache_key: str = "",
    max_total_tokens: int = 250000,
    output_tokens_reserve: int = 32768,
    chunk_size: int = None,
) -> tuple[list[int], dict]:
    """Extract entities for all docs in one or more batch API calls.

    Returns (dummy_scores_list, entity_info_dict).
    """
    enc = get_tiktoken_encoder("gpt-4")

    if chunk_size and chunk_size > 0:
        chunks = [doc_texts[i:i + chunk_size] for i in range(0, len(doc_texts), chunk_size)]
    else:
        chunks = [doc_texts]

    all_dummy_scores = []
    all_doc_entities = []
    all_query_entities = []

    for chunk_idx, chunk_docs in enumerate(chunks):
        if query == translated_query:
            base_user = f"Query: {query}\n\nDocuments:\n"
        else:
            base_user = (
                f"Query (original language): {query}\n"
                f"Query (translated to English): {translated_query}\n\nDocuments:\n"
            )
        base_tokens = len(enc.encode(ENTITY_SYSTEM_PROMPT)) + len(enc.encode(base_user))

        docs_with_indices = [(i, f"{i + 1}. {txt}\n") for i, txt in enumerate(chunk_docs)]

        kept_docs = []
        kept_indices = []
        current_tokens = base_tokens
        for orig_idx, doc_block in docs_with_indices:
            block_tokens = len(enc.encode(doc_block))
            if current_tokens + block_tokens + output_tokens_reserve <= max_total_tokens:
                kept_docs.append(doc_block)
                kept_indices.append(orig_idx)
                current_tokens += block_tokens
            else:
                print(
                    f"    [TRUNC] chunk {chunk_idx + 1}: kept {len(kept_docs)}/{len(chunk_docs)} docs due to token limit"
                )
                break

        if not kept_docs:
            first_doc = docs_with_indices[0][1][:500]
            kept_docs = [first_doc]
            kept_indices = [0]

        user_message = base_user + "".join(kept_docs)

        extra_body = {}
        if "Kimi" in model:
            extra_body = {"chat_template_kwargs": {"thinking": False}}

        n_docs = len(chunk_docs)
        default_doc_entities = [[] for _ in range(n_docs)]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=output_tokens_reserve,
                temperature=0.0,
                extra_body=extra_body,
            )
            content = response.choices[0].message.content
            if content is None:
                content = getattr(response.choices[0].message, "reasoning_content", None) or ""
                if not content:
                    print(f"    [ERROR] {cache_key}: Empty response from API (chunk {chunk_idx + 1})")
                    all_dummy_scores.extend([0] * n_docs)
                    all_doc_entities.extend(default_doc_entities)
                    continue
                print(f"    [INFO] {cache_key}: Used reasoning_content fallback (chunk {chunk_idx + 1})")

            content = content.strip()

            parsed = _try_parse_json_obj(content, label=f"{cache_key}_chunk{chunk_idx}")
            if parsed is None:
                all_dummy_scores.extend([0] * n_docs)
                all_doc_entities.extend(default_doc_entities)
                continue

            query_entities = parsed.get("query_entities", [])
            doc_entities_raw = parsed.get("doc_entities", [])

            if not doc_entities_raw and not query_entities:
                print(
                    f"    [WARN] {cache_key}_chunk{chunk_idx}: Parsed JSON but got empty entities, keys={list(parsed.keys())}"
                )
                all_dummy_scores.extend([0] * n_docs)
                all_doc_entities.extend(default_doc_entities)
                continue

            dummy_scores = [0] * len(kept_indices)

            doc_entities = [list(d) for d in doc_entities_raw[:len(kept_indices)]]
            if len(doc_entities) < len(kept_indices):
                doc_entities += [[] for _ in range(len(kept_indices) - len(doc_entities))]

            chunk_dummy_scores = [0] * n_docs
            chunk_doc_entities = [[] for _ in range(n_docs)]
            for pos, sc, ent in zip(kept_indices, dummy_scores, doc_entities):
                chunk_dummy_scores[pos] = sc
                chunk_doc_entities[pos] = ent

            all_dummy_scores.extend(chunk_dummy_scores)
            all_doc_entities.extend(chunk_doc_entities)
            if query_entities:
                all_query_entities = query_entities

        except Exception as e:
            print(f"    [ERROR] {cache_key}_chunk{chunk_idx}: API call failed: {e}")
            if "content" in locals() and content:
                print(f"    Content (first 300 chars): {content[:300]}")
            all_dummy_scores.extend([0] * n_docs)
            all_doc_entities.extend(default_doc_entities)

    info = {"query_entities": list(all_query_entities), "doc_entities": all_doc_entities}
    return all_dummy_scores, info


# ------------------------------------------------------------
# Entity extraction (vLLM async mode, v3.5)
# ------------------------------------------------------------

def format_paper_full(title: str, venue: str, authors: str, abstract: str) -> str:
    return f"{len(title) + 1}. Title: {title}\nVenue: {venue}\nAuthors: {authors}\nAbstract: {abstract}"


def format_paper_without_authors(title: str, venue: str, abstract: str) -> str:
    return f"Title: {title}\nVenue: {venue}\nAbstract: {abstract}"


async def extract_entities_single_query(
    client: AsyncOpenAI,
    model: str,
    query_text: str,
    translated_query: str,
    titles: list[str],
    venues: list[str],
    authors: list[str],
    abstracts: list[str],
    cache_key: str,
    semaphore: asyncio.Semaphore,
    max_total_tokens: int = 60000,
    output_tokens_reserve: int = 10000,
    max_retries: int = 3,
) -> tuple[dict, str]:
    async with semaphore:
        papers = [format_paper_full(titles[i], venues[i], authors[i], abstracts[i]) for i in range(len(titles))]
        remove_authors = False

        for attempt in range(max_retries):
            try:
                user_msg = build_user_message(query_text, translated_query, papers)
                sys_tokens = count_tokens(ENTITY_SYSTEM_PROMPT, model)
                user_tokens = count_tokens(user_msg, model)
                total = sys_tokens + user_tokens + output_tokens_reserve + 100

                if total > max_total_tokens:
                    if not remove_authors:
                        papers = [
                            format_paper_without_authors(titles[i], venues[i], abstracts[i])
                            for i in range(len(titles))
                        ]
                        remove_authors = True
                        continue
                    allowed_user = max_total_tokens - sys_tokens - output_tokens_reserve - 500
                    if user_tokens > allowed_user and allowed_user > 0:
                        ratio = allowed_user / user_tokens
                        new_papers = []
                        for p in papers:
                            toks = get_tokenizer(model).encode(p)
                            limit = max(200, int(len(toks) * ratio))
                            new_papers.append(get_tokenizer(model).decode(toks[:limit]))
                        papers = new_papers

                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_message(query_text, translated_query, papers)},
                    ],
                    max_tokens=output_tokens_reserve,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content or ""

                parsed = _try_parse_json_obj(content.strip(), label=cache_key)
                if parsed is None:
                    return {"query_entities": [], "doc_entities": [[] for _ in range(len(titles))]}, "Parse failed"

                query_entities = parsed.get("query_entities", [])
                doc_entities_raw = parsed.get("doc_entities", [])

                doc_entities = [list(d) for d in doc_entities_raw[:len(titles)]]
                if len(doc_entities) < len(titles):
                    doc_entities += [[] for _ in range(len(titles) - len(doc_entities))]

                return {"query_entities": query_entities, "doc_entities": doc_entities}, "Success"

            except Exception as e:
                if "maximum context length" in str(e) or "400" in str(e):
                    max_total_tokens = int(max_total_tokens * 0.8)
                    print(f"    [RETRY] {cache_key}: context limit, new budget {max_total_tokens}")
                    await asyncio.sleep(1)
                else:
                    if attempt == max_retries - 1:
                        print(f"    [ERROR] {cache_key}: {e}")
                        return {"query_entities": [], "doc_entities": [[] for _ in range(len(titles))]}, str(e)
                    await asyncio.sleep(2 ** attempt)
        return {"query_entities": [], "doc_entities": [[] for _ in range(len(titles))]}, "Max retries exceeded"


# ------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------

def load_cache(path: str):
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_cache(path: str, data: dict, quiet: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    if not quiet:
        print(f"  Cached to {path}")


# ------------------------------------------------------------
# Reranking + reporting
# ------------------------------------------------------------

def rerank_by_entity_count(doc_indices, doc_entities_list):
    """Re-rank documents by the number of common entities (descending)."""
    scored = [(len(ents), rank, doc_idx) for rank, (doc_idx, ents) in enumerate(zip(doc_indices, doc_entities_list))]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [item[2] for item in scored]


def generate_report(
    all_results,
    output_path,
    reranker_name,
    llm_model_name,
    llm_top_k,
    max_concurrent: Optional[int] = None,
    title: str = "LLM Entity-Count Reranker Evaluation Report",
):
    mrr_k = [1, 3, 5, 10, 50, 100]
    hr_k = [1, 5, 10, 50, 100]

    lines = [
        f"# {title}\n",
        f"**Reranker:** `{reranker_name}`",
        f"**LLM for Entity Extraction:** `{llm_model_name}`",
        f"**Documents scored by LLM (top-K):** {llm_top_k}",
        f"**Reranking strategy:** entity count (descending), stable on ties",
    ]
    if max_concurrent is not None:
        lines.append(f"**Max concurrent:** {max_concurrent}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Mean Reciprocal Rank (MRR@K)\n")
    header = "| Split | " + " | ".join(f"MRR@{k}" for k in mrr_k) + " | Queries |"
    sep = "|---" + "|---" * len(mrr_k) + "|---|"
    lines += [header, sep]
    for split_name, r in all_results.items():
        row = f"| {split_name} | " + " | ".join(f"{r['mrr'].get(k, 0.0):.4f}" for k in mrr_k) + f" | {r['n_queries']} |"
        lines.append(row)

    lines.append("\n## Hit Rate (HR@K)\n")
    header = "| Split | " + " | ".join(f"HR@{k}" for k in hr_k) + " | Queries |"
    sep = "|---" + "|---" * len(hr_k) + "|---|"
    lines += [header, sep]
    for split_name, r in all_results.items():
        row = f"| {split_name} | " + " | ".join(f"{r['hr'].get(k, 0.0):.4f}" for k in hr_k) + f" | {r['n_queries']} |"
        lines.append(row)

    lines.append("\n## Entity Count Statistics\n")
    for split_name, r in all_results.items():
        cnts = r.get("entity_count_stats", {})
        if cnts:
            lines.append(f"### {split_name}\n")
            lines.append(f"- Mean entity count: {cnts.get('mean', 0):.2f}")
            lines.append(f"- Median entity count: {cnts.get('median', 0):.1f}")
            lines.append(f"- Max entity count: {cnts.get('max', 0)}\n")

    lines.append("## Ground Truth Rank Distribution\n")
    lines.append("| Split | Mean Rank | Median Rank | Rank 1 | Rank 2-5 | Rank 6-10 | Rank 11-50 | Rank 51-100 | Not Found |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for split_name, r in all_results.items():
        ranks = r.get("gt_ranks", [])
        if not ranks:
            continue
        found_ranks = [x for x in ranks if x > 0]
        not_found = sum(1 for x in ranks if x == -1)
        if found_ranks:
            mean_rank = float(np.mean(found_ranks))
            median_rank = float(np.median(found_ranks))
        else:
            mean_rank = float("nan")
            median_rank = float("nan")
        lines.append(
            f"| {split_name} | {mean_rank:.1f} | {median_rank:.0f} | "
            f"{sum(1 for x in ranks if x == 1)} | {sum(1 for x in ranks if 2 <= x <= 5)} | "
            f"{sum(1 for x in ranks if 6 <= x <= 10)} | {sum(1 for x in ranks if 11 <= x <= 50)} | "
            f"{sum(1 for x in ranks if 51 <= x <= 100)} | {not_found} |"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {output_path}")


# ------------------------------------------------------------
# Reranker indices loader
# ------------------------------------------------------------

def _load_npz_indices(path: str):
    data = np.load(path, allow_pickle=True)
    if "indices" in data:
        return data["indices"]
    if "arr_0" in data:
        return data["arr_0"]
    if len(data.keys()) == 1:
        key = list(data.keys())[0]
        return data[key]
    raise ValueError(f"Unknown structure in {path}, keys={list(data.keys())}")


def load_reranker_indices(cache_dir: str, reranker_slug: str, languages: list, cache_tag: str = None):
    """Load reranker indices from various possible file formats."""
    all_indices = {}
    cache_tag_suffix = f"_{cache_tag}" if cache_tag else ""

    for lang in languages:
        found = False

        patterns = [
            f"*rerank_{reranker_slug}_{lang}.npz",
            f"*{cache_tag_suffix}_rerank_{reranker_slug}_{lang}.npz",
            f"*{cache_tag_suffix}_rerank_{reranker_slug}_{lang}_dev.npz",
            f"*rerank_{reranker_slug}_{lang}_dev.npz",
            f"*rerank_Qwen_Qwen3-Reranker-8B_{lang}.npz",
            f"*rerank_Qwen_Qwen3-Reranker-8B_{lang}_dev.npz",
        ]

        import glob

        for pattern in patterns:
            matches = glob.glob(os.path.join(cache_dir, pattern))
            if matches:
                print(f"  {lang.upper()}: loading from {os.path.basename(matches[0])}")
                all_indices[lang] = _load_npz_indices(matches[0])
                found = True
                break

        if not found:
            shard_patterns = [
                os.path.join(cache_dir, f"*rerank_{reranker_slug}_{lang}_dev_shard*.npz"),
                os.path.join(cache_dir, f"*rerank_Qwen_Qwen3-Reranker-8B_{lang}_dev_shard*.npz"),
            ]
            shard_files = []
            for sp in shard_patterns:
                shard_files.extend(sorted(glob.glob(sp)))

            if shard_files:
                print(f"  {lang.upper()}: found {len(shard_files)} shard files, combining...")
                all_shard_indices = []
                for shard_file in shard_files:
                    indices = _load_npz_indices(shard_file)
                    all_shard_indices.extend(indices)
                all_indices[lang] = np.array(all_shard_indices, dtype=object)
                print(f"    Loaded {len(all_shard_indices)} queries for {lang}")
                found = True

        if not found:
            print(f"  {lang.upper()}: not found")

    return all_indices


# ------------------------------------------------------------
# Batch mode (v3) with optional sharding
# ------------------------------------------------------------

def run_batch_shard(shard_args: Dict[str, Any]) -> Dict[str, Any]:
    shard_id = shard_args["shard_id"]
    num_shards = shard_args["num_shards"]
    dataset_dir = shard_args["dataset_dir"]
    cache_dir = shard_args["cache_dir"]
    output_dir = shard_args["output_dir"]
    model_path = shard_args["model_path"]
    reranker = shard_args["reranker"]
    llm_chunk_size = shard_args["llm_chunk_size"]
    repetition = shard_args["repetition"]
    api_backend = shard_args["api_backend"]
    vllm_url = shard_args["vllm_url"]
    scads_api_key = shard_args["scads_api_key"]
    openai_api_key = shard_args["openai_api_key"]
    openai_base_url = shard_args["openai_base_url"]
    llm_model = shard_args["llm_model"]
    llm_top_k = shard_args["llm_top_k"]
    max_concurrent = shard_args["max_concurrent"]
    languages = shard_args["languages"]
    max_queries = shard_args["max_queries"]
    force_rescore = shard_args["force_rescore"]
    no_metadata = shard_args["no_metadata"]
    cache_save_interval = shard_args["cache_save_interval"]
    cache_tag = shard_args.get("cache_tag")

    model_slug = get_model_slug(model_path)
    reranker_slug = reranker.replace("/", "_")

    corpus_path = os.path.join(dataset_dir, "collection_data.json")
    pubkeys, doc_texts, doc_titles = load_collection(corpus_path, include_metadata=not no_metadata)

    chunk_size = llm_chunk_size if llm_chunk_size > 0 else None
    chunk_suffix = f"_chunk{llm_chunk_size}" if chunk_size else ""

    all_reranker_indices = load_reranker_indices(cache_dir, reranker_slug, languages, cache_tag)
    if not all_reranker_indices:
        raise FileNotFoundError(f"No reranker results found for '{reranker}' in {cache_dir}")

    llm_slug = llm_model.replace("/", "--").replace(".", "_")
    backend_suffix = api_backend + "_batch"
    shard_suffix = f"_shard{shard_id}"

    scores_cache_path = os.path.join(
        cache_dir,
        f"entity_scores_v3_{reranker_slug}_{llm_slug}_top{llm_top_k}_{backend_suffix}_rep{repetition}{chunk_suffix}{shard_suffix}.json",
    )
    entities_cache_path = os.path.join(
        cache_dir,
        f"entity_extract_v3_{reranker_slug}_{llm_slug}_top{llm_top_k}_{backend_suffix}{chunk_suffix}{shard_suffix}.json",
    )

    cached_scores = None if force_rescore else load_cache(scores_cache_path)
    cached_entities = None if force_rescore else load_cache(entities_cache_path)

    need_scoring = cached_scores is None or cached_entities is None
    client = None
    all_entity_scores = {}
    all_entity_info = {}

    if need_scoring:
        if api_backend == "vllm":
            base_url = vllm_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        elif api_backend == "scads":
            client = AsyncOpenAI(base_url="https://llm.scads.ai/v1", api_key=scads_api_key)
        else:
            base_url = openai_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            api_key = require_openai_api_key(openai_api_key, "openai backend")
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        if not force_rescore:
            partial_scores = load_cache(scores_cache_path)
            partial_entities = load_cache(entities_cache_path)
            if partial_scores is not None:
                all_entity_scores = partial_scores
            if partial_entities is not None:
                all_entity_info = partial_entities

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    all_results = {}
    mrr_k = [1, 3, 5, 10, 50, 100]
    hr_k = [1, 5, 10, 50, 100]
    all_queries_combined = []
    all_reranked_combined = []

    for lang in languages:
        dev_file = os.path.join(dataset_dir, f"{lang}_dev.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue

        queries = load_queries(dev_file)
        reranker_indices = all_reranker_indices[lang]

        original_queries = {}
        if lang in ("de", "fr"):
            original_queries = load_original_queries(dataset_dir, lang)

        if max_queries is not None:
            queries = queries[:max_queries]
            reranker_indices = reranker_indices[:max_queries]

        total_queries = len(queries)
        shard_size = (total_queries + num_shards - 1) // num_shards
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_queries)
        queries = queries[start_idx:end_idx]
        reranker_indices = reranker_indices[start_idx:end_idx]

        effective_top_k = min(llm_top_k, len(reranker_indices[0]) if len(reranker_indices) > 0 else 0)

        reranked_for_eval = []
        entity_counts_all = []
        gt_ranks = []

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            cache_key = f"{lang}_{start_idx + qi}"
            n_available = len(reranker_indices[qi])
            eff_k = min(llm_top_k, n_available)

            scores = None
            info = None

            if cached_scores is not None and cache_key in cached_scores:
                scores = cached_scores[cache_key]
            if cached_entities is not None and cache_key in cached_entities:
                info = cached_entities[cache_key]

            if scores is None and cache_key in all_entity_scores:
                scores = all_entity_scores[cache_key]
            if info is None and cache_key in all_entity_info:
                info = all_entity_info[cache_key]

            if scores is None or info is None:
                top_indices = reranker_indices[qi][:eff_k].tolist()
                top_docs = [doc_texts[idx] for idx in top_indices]
                if need_scoring:
                    orig_text = original_queries.get(q_idx, q_text)
                    scores, info = loop.run_until_complete(
                        score_and_extract_entities_batch(
                            client,
                            llm_model,
                            orig_text,
                            q_text,
                            top_docs,
                            cache_key=cache_key,
                            chunk_size=chunk_size,
                        )
                    )
                    all_entity_scores[cache_key] = scores
                    all_entity_info[cache_key] = info
                    if (qi + 1) % cache_save_interval == 0:
                        save_cache(scores_cache_path, all_entity_scores, quiet=True)
                        save_cache(entities_cache_path, all_entity_info, quiet=True)
                else:
                    scores = [0] * eff_k
                    info = {"query_entities": [], "doc_entities": [[] for _ in range(eff_k)]}

            doc_entities_list = info.get("doc_entities", [[] for _ in range(eff_k)])
            if len(doc_entities_list) < eff_k:
                doc_entities_list += [[] for _ in range(eff_k - len(doc_entities_list))]

            for ent_list in doc_entities_list:
                entity_counts_all.append(len(ent_list))

            top_indices = reranker_indices[qi][:eff_k].tolist()
            reranked_top = rerank_by_entity_count(top_indices, doc_entities_list)
            remaining = reranker_indices[qi][eff_k:].tolist()
            final_ranking = reranked_top + remaining
            reranked_for_eval.append(final_ranking)

            try:
                gt_pos = final_ranking.index(pubkeys.index(q_pubkey))
                gt_ranks.append(gt_pos + 1)
            except ValueError:
                gt_ranks.append(-1)

        mrr = evaluate_mrr(queries, reranked_for_eval, pubkeys, list_k=mrr_k)
        hr = evaluate_hit_rate(queries, reranked_for_eval, pubkeys, list_k=hr_k)

        entity_count_stats = {}
        if entity_counts_all:
            entity_count_stats = {
                "mean": float(np.mean(entity_counts_all)),
                "median": float(np.median(entity_counts_all)),
                "max": int(np.max(entity_counts_all)),
            }

        all_results[f"{lang.upper()} Dev"] = {
            "mrr": mrr,
            "hr": hr,
            "n_queries": len(queries),
            "gt_ranks": gt_ranks,
            "entity_count_stats": entity_count_stats,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

        all_queries_combined.extend(
            [(q_idx, q_text, q_pubkey, start_idx + i) for i, (q_idx, q_text, q_pubkey) in enumerate(queries)]
        )
        all_reranked_combined.extend(reranked_for_eval)

    if need_scoring:
        save_cache(scores_cache_path, all_entity_scores, quiet=True)
        save_cache(entities_cache_path, all_entity_info, quiet=True)

    per_lang_reranked = {}
    for lang in languages:
        dev_file = os.path.join(dataset_dir, f"{lang}_dev.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue
        queries = load_queries(dev_file)
        if max_queries is not None:
            queries = queries[:max_queries]

        total_queries = len(queries)
        shard_size = (total_queries + num_shards - 1) // num_shards
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_queries)

        lang_reranked = []
        lang_offset = 0
        query_ids_in_shard = [q2[0] for q2 in queries[start_idx:end_idx]]
        for q in all_queries_combined:
            if q[0] in query_ids_in_shard:
                break
            lang_offset += 1

        n_in_shard = end_idx - start_idx
        lang_reranked = all_reranked_combined[lang_offset:lang_offset + n_in_shard]
        per_lang_reranked[lang] = np.array(lang_reranked, dtype=object)

    reranked_cache = os.path.join(
        cache_dir,
        f"{model_slug}_rerank_{reranker_slug}_entity_v3_{llm_slug}_{backend_suffix}{chunk_suffix}{shard_suffix}_indices.npz",
    )
    np.savez(reranked_cache, **per_lang_reranked)

    if client is not None:
        loop.run_until_complete(client.close())

    return {
        "shard_id": shard_id,
        "results": all_results,
        "queries_combined": all_queries_combined,
        "reranked_combined": all_reranked_combined,
        "cache_paths": {
            "scores": scores_cache_path,
            "entities": entities_cache_path,
            "indices": reranked_cache,
        },
    }


def merge_batch_shards(shard_results: list, args, model_slug, reranker_slug, llm_slug, backend_suffix, chunk_suffix):
    all_results = {}
    mrr_k = [1, 3, 5, 10, 50, 100]
    hr_k = [1, 5, 10, 50, 100]

    entity_counts_by_lang = {lang: [] for lang in args.languages}

    for shard_result in shard_results:
        for split_name, r in shard_result["results"].items():
            lang = split_name.split()[0].lower()
            if split_name not in all_results:
                all_results[split_name] = {
                    "mrr": {k: [] for k in mrr_k},
                    "hr": {k: [] for k in hr_k},
                    "n_queries": 0,
                    "gt_ranks": [],
                    "entity_count_stats": {},
                }

            for k in mrr_k:
                all_results[split_name]["mrr"][k].append(r["mrr"].get(k, 0.0))
            for k in hr_k:
                all_results[split_name]["hr"][k].append(r["hr"].get(k, 0.0))
            all_results[split_name]["n_queries"] += r["n_queries"]
            all_results[split_name]["gt_ranks"].extend(r.get("gt_ranks", []))

            if "entity_count_stats" in r and r["entity_count_stats"]:
                if "mean" in r["entity_count_stats"]:
                    entity_counts_by_lang[lang].extend([r["entity_count_stats"]["mean"]] * r["n_queries"])

    for split_name in all_results:
        for k in mrr_k:
            if all_results[split_name]["mrr"][k]:
                all_results[split_name]["mrr"][k] = sum(all_results[split_name]["mrr"][k]) / len(
                    all_results[split_name]["mrr"][k]
                )
        for k in hr_k:
            if all_results[split_name]["hr"][k]:
                all_results[split_name]["hr"][k] = sum(all_results[split_name]["hr"][k]) / len(
                    all_results[split_name]["hr"][k]
                )

        lang = split_name.split()[0].lower()
        if entity_counts_by_lang[lang]:
            all_results[split_name]["entity_count_stats"] = {
                "mean": float(np.mean(entity_counts_by_lang[lang])),
                "median": float(np.median(entity_counts_by_lang[lang])),
                "max": int(np.max(entity_counts_by_lang[lang])),
            }

    per_lang_reranked = {}
    for lang in args.languages:
        dev_file = os.path.join(args.dataset_dir, f"{lang}_dev.json")
        if not os.path.isfile(dev_file):
            continue
        queries = load_queries(dev_file)
        if args.max_queries is not None:
            queries = queries[:args.max_queries]

        lang_indices = []
        for shard_id in range(args.parallel_shards):
            shard_cache = os.path.join(
                args.cache_dir,
                f"{model_slug}_rerank_{reranker_slug}_entity_v3_{llm_slug}_{backend_suffix}{chunk_suffix}_shard{shard_id}_indices.npz",
            )
            if os.path.isfile(shard_cache):
                data = np.load(shard_cache, allow_pickle=True)
                if lang in data:
                    lang_indices.extend(data[lang])

        if lang_indices:
            per_lang_reranked[lang] = np.array(lang_indices, dtype=object)

    reranked_cache = os.path.join(
        args.cache_dir,
        f"{model_slug}_rerank_{reranker_slug}_entity_v3_{llm_slug}_{backend_suffix}{chunk_suffix}_indices.npz",
    )
    np.savez(reranked_cache, **per_lang_reranked)
    print(f"  Merged reranked indices cached to {reranked_cache}")

    merged_scores = {}
    merged_entities = {}
    for shard_id in range(args.parallel_shards):
        shard_scores_path = os.path.join(
            args.cache_dir,
            f"entity_scores_v3_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}_rep{args.repetition}{chunk_suffix}_shard{shard_id}.json",
        )
        shard_entities_path = os.path.join(
            args.cache_dir,
            f"entity_extract_v3_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}{chunk_suffix}_shard{shard_id}.json",
        )

        shard_scores = load_cache(shard_scores_path)
        shard_entities = load_cache(shard_entities_path)

        if shard_scores:
            merged_scores.update(shard_scores)
        if shard_entities:
            merged_entities.update(shard_entities)

    if merged_scores:
        merged_scores_path = os.path.join(
            args.cache_dir,
            f"entity_scores_v3_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}_rep{args.repetition}{chunk_suffix}.json",
        )
        save_cache(merged_scores_path, merged_scores, quiet=False)

    if merged_entities:
        merged_entities_path = os.path.join(
            args.cache_dir,
            f"entity_extract_v3_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}{chunk_suffix}.json",
        )
        save_cache(merged_entities_path, merged_entities, quiet=False)

    lang_suffix = "_".join(sorted(args.languages))
    rep_suffix = f"_rep{args.repetition}" if hasattr(args, "repetition") else ""
    short_llm = args.llm_model.split("/")[-1]
    report_name = (
        f"eval_entity_v3_{reranker_slug}_{short_llm}_top{args.llm_top_k}_{backend_suffix}_{lang_suffix}{rep_suffix}.md"
    )
    report_path = os.path.join(args.output_dir, report_name)
    generate_report(all_results, report_path, args.reranker, args.llm_model, args.llm_top_k)

    return all_results


def run_batch_single(args):
    print("Loading corpus...")
    corpus_path = os.path.join(args.dataset_dir, "collection_data.json")
    pubkeys, doc_texts, doc_titles = load_collection(corpus_path, include_metadata=not args.no_metadata)
    print(f"  {len(pubkeys)} documents loaded")

    chunk_size = args.llm_chunk_size if args.llm_chunk_size > 0 else None
    chunk_suffix = f"_chunk{args.llm_chunk_size}" if chunk_size else ""

    print("Loading per-language reranker caches...")
    reranker_slug = args.reranker.replace("/", "_")
    all_reranker_indices = load_reranker_indices(args.cache_dir, reranker_slug, args.languages, args.cache_tag)

    if not all_reranker_indices:
        raise FileNotFoundError(f"No reranker results found for '{args.reranker}' in {args.cache_dir}")

    llm_slug = args.llm_model.replace("/", "--").replace(".", "_")
    backend_suffix = args.api_backend + "_batch"

    scores_cache_path = os.path.join(
        args.cache_dir,
        f"entity_scores_v3_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}_rep{args.repetition}{chunk_suffix}.json",
    )
    entities_cache_path = os.path.join(
        args.cache_dir,
        f"entity_extract_v3_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}{chunk_suffix}.json",
    )

    cached_scores = None if args.force_rescore else load_cache(scores_cache_path)
    cached_entities = None if args.force_rescore else load_cache(entities_cache_path)
    if cached_scores is not None:
        print(f"  Loaded cached scores from {scores_cache_path}")
    if cached_entities is not None:
        print(f"  Loaded cached entities from {entities_cache_path}")

    need_scoring = cached_scores is None or cached_entities is None
    client = None
    all_entity_scores = {}
    all_entity_info = {}

    if need_scoring:
        if args.api_backend == "vllm":
            print(f"\nConnecting to vLLM at {args.vllm_url}")
            base_url = args.vllm_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        elif args.api_backend == "scads":
            print("\nConnecting to ScaDS.AI API (batch mode)")
            print(f"  Model: {args.llm_model}")
            client = AsyncOpenAI(base_url="https://llm.scads.ai/v1", api_key=args.scads_api_key)
        else:
            base_url = args.openai_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            api_key = require_openai_api_key(args.openai_api_key, "openai backend")
            print(f"\nConnecting to OpenAI-compatible API at {base_url}")
            print(f"  Model: {args.llm_model}")
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        if not args.force_rescore:
            partial_scores = load_cache(scores_cache_path)
            partial_entities = load_cache(entities_cache_path)
            if partial_scores is not None:
                all_entity_scores = partial_scores
                print(f"  Resuming scores: {len(partial_scores)} queries already done")
            if partial_entities is not None:
                all_entity_info = partial_entities
                print(f"  Resuming entities: {len(partial_entities)} queries already done")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    all_results = {}
    mrr_k = [1, 3, 5, 10, 50, 100]
    hr_k = [1, 5, 10, 50, 100]
    all_queries_combined = []
    all_reranked_combined = []

    for lang in args.languages:
        dev_file = os.path.join(args.dataset_dir, f"{lang}_dev.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue

        queries = load_queries(dev_file)
        reranker_indices = all_reranker_indices[lang]

        original_queries = {}
        if lang in ("de", "fr"):
            original_queries = load_original_queries(args.dataset_dir, lang)

        if args.max_queries is not None:
            queries = queries[:args.max_queries]
            reranker_indices = reranker_indices[:args.max_queries]

        if args.shard_id is not None and args.num_shards > 1:
            total_queries = len(queries)
            shard_size = (total_queries + args.num_shards - 1) // args.num_shards
            start_idx = args.shard_id * shard_size
            end_idx = min(start_idx + shard_size, total_queries)
            queries = queries[start_idx:end_idx]
            reranker_indices = reranker_indices[start_idx:end_idx]
            print(f"  Shard {args.shard_id}/{args.num_shards}: processing queries {start_idx}-{end_idx - 1}")

        effective_top_k = min(args.llm_top_k, len(reranker_indices[0]) if len(reranker_indices) > 0 else 0)
        print(f"\n{'=' * 60}")
        print(f"  {lang.upper()}: {len(queries)} queries, top-{effective_top_k} docs (entity-count reranking)")
        if original_queries:
            print(f"  Original {lang.upper()} queries loaded for bilingual prompting")
        print(f"{'=' * 60}")

        reranked_for_eval = []
        entity_counts_all = []
        gt_ranks = []
        t0 = time.time()

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            cache_key = f"{lang}_{qi}"
            n_available = len(reranker_indices[qi])
            eff_k = min(args.llm_top_k, n_available)

            scores = None
            info = None

            if cached_scores is not None and cache_key in cached_scores:
                scores = cached_scores[cache_key]
            if cached_entities is not None and cache_key in cached_entities:
                info = cached_entities[cache_key]

            if scores is None and cache_key in all_entity_scores:
                scores = all_entity_scores[cache_key]
            if info is None and cache_key in all_entity_info:
                info = all_entity_info[cache_key]

            if scores is None or info is None:
                top_indices = reranker_indices[qi][:eff_k].tolist()
                top_docs = [doc_texts[idx] for idx in top_indices]
                if need_scoring:
                    orig_text = original_queries.get(q_idx, q_text)
                    scores, info = loop.run_until_complete(
                        score_and_extract_entities_batch(
                            client,
                            args.llm_model,
                            orig_text,
                            q_text,
                            top_docs,
                            cache_key=cache_key,
                            chunk_size=chunk_size,
                        )
                    )
                    all_entity_scores[cache_key] = scores
                    all_entity_info[cache_key] = info
                    if (qi + 1) % args.cache_save_interval == 0:
                        save_cache(scores_cache_path, all_entity_scores, quiet=True)
                        save_cache(entities_cache_path, all_entity_info, quiet=True)
                else:
                    scores = [0] * eff_k
                    info = {"query_entities": [], "doc_entities": [[] for _ in range(eff_k)]}

            doc_entities_list = info.get("doc_entities", [[] for _ in range(eff_k)])
            if len(doc_entities_list) < eff_k:
                doc_entities_list += [[] for _ in range(eff_k - len(doc_entities_list))]

            for ent_list in doc_entities_list:
                entity_counts_all.append(len(ent_list))

            top_indices = reranker_indices[qi][:eff_k].tolist()

            reranked_top = rerank_by_entity_count(top_indices, doc_entities_list)
            remaining = reranker_indices[qi][eff_k:].tolist()
            final_ranking = reranked_top + remaining
            reranked_for_eval.append(final_ranking)

            try:
                gt_pos = final_ranking.index(pubkeys.index(q_pubkey))
                gt_ranks.append(gt_pos + 1)
            except ValueError:
                gt_ranks.append(-1)

            elapsed = time.time() - t0
            n_query_ents = len(info.get("query_entities", []))
            print(f"    Query {qi + 1}/{len(queries)} | {elapsed:.1f}s | query_entities={n_query_ents}")

        mrr = evaluate_mrr(queries, reranked_for_eval, pubkeys, list_k=mrr_k)
        hr = evaluate_hit_rate(queries, reranked_for_eval, pubkeys, list_k=hr_k)

        entity_count_stats = {}
        if entity_counts_all:
            entity_count_stats = {
                "mean": float(np.mean(entity_counts_all)),
                "median": float(np.median(entity_counts_all)),
                "max": int(np.max(entity_counts_all)),
            }

        all_results[f"{lang.upper()} Dev"] = {
            "mrr": mrr,
            "hr": hr,
            "n_queries": len(queries),
            "gt_ranks": gt_ranks,
            "entity_count_stats": entity_count_stats,
        }
        print(f"  MRR@5={mrr[5]:.4f}  MRR@10={mrr[10]:.4f}  HR@1={hr[1]:.4f}  HR@10={hr[10]:.4f}")

        all_queries_combined.extend(queries)
        all_reranked_combined.extend(reranked_for_eval)

    macro = macro_average_results(all_results, args.languages)
    if macro is not None:
        combined_gt = []
        for r in all_results.values():
            combined_gt.extend(r.get("gt_ranks", []))
        macro["gt_ranks"] = combined_gt
        all_results["Combined"] = macro
        print(f"\n  Combined (Macro-Average): MRR@5={macro['mrr'][5]:.4f}  HR@1={macro['hr'][1]:.4f}")

    if need_scoring:
        save_cache(scores_cache_path, all_entity_scores)
        save_cache(entities_cache_path, all_entity_info)

    model_slug = get_model_slug(args.model_path)
    reranker_slug = args.reranker.replace("/", "_")

    per_lang_reranked = {}
    offset = 0
    for lang in args.languages:
        dev_file = os.path.join(args.dataset_dir, f"{lang}_dev.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue
        queries = load_queries(dev_file)
        if args.max_queries is not None:
            queries = queries[:args.max_queries]
        n = len(queries)
        per_lang_reranked[lang] = np.array(all_reranked_combined[offset:offset + n], dtype=object)
        offset += n

    reranked_cache = os.path.join(
        args.cache_dir,
        f"{model_slug}_rerank_{reranker_slug}_entity_v3_{llm_slug}_{backend_suffix}{chunk_suffix}_indices.npz",
    )
    np.savez(reranked_cache, **per_lang_reranked)
    print(f"  Final reranked indices cached to {reranked_cache}")

    short_llm = args.llm_model.split("/")[-1]
    lang_suffix = "_".join(sorted(args.languages))
    shard_suffix = f"_shard{args.shard_id}" if args.shard_id is not None else ""
    rep_suffix = f"_rep{args.repetition}" if hasattr(args, "repetition") else ""
    report_name = (
        f"eval_entity_v3_{reranker_slug}_{short_llm}_top{args.llm_top_k}_{backend_suffix}{shard_suffix}_{lang_suffix}{rep_suffix}.md"
    )
    report_path = os.path.join(args.output_dir, report_name)
    generate_report(all_results, report_path, args.reranker, args.llm_model, args.llm_top_k)

    if client is not None:
        loop.run_until_complete(client.close())

    print("\nDone! Outputs:")
    print(f"  1. Scores cache:   {scores_cache_path}")
    print(f"  2. Entities cache: {entities_cache_path}")
    print(f"  3. Report:         {report_path}")


# ------------------------------------------------------------
# vLLM async mode (v3.5)
# ------------------------------------------------------------

async def run_vllm_async_mode(args):
    model_slug = get_model_slug(args.model_path)
    reranker_slug = args.reranker.replace("/", "_")

    print("Loading corpus...")
    corpus_path = os.path.join(args.dataset_dir, "collection_data.json")
    pubkeys, doc_texts, titles, venues, authors, abstracts = load_collection(
        corpus_path, include_metadata=not args.no_metadata, return_all_metadata=True
    )
    print(f"  {len(pubkeys)} documents loaded")

    all_reranker_indices = load_reranker_indices(args.cache_dir, reranker_slug, args.languages, args.cache_tag)
    if not all_reranker_indices:
        raise FileNotFoundError(f"No reranker results found for '{args.reranker}' in {args.cache_dir}")

    if args.api_backend == "openai":
        base_url = args.openai_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        api_key = require_openai_api_key(args.openai_api_key, "openai-compatible async backend")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        backend_label = "OpenAI-compatible API"
    else:
        base_url = args.vllm_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        backend_label = "vLLM"
    print(f"\nConnected to {backend_label} at {base_url}")
    print(f"  Model: {args.llm_model}")
    print(f"  Max concurrent: {args.max_concurrent}")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    llm_slug = args.llm_model.replace("/", "--").replace(".", "_")
    run_id_suffix = f"_{args.run_id}" if args.run_id else ""
    entity_cache_path = os.path.join(
        args.cache_dir,
        f"entity_extract_v35_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_vllm{run_id_suffix}.json",
    )
    cached_entities = None if args.force_rescore else load_cache(entity_cache_path)
    all_entity_info = dict(cached_entities) if cached_entities else {}

    all_results = {}
    all_reranked_combined = []

    for lang in args.languages:
        dev_file = os.path.join(args.dataset_dir, f"{lang}_dev.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue

        queries = load_queries(dev_file)
        reranker_indices = all_reranker_indices[lang]

        original_queries = {}
        if lang in ("de", "fr"):
            original_queries = load_original_queries(args.dataset_dir, lang)

        if args.max_queries is not None:
            queries = queries[:args.max_queries]
            reranker_indices = reranker_indices[:args.max_queries]

        effective_top_k = min(args.llm_top_k, len(reranker_indices[0]))
        print(f"\n{'=' * 60}")
        print(f"  {lang.upper()}: {len(queries)} queries, top-{effective_top_k} candidates")
        print(f"{'=' * 60}")

        async def dummy_cached_result(result):
            return result

        t0 = time.time()
        tasks = []
        task_keys = []

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            cache_key = f"{lang}_{qi}"
            task_keys.append(cache_key)

            if cache_key in all_entity_info:
                tasks.append(dummy_cached_result((all_entity_info[cache_key], "Cached")))
            else:
                eff_k = min(args.llm_top_k, len(reranker_indices[qi]))
                top_indices = reranker_indices[qi][:eff_k].tolist()
                top_titles = [titles[idx] for idx in top_indices]
                top_venues = [venues[idx] for idx in top_indices]
                top_authors = [authors[idx] for idx in top_indices]
                top_abstracts = [abstracts[idx] for idx in top_indices]
                orig_text = original_queries.get(q_idx, q_text)
                tasks.append(
                    extract_entities_single_query(
                        client,
                        args.llm_model,
                        orig_text,
                        q_text,
                        top_titles,
                        top_venues,
                        top_authors,
                        top_abstracts,
                        cache_key,
                        semaphore,
                        max_total_tokens=args.max_context_tokens,
                    )
                )

        print(f"  Dispatching {len(tasks)} LLM calls with concurrency={args.max_concurrent}...")

        progress = {
            "done": 0,
            "total": len(tasks),
            "ok": 0,
            "fail": 0,
            "start_time": t0,
            "new_entries": 0,
        }
        cache_lock = asyncio.Lock()

        async def track_result(coro, cache_key):
            if asyncio.iscoroutine(coro):
                result = await coro
            else:
                result = coro
            progress["done"] += 1
            info, status = result
            if status in ("Success", "Cached"):
                progress["ok"] += 1
            else:
                progress["fail"] += 1
            elapsed = time.time() - progress["start_time"]
            rate = progress["done"] / elapsed if elapsed > 0 else 0
            eta = (progress["total"] - progress["done"]) / rate if rate > 0 else 0
            n_q_ents = len(info.get("query_entities", []))
            print(f"    {progress['done']}/{progress['total']} | {elapsed:.1f}s | query_entities={n_q_ents} | {status}")
            if progress["done"] % 100 == 0:
                print(
                    f"  === Progress: {progress['done']}/{progress['total']} "
                    f"(ok={progress['ok']}, fail={progress['fail']}) | "
                    f"{rate:.1f} queries/s | ETA {eta:.0f}s ==="
                )

            if cache_key not in all_entity_info:
                all_entity_info[cache_key] = info
                progress["new_entries"] += 1
                if progress["new_entries"] % 100 == 0:
                    async with cache_lock:
                        save_cache(entity_cache_path, all_entity_info, quiet=True)
                        print(f"  [Cache] Saved {progress['new_entries']} new entries to {entity_cache_path}")

            return result

        tracked_tasks = [track_result(task, key) for task, key in zip(tasks, task_keys)]
        results = await asyncio.gather(*tracked_tasks)

        elapsed = time.time() - t0
        print(f"  LLM calls completed in {elapsed:.1f}s")

        reranked_for_eval = []
        entity_counts_all = []
        gt_ranks = []

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            cache_key = f"{lang}_{qi}"
            entity_info = all_entity_info[cache_key]
            eff_k = min(args.llm_top_k, len(reranker_indices[qi]))
            top_indices = reranker_indices[qi][:eff_k].tolist()

            doc_entities_list = entity_info.get("doc_entities", [[] for _ in range(eff_k)])
            if len(doc_entities_list) < eff_k:
                doc_entities_list += [[] for _ in range(eff_k - len(doc_entities_list))]

            for ent_list in doc_entities_list:
                entity_counts_all.append(len(ent_list))

            reranked_top = rerank_by_entity_count(top_indices, doc_entities_list)
            remaining = reranker_indices[qi][eff_k:].tolist()
            final_ranking = reranked_top + remaining
            reranked_for_eval.append(final_ranking)

            try:
                gt_pos = final_ranking.index(pubkeys.index(q_pubkey))
                gt_ranks.append(gt_pos + 1)
            except ValueError:
                gt_ranks.append(-1)

        mrr = evaluate_mrr(queries, reranked_for_eval, pubkeys, list_k=[1, 3, 5, 10, 50, 100])
        hr = evaluate_hit_rate(queries, reranked_for_eval, pubkeys, list_k=[1, 5, 10, 50, 100])

        entity_count_stats = {}
        if entity_counts_all:
            entity_count_stats = {
                "mean": float(np.mean(entity_counts_all)),
                "median": float(np.median(entity_counts_all)),
                "max": int(np.max(entity_counts_all)),
            }

        all_results[f"{lang.upper()} Dev"] = {
            "mrr": mrr,
            "hr": hr,
            "n_queries": len(queries),
            "gt_ranks": gt_ranks,
            "entity_count_stats": entity_count_stats,
        }
        print(f"  MRR@5={mrr[5]:.4f}  HR@1={hr[1]:.4f}")
        all_reranked_combined.extend(reranked_for_eval)

        save_cache(entity_cache_path, all_entity_info, quiet=True)
        print(f"  [Cache] Saved after {lang.upper()} processing")

    macro = macro_average_results(all_results, args.languages)
    if macro:
        combined_gt = []
        for r in all_results.values():
            combined_gt.extend(r.get("gt_ranks", []))
        macro["gt_ranks"] = combined_gt
        all_results["Combined"] = macro

    save_cache(entity_cache_path, all_entity_info)
    report_path = os.path.join(
        args.output_dir,
        f"eval_entity_v35_{reranker_slug}_{llm_slug}_top{args.llm_top_k}{run_id_suffix}.md",
    )
    generate_report(
        all_results,
        report_path,
        args.reranker,
        args.llm_model,
        args.llm_top_k,
        max_concurrent=args.max_concurrent,
        title="LLM Entity-Count Reranker v3.5 Evaluation Report",
    )

    await client.close()
    print("\nDone!")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Entity-Count Reranker (merged v3 batch + v3.5 vLLM async)"
    )
    parser.add_argument("--engine", choices=["batch", "vllm_async"], default="batch")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="CT26/Dataset_translated",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="GRITLM_finetune/eval_cache_gritlm_translated",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="GRITLM_finetune/eval_results_translated",
    )
    parser.add_argument("--model_path", type=str, default="GRITLM")
    parser.add_argument("--reranker", type=str, default="Qwen/Qwen3-Reranker-8B")

    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_top_k", type=int, default=10)
    parser.add_argument("--languages", nargs="+", default=["en", "de", "fr"])
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--force_rescore", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--cache_tag", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--max_concurrent", type=int, default=50)
    parser.add_argument("--max_context_tokens", type=int, default=60000)

    parser.add_argument("--llm_chunk_size", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--parallel_shards", type=int, default=None)

    parser.add_argument("--api_backend", type=str, choices=["vllm", "scads", "openai"], default="vllm")
    parser.add_argument("--scads_api_key", type=str, default=None)
    parser.add_argument("--scads_disable_fallbacks", action="store_true")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--cache_save_interval", type=int, default=10)

    args = parser.parse_args()

    if args.llm_model is None:
        if args.engine == "vllm_async":
            args.llm_model = "Qwen/Qwen3.5-9B"
        else:
            args.llm_model = "moonshotai/Kimi-K2.6"

    if args.engine == "batch":
        if args.api_backend == "scads" and not args.scads_api_key:
            parser.error("--scads_api_key is required for scads backend")
        if args.api_backend == "openai":
            require_openai_api_key(args.openai_api_key, "openai backend")

        model_slug = get_model_slug(args.model_path)
        reranker_slug = args.reranker.replace("/", "_")

        if args.parallel_shards is not None and args.parallel_shards > 1:
            print(f"\n{'=' * 60}")
            print(f"Running in automatic parallel mode with {args.parallel_shards} shards")
            print(f"{'=' * 60}\n")

            shard_args_list = []
            for shard_id in range(args.parallel_shards):
                shard_args = {
                    "shard_id": shard_id,
                    "num_shards": args.parallel_shards,
                    "dataset_dir": args.dataset_dir,
                    "cache_dir": args.cache_dir,
                    "output_dir": args.output_dir,
                    "model_path": args.model_path,
                    "reranker": args.reranker,
                    "llm_chunk_size": args.llm_chunk_size,
                    "repetition": args.repetition,
                    "api_backend": args.api_backend,
                    "vllm_url": args.vllm_url,
                    "scads_api_key": args.scads_api_key,
                    "openai_api_key": args.openai_api_key,
                    "openai_base_url": args.openai_base_url,
                    "llm_model": args.llm_model,
                    "llm_top_k": args.llm_top_k,
                    "max_concurrent": args.max_concurrent,
                    "languages": args.languages,
                    "max_queries": args.max_queries,
                    "force_rescore": args.force_rescore,
                    "no_metadata": args.no_metadata,
                    "cache_save_interval": args.cache_save_interval,
                    "cache_tag": args.cache_tag,
                }
                shard_args_list.append(shard_args)

            print(f"Starting {args.parallel_shards} parallel shards...\n")
            with Pool(processes=args.parallel_shards) as pool:
                shard_results = pool.map(run_batch_shard, shard_args_list)

            print(f"\n{'=' * 60}")
            print(f"All {args.parallel_shards} shards completed!")
            print(f"{'=' * 60}\n")

            print("Merging results from all shards...\n")
            chunk_size = args.llm_chunk_size if args.llm_chunk_size > 0 else None
            chunk_suffix = f"_chunk{args.llm_chunk_size}" if chunk_size else ""
            llm_slug = args.llm_model.replace("/", "--").replace(".", "_")
            backend_suffix = args.api_backend + "_batch"

            merge_batch_shards(
                shard_results,
                args,
                model_slug,
                reranker_slug,
                llm_slug,
                backend_suffix,
                chunk_suffix,
            )

            print("\nDone! Parallel execution completed.")
            print(f"  Merged report saved to {args.output_dir}")
            return

        run_batch_single(args)
        return

    asyncio.run(run_vllm_async_mode(args))


if __name__ == "__main__":
    main()
