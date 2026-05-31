#!/usr/bin/env python3
"""
LLM single-best-paper reranker (merged v4 batch + v4.6 vLLM async).

Usage:
  # Batch mode (v4 behavior)
  python verification_reranker/llm_entity_reranker_single_best.py \
    --engine batch \
    --api_backend scads \
    --scads_api_key "$SCADS_API_KEY" \
    --llm_model moonshotai/Kimi-K2.6 \
    --llm_top_k 10 \
    --languages de \
    --split dev

    # Batch mode with OpenAI-compatible API
    python verification_reranker/llm_entity_reranker_single_best.py \
        --engine batch \
        --api_backend openai \
        --openai_api_key "$OPENAI_API_KEY" \
        --openai_base_url https://api.openai.com/v1 \
        --llm_model gpt-4.1 \
        --llm_top_k 10 \
        --languages en

  # vLLM async mode (v4.6 behavior)
  python verification_reranker/llm_entity_reranker_single_best.py \
    --engine vllm_async \
    --vllm_url http://localhost:8000/v1 \
    --llm_model Qwen/Qwen3.5-27B \
    --llm_top_k 10 \
    --max_concurrent 50 \
    --max_context_tokens 60000 \
    --languages en de fr
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

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

def load_original_queries(dataset_dir: str, lang: str, split: str = "dev") -> dict:
    """Load original-language queries from the 'copy' files for de/fr."""
    copy_file = os.path.join(dataset_dir, f"{lang}_{split} copy.json")
    if not os.path.isfile(copy_file):
        print(f"  No original-language file found: {copy_file}")
        return {}
    with open(copy_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {q["index"]: q["text"] for q in data}
    print(f"  Loaded {len(mapping)} original {lang.upper()} queries from copy file")
    return mapping


# ------------------------------------------------------------
# Prompts
# ------------------------------------------------------------

SINGLE_BEST_SYSTEM_PROMPT = """[Role & Objective]
You are an expert research assistant. Your task is to identify the single most relevant paper from a provided list of candidates based on a specific user query.

**Handling bilingual queries:**
The user may provide a single query (`QUERY: ...`) OR a pair (`QUERY (original language): ...` and `QUERY (translated to English): ...`).
- If both are given, the original is in French or German. Use it as the PRIMARY source for matching.
- The English translation is a SUPPLEMENT to help recognise cross-lingual synonyms and terminology.
- Always consider both versions when evaluating relevance.

[Instructions & Constraints]
Analyze the Query and the Candidates below. Select the ONE paper that is the most relevant. Before making your final decision, you must perform the following checks:
    Source Verification: Does the paper title appear verbatim (or near-verbatim) in the query? (e.g., "Ultrapotent antibodies...").
    Evidence Alignment: Does the paper's abstract explicitly support the specific claim made in the query? (e.g., If the query claims "increased deaths," does the paper's results section confirm increased deaths?)
    Contextual Specificity: Does the specific population, location, or intervention in the paper match the query? (e.g., "US Veterans" vs. "France"; "First Wave" vs. "Vaccinated").
    Exclusion Check: Are there other papers with similar keywords but contradictory conclusions? Discard them.

You MUST respond with ONLY a JSON object with exactly two keys:
- "reasoning": a brief string (2-4 sentences) explaining why this specific paper was chosen over the others based on the checks above. Mention any title matches or specific data points that aligned.
- "selected_paper": an integer (1-based), the paper number that is most relevant.

Example response:
{"reasoning": "Paper 3 title matches the drug name in the query verbatim. Its abstract confirms the 50% reduction claim in the specific population mentioned. No other paper addresses this exact intervention.", "selected_paper": 3}

No extra text, no explanation outside the JSON."""

SIMPLE_SYSTEM_PROMPT = """You are given a tweet and a list of research papers. Your task is to select the single most relevant paper to the tweet's core claim.

Definition of "most relevant":
- Directly addresses the main claim (not just keywords)
- Supports, contradicts, or specifically tests the tweet's central idea
- If tied, choose stronger methodology (peer-reviewed > preprint, larger N)

Begin your response with the number of the paper (1-10), then write a short reason.

If none are relevant, say "None" and explain why.

Example response:
3: This study directly reports the increased mortality found in the VA study.

Now answer."""


def build_user_message(query: str, translated_query: str, papers: list[str]) -> str:
    if translated_query and translated_query != query:
        return (
            f"Query (original language):\n\"{query}\"\n\n"
            f"Query (translated to English):\n\"{translated_query}\"\n\nCandidates:\n"
            + "\n".join(papers)
        )
    return f"Query:\n\"{query}\"\n\nCandidates:\n" + "\n".join(papers)


def build_tweet_message(tweet: str, translated_tweet: str | None, papers: list[str]) -> str:
    if translated_tweet and translated_tweet != tweet:
        tweet_block = (
            f"Tweet (original language):\n\"{tweet}\"\n\n"
            f"Tweet (translated to English):\n\"{translated_tweet}\"\n"
        )
    else:
        tweet_block = f"Tweet:\n\"{tweet}\"\n"
    papers_block = "Papers:\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(papers))
    return tweet_block + "\n" + papers_block


# ------------------------------------------------------------
# Robust JSON parsing (batch mode)
# ------------------------------------------------------------

_parse_fail_count = 0
_PARSE_FAIL_MAX_PRINT = 5


def _extract_balanced(text: str, open_char: str = "{", close_char: str = "}") -> str | None:
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
            if isinstance(obj, dict) and ("selected_paper" in obj or "reasoning" in obj):
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

    sel_match = re.search(r'"selected_paper"\s*:\s*(\d+)', content)
    if not sel_match:
        sel_match = re.search(r"[Ss]elected\s+[Pp]aper\s*:?\s*(\d+)", content)
    if not sel_match:
        sel_match = re.search(r"[Pp]aper\s+(\d+)", content)

    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
    if not reason_match:
        reason_match = re.search(r"[Rr]easoning\s*:?\s*(.+?)(?:\n|Selected|$)", content, re.DOTALL)

    if sel_match:
        return {
            "selected_paper": int(sel_match.group(1)),
            "reasoning": reason_match.group(1).strip() if reason_match else "Parsed from fallback regex",
        }

    _parse_fail_count += 1
    if _parse_fail_count <= _PARSE_FAIL_MAX_PRINT:
        print(f"    [WARN] JSON parse failed for {label} (failure #{_parse_fail_count}):")
        print(f"    Raw content (first 500 chars): {content[:500]}")
    elif _parse_fail_count == _PARSE_FAIL_MAX_PRINT + 1:
        print(f"    [WARN] Suppressing further parse failure logs (total: {_parse_fail_count})")
    return None


# ------------------------------------------------------------
# vLLM parsing (async mode)
# ------------------------------------------------------------

def parse_paper_selection(content: str, n_papers: int, label: str = "") -> tuple[int | None, str]:
    content = content.strip()

    if content.lower().startswith("none"):
        return None, content[:200]

    match = re.match(r"^(\d+)\s*[:.\-]?\s*(.*)", content, re.DOTALL)
    if match:
        paper_num = int(match.group(1))
        reason = match.group(2).strip()
        if 1 <= paper_num <= n_papers:
            return paper_num, reason[:200]
        print(f"    [WARN] {label}: number {paper_num} out of range")
        return None, "Out of range"

    nums = re.findall(r"\b([1-9][0-9]*)\b", content)
    if nums:
        return int(nums[0]), "Fallback"
    return None, "No number found"


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
# Batch mode selection (v4)
# ------------------------------------------------------------

async def select_single_best_batch(
    client: AsyncOpenAI,
    model: str,
    query: str,
    translated_query: str,
    doc_texts: list[str],
    cache_key: str = "",
    max_total_tokens: int = 250000,
    output_tokens_reserve: int = 4096,
) -> tuple[int | None, str]:
    enc = get_tiktoken_encoder("gpt-4")

    if query == translated_query:
        user_header = f"QUERY:\n{query}\n\n"
    else:
        user_header = (
            f"QUERY (original language):\n{query}\n\n"
            f"QUERY (translated to English):\n{translated_query}\n\n"
        )

    user_header += "THE CANDIDATES:\n"
    base_tokens = len(enc.encode(SINGLE_BEST_SYSTEM_PROMPT)) + len(enc.encode(user_header))

    papers_with_indices = [(i, f"Paper {i+1}\n{txt}\n") for i, txt in enumerate(doc_texts)]

    kept_papers = []
    kept_indices = []
    current_tokens = base_tokens
    for orig_idx, paper_block in papers_with_indices:
        block_tokens = len(enc.encode(paper_block))
        if current_tokens + block_tokens + output_tokens_reserve <= max_total_tokens:
            kept_papers.append(paper_block)
            kept_indices.append(orig_idx)
            current_tokens += block_tokens
        else:
            print(f"    [TRUNC] {cache_key}: kept {len(kept_papers)}/{len(doc_texts)} papers due to token limit")
            break

    if not kept_papers:
        kept_papers = [papers_with_indices[0][1][:500]]
        kept_indices = [0]

    user_message = user_header + "".join(kept_papers)

    extra_body = {}
    if "Kimi" in model:
        extra_body = {"chat_template_kwargs": {"thinking": False}}

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SINGLE_BEST_SYSTEM_PROMPT},
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
                print(f"    [ERROR] {cache_key}: Empty response from API")
                return None, ""
            print(f"    [INFO] {cache_key}: Used reasoning_content fallback")

        parsed = _try_parse_json_obj(content.strip(), label=cache_key)
        if parsed is None:
            return None, ""

        selected_1based = parsed.get("selected_paper")
        reasoning = parsed.get("reasoning", "")

        if selected_1based is None:
            print(f"    [WARN] {cache_key}: Parsed JSON but no selected_paper field, keys={list(parsed.keys())}")
            return None, ""

        selected_1based = int(selected_1based)
        if selected_1based < 1 or selected_1based > len(kept_indices):
            print(f"    [WARN] {cache_key}: selected_paper={selected_1based} out of range [1,{len(kept_indices)}]")
            return None, ""

        original_1based = kept_indices[selected_1based - 1] + 1
        return original_1based, reasoning

    except Exception as e:
        print(f"    [ERROR] {cache_key}: API call failed: {e}")
        if "content" in locals() and content:
            print(f"    Content (first 300 chars): {content[:300]}")
        return None, ""


# ------------------------------------------------------------
# vLLM async selection (v4.6)
# ------------------------------------------------------------

def format_paper_full(title: str, venue: str, authors: str, abstract: str) -> str:
    return f"Title: {title}\nVenue: {venue}\nAuthors: {authors}\nAbstract: {abstract}"


def format_paper_without_authors(title: str, venue: str, abstract: str) -> str:
    return f"Title: {title}\nVenue: {venue}\nAbstract: {abstract}"


async def score_single_query_vllm(
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
    output_tokens_reserve: int = 256,
    max_retries: int = 3,
) -> tuple[int | None, str]:
    async with semaphore:
        papers = [format_paper_full(titles[i], venues[i], authors[i], abstracts[i]) for i in range(len(titles))]
        remove_authors = False

        for attempt in range(max_retries):
            try:
                user_msg = build_tweet_message(query_text, translated_query, papers)
                sys_tokens = count_tokens(SIMPLE_SYSTEM_PROMPT, model)
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
                        {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
                        {"role": "user", "content": build_tweet_message(query_text, translated_query, papers)},
                    ],
                    max_tokens=output_tokens_reserve,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content or ""
                selected, reason = parse_paper_selection(content.strip(), len(papers), label=cache_key)
                return selected, reason

            except Exception as e:
                if "maximum context length" in str(e) or "400" in str(e):
                    max_total_tokens = int(max_total_tokens * 0.8)
                    print(f"    [RETRY] {cache_key}: context limit, new budget {max_total_tokens}")
                    await asyncio.sleep(1)
                else:
                    if attempt == max_retries - 1:
                        print(f"    [ERROR] {cache_key}: {e}")
                        return None, ""
                    await asyncio.sleep(2 ** attempt)
        return None, ""


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


def rerank_promote_single(doc_indices, selected_paper_1based):
    if selected_paper_1based is None:
        return list(doc_indices)
    pos = selected_paper_1based - 1
    if pos < 0 or pos >= len(doc_indices):
        return list(doc_indices)
    selected = doc_indices[pos]
    rest = [doc for i, doc in enumerate(doc_indices) if i != pos]
    return [selected] + rest


# ------------------------------------------------------------
# Report generation
# ------------------------------------------------------------

def generate_report(
    all_results,
    output_path,
    reranker_name,
    llm_model_name,
    llm_top_k,
    max_concurrent: Optional[int] = None,
    title: str = "LLM Single-Best Reranker Evaluation Report",
):
    mrr_k = [1, 3, 5, 10, 50, 100]
    hr_k = [1, 5, 10, 50, 100]

    lines = [
        f"# {title}\n",
        f"**Reranker:** `{reranker_name}`",
        f"**LLM for Paper Selection:** `{llm_model_name}`",
        f"**Top-K candidates:** {llm_top_k}",
    ]
    if max_concurrent is not None:
        lines.append(f"**Max concurrent:** {max_concurrent}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## MRR\n")
    header = "| Split | " + " | ".join(f"MRR@{k}" for k in mrr_k) + " | Queries |"
    lines.append(header)
    lines.append("|---" + "|---" * len(mrr_k) + "|---|")
    for split, r in all_results.items():
        row = (
            f"| {split} | "
            + " | ".join(f"{r['mrr'].get(k, 0.0):.4f}" for k in mrr_k)
            + f" | {r['n_queries']} |"
        )
        lines.append(row)

    lines.append("\n## HR\n")
    header = "| Split | " + " | ".join(f"HR@{k}" for k in hr_k) + " | Queries |"
    lines.append(header)
    lines.append("|---" + "|---" * len(hr_k) + "|---|")
    for split, r in all_results.items():
        row = (
            f"| {split} | "
            + " | ".join(f"{r['hr'].get(k, 0.0):.4f}" for k in hr_k)
            + f" | {r['n_queries']} |"
        )
        lines.append(row)

    lines.append("\n## LLM Selection Statistics\n")
    for split, r in all_results.items():
        sel = r.get("selection_stats", {})
        lines.append(f"### {split}\n")
        lines.append("| Metric | Count |")
        lines.append("|---|---|")
        lines.append(f"| Queries total | {sel.get('total', 0)} |")
        lines.append(f"| Successful selections | {sel.get('selected', 0)} |")
        lines.append(f"| Failed selections | {sel.get('failed', 0)} |")
        lines.append(f"| GT was in top-K candidates | {sel.get('gt_in_topk', 0)} |")
        lines.append(f"| LLM correctly selected GT | {sel.get('gt_selected', 0)} |")
        lines.append(f"| GT already rank 1 before LLM | {sel.get('gt_already_rank1', 0)} |")
        lines.append(f"| GT promoted to rank 1 by LLM | {sel.get('gt_promoted', 0)} |\n")

    lines.append("## Ground Truth Rank Distribution\n")
    lines.append("| Split | Mean Rank | Median Rank | Rank 1 | Rank 2-5 | Rank 6-10 | Rank 11-50 | Rank 51-100 | Not Found |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for split, r in all_results.items():
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
            f"| {split} | {mean_rank:.1f} | {median_rank:.0f} | "
            f"{sum(1 for x in ranks if x == 1)} | {sum(1 for x in ranks if 2 <= x <= 5)} | "
            f"{sum(1 for x in ranks if 6 <= x <= 10)} | {sum(1 for x in ranks if 11 <= x <= 50)} | "
            f"{sum(1 for x in ranks if 51 <= x <= 100)} | {not_found} |"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {output_path}")


# ------------------------------------------------------------
# Batch engine (v4)
# ------------------------------------------------------------

def run_batch_mode(args):
    if args.api_backend == "scads" and not args.scads_api_key:
        raise ValueError("--scads_api_key is required for scads backend")
    if args.api_backend == "openai":
        require_openai_api_key(args.openai_api_key, "openai backend")

    model_slug = get_model_slug(args.model_path)
    tagged_slug = f"{model_slug}_{args.cache_tag}" if args.cache_tag else model_slug
    reranker_slug = args.reranker.replace("/", "_")

    print("Loading corpus...")
    corpus_path = os.path.join(args.dataset_dir, "collection_data.json")
    pubkeys, doc_texts, doc_titles = load_collection(corpus_path, include_metadata=not args.no_metadata)
    print(f"  {len(pubkeys)} documents loaded")

    combined_path = os.path.join(args.cache_dir, f"{tagged_slug}_rerank_{reranker_slug}_indices.npz")
    all_reranker_indices = {}
    if os.path.isfile(combined_path):
        print(f"Loading combined reranker cache: {combined_path}")
        reranker_data = np.load(combined_path, allow_pickle=True)
        all_reranker_indices = {lang: reranker_data[lang] for lang in reranker_data.files}
    else:
        print("Loading per-language reranker caches...")
        for lang in args.languages:
            per_lang_path = os.path.join(
                args.cache_dir,
                f"{tagged_slug}_rerank_{reranker_slug}_{lang}_{args.split}.npz",
            )
            if os.path.isfile(per_lang_path):
                data = np.load(per_lang_path)
                all_reranker_indices[lang] = data["indices"]
                print(f"  {lang.upper()}: loaded")
            else:
                print(f"  {lang.upper()}: not found")

    if not all_reranker_indices:
        raise FileNotFoundError(f"No reranker results found for '{args.reranker}' in {args.cache_dir}")

    llm_slug = args.llm_model.replace("/", "--").replace(".", "_")
    backend_suffix = args.api_backend
    cache_tag_suffix = f"_{args.cache_tag}" if args.cache_tag else ""

    shared_cache_path = os.path.join(
        args.cache_dir,
        f"single_best_v4{cache_tag_suffix}_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}_{args.split}.json",
    )

    def _lang_cache_path(lang):
        return os.path.join(
            args.cache_dir,
            f"single_best_v4{cache_tag_suffix}_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_{backend_suffix}_{lang}_{args.split}.json",
        )

    shared_cache = None if args.force_rescore else load_cache(shared_cache_path)
    if shared_cache is not None:
        print(f"  Loaded shared cache ({len(shared_cache)} entries) from {shared_cache_path}")

    if args.api_backend == "vllm":
        print(f"\nConnecting to vLLM at {args.vllm_url}")
        base_url = args.vllm_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    elif args.api_backend == "scads":
        print("\nConnecting to ScaDS.AI API")
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
        dev_file = os.path.join(args.dataset_dir, f"{args.query_prefix}{lang}_{args.split}.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue

        queries = load_queries(dev_file)
        reranker_indices = all_reranker_indices[lang]

        original_queries = {}
        if lang in ("de", "fr"):
            original_queries = load_original_queries(args.dataset_dir, lang, args.split)

        if args.max_queries is not None:
            queries = queries[:args.max_queries]
            reranker_indices = reranker_indices[:args.max_queries]

        shard_start = (len(queries) * args.shard) // args.num_shards
        shard_end = (len(queries) * (args.shard + 1)) // args.num_shards
        if args.num_shards > 1:
            print(f"\n  Shard {args.shard}/{args.num_shards}: queries [{shard_start}:{shard_end}]")
        queries = queries[shard_start:shard_end]
        reranker_indices = reranker_indices[shard_start:shard_end]

        shard_suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
        lang_cache_path = _lang_cache_path(lang).replace(".json", f"{shard_suffix}.json")
        all_selections = {}
        if not args.force_rescore:
            partial = load_cache(lang_cache_path)
            if partial is not None:
                all_selections = partial
                print(f"  Resuming {lang.upper()}: {len(partial)} queries already done")

        effective_top_k = min(args.llm_top_k, len(reranker_indices[0]))
        print(f"\n{'=' * 60}")
        print(f"  {lang.upper()}: {len(queries)} queries, top-{effective_top_k} candidates (single-best selection)")
        if original_queries:
            print(f"  Original {lang.upper()} queries loaded for bilingual prompting")
        print(f"{'=' * 60}")

        reranked_for_eval = []
        gt_ranks = []
        sel_stats = {
            "total": len(queries),
            "selected": 0,
            "failed": 0,
            "gt_in_topk": 0,
            "gt_selected": 0,
            "gt_already_rank1": 0,
            "gt_promoted": 0,
        }
        t0 = time.time()

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            abs_idx = shard_start + qi
            cache_key = f"{lang}_{abs_idx}"
            n_available = len(reranker_indices[qi])
            eff_k = min(args.llm_top_k, n_available)

            selected_paper_1based = None
            reasoning = ""

            if cache_key in all_selections:
                entry = all_selections[cache_key]
                if isinstance(entry, dict):
                    selected_paper_1based = entry.get("selected_paper")
                    reasoning = entry.get("reasoning", "")
                else:
                    selected_paper_1based = entry
            elif shared_cache is not None and cache_key in shared_cache:
                entry = shared_cache[cache_key]
                if isinstance(entry, dict):
                    selected_paper_1based = entry.get("selected_paper")
                    reasoning = entry.get("reasoning", "")

            if selected_paper_1based is None:
                top_indices = reranker_indices[qi][:eff_k].tolist()
                top_docs = [doc_texts[idx] for idx in top_indices]
                orig_text = original_queries.get(q_idx, q_text)
                selected_paper_1based, reasoning = loop.run_until_complete(
                    select_single_best_batch(
                        client,
                        args.llm_model,
                        orig_text,
                        q_text,
                        top_docs,
                        cache_key=cache_key,
                    )
                )
                all_selections[cache_key] = {
                    "selected_paper": selected_paper_1based,
                    "reasoning": reasoning,
                }
                save_cache(lang_cache_path, all_selections, quiet=True)

            if selected_paper_1based is not None:
                sel_stats["selected"] += 1
            else:
                sel_stats["failed"] += 1

            top_indices = reranker_indices[qi][:eff_k].tolist()
            reranked_top = rerank_promote_single(top_indices, selected_paper_1based)
            remaining_indices = reranker_indices[qi][eff_k:].tolist()
            final_ranking = reranked_top + remaining_indices
            reranked_for_eval.append(final_ranking)

            has_gt = q_pubkey is not None
            if has_gt:
                gt_pubkey_idx = pubkeys.index(q_pubkey)
                if gt_pubkey_idx in top_indices:
                    sel_stats["gt_in_topk"] += 1
                    original_gt_pos = top_indices.index(gt_pubkey_idx)
                    if original_gt_pos == 0:
                        sel_stats["gt_already_rank1"] += 1
                    if selected_paper_1based is not None:
                        selected_pos_0 = selected_paper_1based - 1
                        if selected_pos_0 < len(top_indices) and top_indices[selected_pos_0] == gt_pubkey_idx:
                            sel_stats["gt_selected"] += 1
                            if original_gt_pos != 0:
                                sel_stats["gt_promoted"] += 1

                try:
                    gt_pos = final_ranking.index(gt_pubkey_idx)
                    gt_ranks.append(gt_pos + 1)
                except ValueError:
                    gt_ranks.append(-1)
            else:
                gt_ranks.append(-1)

            elapsed = time.time() - t0
            sel_display = f"Paper {selected_paper_1based}" if selected_paper_1based else "None"
            reason_short = reasoning[:60] + ("..." if len(reasoning) > 60 else "")
            print(
                f"    Query {qi + 1}/{len(queries)} | {elapsed:.1f}s | "
                f"Selected: {sel_display} | {reason_short}"
            )

        has_any_gt = any(q[2] is not None for q in queries)
        if has_any_gt:
            mrr = evaluate_mrr(queries, reranked_for_eval, pubkeys, list_k=mrr_k)
            hr = evaluate_hit_rate(queries, reranked_for_eval, pubkeys, list_k=hr_k)
        else:
            mrr = {k: 0.0 for k in mrr_k}
            hr = {k: 0.0 for k in hr_k}

        split_label = args.split.capitalize()
        all_results[f"{lang.upper()} {split_label}"] = {
            "mrr": mrr,
            "hr": hr,
            "n_queries": len(queries),
            "selection_stats": sel_stats,
            "gt_ranks": gt_ranks,
        }

        if has_any_gt:
            print(
                f"  MRR@5={mrr[5]:.4f}  MRR@10={mrr[10]:.4f}  HR@1={hr[1]:.4f}  HR@10={hr[10]:.4f}"
            )
            print(
                f"  Selections: {sel_stats['selected']}/{sel_stats['total']} | "
                f"GT in top-K: {sel_stats['gt_in_topk']} | "
                f"GT selected: {sel_stats['gt_selected']} | "
                f"GT promoted: {sel_stats['gt_promoted']} | "
                f"GT already rank1: {sel_stats['gt_already_rank1']}"
            )
        else:
            print("  No ground truth - skipping MRR/HR evaluation")
            print(f"  Selections: {sel_stats['selected']}/{sel_stats['total']} successful")

        all_queries_combined.extend(queries)
        all_reranked_combined.extend(reranked_for_eval)

    macro = macro_average_results(all_results, args.languages)
    if macro is not None:
        combined_gt = []
        combined_sel = {
            "total": 0,
            "selected": 0,
            "failed": 0,
            "gt_in_topk": 0,
            "gt_selected": 0,
            "gt_already_rank1": 0,
            "gt_promoted": 0,
        }
        for r in all_results.values():
            combined_gt.extend(r.get("gt_ranks", []))
            for k, v in r.get("selection_stats", {}).items():
                combined_sel[k] = combined_sel.get(k, 0) + v
        macro["gt_ranks"] = combined_gt
        macro["selection_stats"] = combined_sel
        all_results["Combined"] = macro
        print(f"\n  Combined (Macro-Average): MRR@5={macro['mrr'][5]:.4f}  HR@1={macro['hr'][1]:.4f}")

    per_lang_reranked = {}
    offset = 0
    for lang in args.languages:
        dev_file = os.path.join(args.dataset_dir, f"{args.query_prefix}{lang}_{args.split}.json")
        if not os.path.isfile(dev_file) or lang not in all_reranker_indices:
            continue
        queries = load_queries(dev_file)
        if args.max_queries is not None:
            queries = queries[:args.max_queries]
        n_full = len(queries)
        if args.num_shards > 1:
            shard_start = (n_full * args.shard) // args.num_shards
            shard_end = (n_full * (args.shard + 1)) // args.num_shards
            shard_size = shard_end - shard_start
            full_array = [None] * n_full
            for j in range(shard_size):
                full_array[shard_start + j] = all_reranked_combined[offset + j]
            per_lang_reranked[lang] = np.array(full_array, dtype=object)
            offset += shard_size
        else:
            per_lang_reranked[lang] = np.array(all_reranked_combined[offset:offset + n_full], dtype=object)
            offset += n_full

    shard_suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    reranked_cache = os.path.join(
        args.cache_dir,
        f"{tagged_slug}_rerank_{reranker_slug}_singlebest_v4_{llm_slug}_{backend_suffix}_{args.split}{shard_suffix}_indices.npz",
    )
    np.savez(reranked_cache, **per_lang_reranked)
    print(f"  Final reranked indices cached to {reranked_cache}")

    short_llm = args.llm_model.split("/")[-1]
    report_name = f"eval_singlebest_v4_{reranker_slug}_{short_llm}_top{args.llm_top_k}_{backend_suffix}{shard_suffix}.md"
    report_path = os.path.join(args.output_dir, report_name)
    generate_report(all_results, report_path, args.reranker, args.llm_model, args.llm_top_k)

    if client is not None:
        loop.run_until_complete(client.close())

    print("\nDone! Outputs:")
    print(f"  1. Report: {report_path}")


# ------------------------------------------------------------
# vLLM async engine (v4.6)
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

    cache_tag_suffix = f"_{args.cache_tag}" if args.cache_tag else ""
    combined_path = os.path.join(
        args.cache_dir,
        f"{model_slug}{cache_tag_suffix}_rerank_{reranker_slug}_indices.npz",
    )
    all_reranker_indices = {}
    if os.path.isfile(combined_path):
        print(f"Loading combined reranker cache: {combined_path}")
        reranker_data = np.load(combined_path, allow_pickle=True)
        all_reranker_indices = {lang: reranker_data[lang] for lang in reranker_data.files}
    else:
        print("Loading per-language reranker caches...")
        for lang in args.languages:
            per_lang_path = os.path.join(
                args.cache_dir,
                f"{model_slug}{cache_tag_suffix}_rerank_{reranker_slug}_{lang}.npz",
            )
            if not os.path.isfile(per_lang_path):
                per_lang_path = os.path.join(
                    args.cache_dir,
                    f"{model_slug}{cache_tag_suffix}_rerank_{reranker_slug}_{lang}_dev.npz",
                )

            if os.path.isfile(per_lang_path):
                data = np.load(per_lang_path)
                all_reranker_indices[lang] = data["indices"]
                print(f"  {lang.upper()}: loaded")
            else:
                print(f"  {lang.upper()}: not found")

    if not all_reranker_indices:
        raise FileNotFoundError(f"No reranker results found for '{args.reranker}' in {args.cache_dir}")

    if args.openai_base_url:
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
    selection_cache_path = os.path.join(
        args.cache_dir,
        f"single_best_v46_{reranker_slug}_{llm_slug}_top{args.llm_top_k}_vllm{run_id_suffix}.json",
    )
    cached_selections = None if args.force_rescore else load_cache(selection_cache_path)
    all_selections = dict(cached_selections) if cached_selections else {}

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

        t0 = time.time()
        tasks = []
        task_keys = []

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            cache_key = f"{lang}_{qi}"
            task_keys.append(cache_key)

            if cache_key in all_selections:
                entry = all_selections[cache_key]
                sel = entry.get("selected_paper") if isinstance(entry, dict) else entry
                reas = entry.get("reasoning", "") if isinstance(entry, dict) else ""
                tasks.append(asyncio.sleep(0, result=(sel, reas)))
            else:
                eff_k = min(args.llm_top_k, len(reranker_indices[qi]))
                top_indices = reranker_indices[qi][:eff_k].tolist()
                top_titles = [titles[idx] for idx in top_indices]
                top_venues = [venues[idx] for idx in top_indices]
                top_authors = [authors[idx] for idx in top_indices]
                top_abstracts = [abstracts[idx] for idx in top_indices]
                orig_text = original_queries.get(q_idx, q_text)
                tasks.append(
                    score_single_query_vllm(
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

        progress = {"done": 0, "total": len(tasks), "ok": 0, "fail": 0, "start_time": t0}

        async def track_result(coro, cache_key):
            if asyncio.iscoroutine(coro):
                result = await coro
            else:
                result = coro
            progress["done"] += 1
            sel, reason = result
            if sel is not None:
                progress["ok"] += 1
            else:
                progress["fail"] += 1
            elapsed = time.time() - progress["start_time"]
            rate = progress["done"] / elapsed if elapsed > 0 else 0
            eta = (progress["total"] - progress["done"]) / rate if rate > 0 else 0
            reason_short = reason[:60] + "..." if len(reason) > 60 else reason
            sel_display = f"Paper {sel}" if sel is not None else "None"
            print(
                f"    {progress['done']}/{progress['total']} | {elapsed:.1f}s | "
                f"Selected: {sel_display} | {reason_short}"
            )
            if progress["done"] % 100 == 0:
                print(
                    f"  === Progress: {progress['done']}/{progress['total']} "
                    f"(ok={progress['ok']}, fail={progress['fail']}) | "
                    f"{rate:.1f} queries/s | ETA {eta:.0f}s ==="
                )
            return result

        tracked_tasks = [track_result(task, key) for task, key in zip(tasks, task_keys)]
        results = await asyncio.gather(*tracked_tasks)

        for cache_key, (sel, reas) in zip(task_keys, results):
            if cache_key not in all_selections:
                all_selections[cache_key] = {"selected_paper": sel, "reasoning": reas}

        elapsed = time.time() - t0
        print(f"  LLM calls completed in {elapsed:.1f}s")

        reranked_for_eval = []
        gt_ranks = []
        sel_stats = {
            "total": len(queries),
            "selected": 0,
            "failed": 0,
            "gt_in_topk": 0,
            "gt_selected": 0,
            "gt_already_rank1": 0,
            "gt_promoted": 0,
        }

        for qi, (q_idx, q_text, q_pubkey) in enumerate(queries):
            cache_key = f"{lang}_{qi}"
            selected_paper = all_selections[cache_key]["selected_paper"]
            eff_k = min(args.llm_top_k, len(reranker_indices[qi]))
            top_indices = reranker_indices[qi][:eff_k].tolist()

            if selected_paper is not None:
                sel_stats["selected"] += 1
            else:
                sel_stats["failed"] += 1

            reranked = rerank_promote_single(top_indices, selected_paper)
            remaining = reranker_indices[qi][eff_k:].tolist()
            final_ranking = reranked + remaining
            reranked_for_eval.append(final_ranking)

            has_gt = q_pubkey is not None
            if has_gt:
                gt_pubkey_idx = pubkeys.index(q_pubkey)
                if gt_pubkey_idx in top_indices:
                    sel_stats["gt_in_topk"] += 1
                    orig_pos = top_indices.index(gt_pubkey_idx)
                    if orig_pos == 0:
                        sel_stats["gt_already_rank1"] += 1
                    if selected_paper is not None:
                        sel_pos = selected_paper - 1
                        if 0 <= sel_pos < len(top_indices) and top_indices[sel_pos] == gt_pubkey_idx:
                            sel_stats["gt_selected"] += 1
                            if orig_pos != 0:
                                sel_stats["gt_promoted"] += 1

                try:
                    gt_ranks.append(final_ranking.index(gt_pubkey_idx) + 1)
                except ValueError:
                    gt_ranks.append(-1)
            else:
                gt_ranks.append(-1)

        has_any_gt = any(q[2] is not None for q in queries)
        if has_any_gt:
            mrr = evaluate_mrr(queries, reranked_for_eval, pubkeys, list_k=[1, 3, 5, 10, 50, 100])
            hr = evaluate_hit_rate(queries, reranked_for_eval, pubkeys, list_k=[1, 5, 10, 50, 100])
        else:
            mrr = {k: 0.0 for k in [1, 3, 5, 10, 50, 100]}
            hr = {k: 0.0 for k in [1, 5, 10, 50, 100]}

        all_results[f"{lang.upper()} Dev"] = {
            "mrr": mrr,
            "hr": hr,
            "n_queries": len(queries),
            "selection_stats": sel_stats,
            "gt_ranks": gt_ranks,
        }
        if has_any_gt:
            print(f"  MRR@5={mrr[5]:.4f}  HR@1={hr[1]:.4f}")
        else:
            print("  No ground truth - skipping MRR/HR evaluation")
        all_reranked_combined.extend(reranked_for_eval)

    macro = macro_average_results(all_results, args.languages)
    if macro:
        all_results["Combined"] = macro

    save_cache(selection_cache_path, all_selections)
    report_path = os.path.join(
        args.output_dir,
        f"eval_singlebest_v46_{reranker_slug}_{llm_slug}_top{args.llm_top_k}{run_id_suffix}.md",
    )
    generate_report(
        all_results,
        report_path,
        args.reranker,
        args.llm_model,
        args.llm_top_k,
        max_concurrent=args.max_concurrent,
        title="LLM Single-Best Reranker v4.6 Evaluation Report",
    )

    await client.close()
    print("\nDone!")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM single-best-paper reranker (merged v4 batch + v4.6 vLLM async)"
    )
    parser.add_argument("--engine", choices=["batch", "vllm_async"], default="batch")

    parser.add_argument("--dataset_dir", type=str, default="CT26/Dataset_translated")
    parser.add_argument("--cache_dir", type=str, default="GRITLM_finetune/eval_cache_gritlm_translated")
    parser.add_argument("--output_dir", type=str, default="GRITLM_finetune/eval_results_translated")
    parser.add_argument("--model_path", type=str, default="GRITLM")
    parser.add_argument("--reranker", type=str, default="Qwen/Qwen3-Reranker-8B")

    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_top_k", type=int, default=10)
    parser.add_argument("--max_concurrent", type=int, default=50)
    parser.add_argument("--max_context_tokens", type=int, default=60000)
    parser.add_argument("--languages", nargs="+", default=["en", "de", "fr"])
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--force_rescore", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--cache_tag", type=str, default="")
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--api_backend", type=str, choices=["vllm", "scads", "openai"], default="vllm")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--scads_api_key", type=str, default=None)
    parser.add_argument("--scads_disable_fallbacks", action="store_true")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default="https://api.openai.com/v1")

    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--query_prefix", type=str, default="")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    args = parser.parse_args()

    if args.llm_model is None:
        if args.engine == "vllm_async":
            args.llm_model = "Qwen/Qwen3.5-9B"
        else:
            args.llm_model = "moonshotai/Kimi-K2.6"

    if args.engine == "batch":
        run_batch_mode(args)
    else:
        asyncio.run(run_vllm_async_mode(args))


if __name__ == "__main__":
    main()
