"""
================================================================================
ATTRIBUTION-BASED NLI RERANKING WORKFLOW (SELF-CONTAINED ENGINE)
================================================================================

This module provides a unified command-line pipeline for Scientific Claim Extraction, 
Natural Language Inference (NLI) alignment, and Reranking Evaluation (MRR@5).

Model Specifications (Paper-Aligned):
    1. Query Extractor: Supports three models:
       - Qwen3.5 27B (vLLM Backend)
       - Gemma (vLLM Backend)
       - Kimi (API Backend)
    2. LLM-as-NLI Reranker: Restrained to Qwen3.5 27B.

Prerequisites:
    1. test data
       - ./data : should contain "{lang}_dev.json" files for query extraction, "grit_qwen_top10_{lang}" files for NLI reranking)
    2. Configure environment variables for API authentication (if using api backend):
       export LLM_API_KEY="your_api_key"
       export OPENAI_API_BASE="https://api.yourprovider.com/v1"

Usage:
    Step 1: Extract Atomic Facts from tweets across all versions (V1, V2, V3) and languages
        # Option A: Run with Qwen3.5 27B (vLLM)
        python llm_request.py --task query --version -1 --backend vllm --vllm-model Qwen/Qwen3.5-27B

        # Option B: Run with Gemma (vLLM)
        python llm_request.py --task query --version -1 --backend vllm --vllm-model google/gemma-4-31B-it

        # Option C: Run with Kimi (API)
        python llm_request.py --task query --version -1 --backend api

    Step 2: Calculate NLI support scores and perform scoring fusion reranking (Automatically defaults to Qwen3.5 27B)
        python llm_request.py --task nli --version -1 --backend vllm --extractor qwen-qwen

    Step 3: (Optional) Standalone calculation of MRR@5 metrics across all iterations:
        python llm_request.py --task mrr --extractor qwen-qwen
================================================================================
"""

import os
import json
import argparse
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
# Try importing vLLM engine safely; fallback cleanly if absent
try:
    from vllm import LLM as VLLMEngine, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Unified Model Configurations Aligned with Paper Specifications
MODEL_KIMI = "moonshotai/Kimi-K2.6"
MODEL_GEMMA = "google/gemma-4-31B-it"
MODEL_QWEN = "Qwen/Qwen3.5-27B"

BASE_URL = "Specify your model URL"
MODEL_NAME = MODEL_KIMI           # Default API Endpoint target
VLLM_MODEL_NAME = MODEL_QWEN     # Default vLLM target for query extraction
VLLM_TENSOR_PARALLEL = 1
VLLM_BATCH_SIZE = 64              # Number of prompts sent to vLLM per generate() call
API_TIMEOUT = 1000

_vllm_engine = None
_vllm_tokenizer = None
_global_client = None

# MRR Evaluation constants
# Configure base paths dynamically relative to the script location
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "..", "sample_data")
LANGUAGES = ['en', 'de', 'fr']
VERSIONS = [1, 2, 3]
STRATEGIES = ["baseline", "nli", "plus", "multi", "weight", "alpha"]
MRR_K = 5
ALPHA = 0.7
# Integrated Prompt for Claim and Atomic Fact Decomposition
PROMPT_FOR_QUERY_EXTRACT = """You are a Scientific Data Analyst. Extract 'NLI-Ready Claims' from the tweet to match academic papers (Title, Abstract, Venue, Authors). 
- ATOMIC DECOMPOSITION: Split the text into standalone sentences, each containing exactly ONE fact. Replace all pronouns (e.g., 'it', 'they', 'this') with explicit entity names from the context.
- ENTITY & METADATA RETENTION: Preserve all technical terms, numerical data (percentages, dosages), and research methodologies.
- STRATEGIC METADATA: Convert @handles or #hashtags into associations (e.g., 'This research is linked to [Author/Journal]') ONLY if they clearly identify a specific author or venue. Ignore generic or noisy tags.
- ZERO REDUNDANCY: Each claim must provide unique information. Do not generate active/passive variations or synonymous rephrasings (e.g., if 'A impacts B' is extracted, DO NOT add 'B is affected by A').
Output ONLY the JSON object, No markdown: {"facts": ["...", "...",...]}"""
# System Prompt utilized for NLI alignment determination
SYSTEM_PROMPT_NLI = (
    "You are a strict scientific relevance judge. "
    "Given one paper candidate and a list of query facts, judge each (query_fact, paper_candidate) pair independently. "
    "Return 1 only when the paper candidate clearly supports or directly matches the fact. "
    "Return 0 when the fact is unrelated, only weakly related, uncertain, or not supported by the paper candidate. "
    "Do not use external knowledge. "
    "Output strictly in JSON with this schema: {\"labels\": [0 or 1, ...]} "
    "The number of labels must equal the number of facts and keep the same order."
)
def load_queries(lang: str = 'en',split: str = "dev", k: int = -1) -> pd.DataFrame:
    """
    Loads query data (e.g., en_dev.json). Searches directly in data/
    :param lang: Language code (en, de, fr)
    :param k: Load quantity (-1 for all)
    :param t: Number of test samples to exclude for experimental splits
    :return: Query Dataframe
    """
    file_name = f"{lang}_{split}.json"
    
    # Check directly inside data/ first (as shown in the simplified structure)
    query_path = os.path.join(DATASET_PATH, file_name)
    # Read JSON data
    df_query: DataFrame = pd.read_json(query_path)
    # If k is specified, randomly select k samples
    if k > 0:
        k = min(k, len(df_query))
        df_query = df_query.sample(n=k, random_state=42)

    return df_query



def _get_api_client():
    global _global_client
    if _global_client is None:
        my_api_key = _load_api_key()
        _global_client = OpenAI(base_url=BASE_URL, api_key=my_api_key)
    return _global_client


def _get_vllm_engine():
    global _vllm_engine, _vllm_tokenizer
    if _vllm_engine is None:
        _vllm_tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_NAME)
        # Force Triton backend to circumvent specific CUDA SM90 compilation issues
        _vllm_engine = VLLMEngine(
            model=VLLM_MODEL_NAME,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL,
            gdn_prefill_backend="triton", 
            attention_backend="TRITON_ATTN",
            max_num_seqs=512,
            max_model_len=16384,
            trust_remote_code=True
        )
    return _vllm_engine


def _format_prompt(system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        return _vllm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return _vllm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )


def _load_api_key():
    path = os.getcwd()
    api_path = os.path.join(path, "api_keys")
    try:
        with open(os.path.join(api_path, "scads_llm.txt")) as keyfile:
            return keyfile.readline().strip()
    except FileNotFoundError:
        print("Error: The file 'scads_llm.txt' was not found. Please make sure the file exists and contains your API key.")
        exit(1)


def _generate_batch(system_prompt: str, items: list[str], item_label: str = None, temperature: float = 0.1,
                    backend: str = "api") -> list[str]:
    """
    Executes batched LLM text generation utilizing API threading or local vLLM.
    """
    if backend == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install it with: pip install vllm")
        engine = _get_vllm_engine()
        sampling_params = SamplingParams(temperature=temperature, max_tokens=1024)
        
        if item_label:
            prompts = [
                _format_prompt(system_prompt, f"{item_label}: {item}\nOutput:")
                for item in items
            ]
        else:
            prompts = [
                _format_prompt(system_prompt, item)
                for item in items
            ]
            
        all_results = []
        for i in range(0, len(prompts), VLLM_BATCH_SIZE):
            batch_prompts = prompts[i:i + VLLM_BATCH_SIZE]
            outputs = engine.generate(batch_prompts, sampling_params)
            all_results.extend([o.outputs[0].text.strip() for o in outputs])
        return all_results
    else:
        client = _get_api_client()
        def process_item(item):
            user_content = f"{item_label}: {item}\nOutput:" if item_label else item
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=MODEL_NAME,
                temperature=temperature,
                extra_body={'chat_template_kwargs': {"thinking": False}}, 
                timeout=API_TIMEOUT
            )
            content = response.choices[0].message.content
            if content is None:
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                print(f"\n⚠️ Warning: LLM returned None. Finish reason: {finish_reason}. Input: {item[:30]}...")
            return content.strip() if content else ""

        with ThreadPoolExecutor(max_workers=min(len(items), 32)) as executor:
            return list(executor.map(process_item, items))


def clean_and_parse_json_nli(raw_text: str, n_facts: int) -> list[int]:
    """
    A robust JSON parser built to repair markdown codeblocks or truncated JSON structures.
    """
    clean_res = raw_text.strip()
    if clean_res.startswith("```json"):
        clean_res = clean_res[7:]
    elif clean_res.startswith("```"):
        clean_res = clean_res[3:]
    if clean_res.endswith("```"):
        clean_res = clean_res[:-3]
    clean_res = clean_res.strip()
    
    if clean_res.startswith("{") and not clean_res.endswith("}"):
        if clean_res.endswith('"]') or clean_res.endswith('"]\n'):
            clean_res = clean_res + "}"
        elif clean_res.endswith('"') or clean_res.endswith('"\n'):
            clean_res = clean_res + "]}"
        elif clean_res.endswith(']') or clean_res.endswith(']\n'):
            clean_res = clean_res + "}"
        else:
            clean_res = clean_res + "]}"
            
    try:
        data = json.loads(clean_res)
        labels = data.get("labels", [])
        
        normalized = []
        for v in labels[:n_facts]:
            if isinstance(v, bool):
                normalized.append(1 if v else 0)
            elif isinstance(v, (int, float)):
                normalized.append(1 if int(v) == 1 else 0)
            elif isinstance(v, str):
                s = v.strip().lower()
                if s in {"1", "support", "supported", "entail", "entails", "yes", "true"}:
                    normalized.append(1)
                else:
                    normalized.append(0)
            else:
                normalized.append(0)
                
        if len(normalized) < n_facts:
            normalized.extend([0] * (n_facts - len(normalized)))
            
        return normalized
    except Exception:
        return [0] * n_facts


def style_transfer_query(tweets: list[str], few_shot: int = 5, temperature: float = 0.1,
                         backend: str = "api") -> list[str]:
    """
    Rewrite tweets to scientific claims in parallel.
    """
    system_prompt = PROMPT_FOR_QUERY_EXTRACT
    return _generate_batch(system_prompt, tweets, "Tweet", temperature, backend)


def run_style_transfer_query(top_k: int = -1, batch_size: int = 5, backend: str = "api",
                             lang: str = "en", split: str = "dev", version: int = 1):
    """
    Loads query tweets, performs style transfer via LLM, and outputs JSON files.
    """
    df_queries = load_queries(lang=lang,split=split, k=top_k)
    all_queries = df_queries

    total_tweets = len(all_queries)
    query_facts = [""] * total_tweets

    out_dir = os.path.join(BASE_PATH, "out", "extract-query")
    os.makedirs(out_dir, exist_ok=True)
    
    start_index = 0
    system_prompt = PROMPT_FOR_QUERY_EXTRACT

    for i in range(start_index, total_tweets, batch_size):
        batch = all_queries['text'][i: i + batch_size].tolist()
        batch_results = _generate_batch(system_prompt, batch, "Tweet", 0.1, backend)
        
        parsed_results = []
        for res in batch_results:
            try:
                data = json.loads(res)
                facts_list = data.get("facts", [])
                parsed_results.append(facts_list)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Failed to parse JSON from LLM response: {res}")
                parsed_results.append([])

        if len(parsed_results) == len(batch):
            query_facts[i: i + batch_size] = parsed_results
        else:
            print(f"Warning: Expected {len(batch)} results, but got {len(parsed_results)}. Appending missing as empty.")
            query_facts[i: i + len(parsed_results)] = parsed_results
        
        if (i + batch_size) % 10 == 0 or i + batch_size >= total_tweets:
            print(f"Processed {min(i + batch_size, total_tweets)} out of {total_tweets} tweets.")

    if 'index' in all_queries.columns:
        indices = all_queries['index'].tolist()
    else:
        indices = all_queries.index.tolist()

    if 'pubkey' in all_queries.columns:
        pubkeys = all_queries['pubkey'].tolist()
    else:
        pubkeys = [None] * len(indices)

    records = []
    texts = all_queries['text'].tolist()
    for idx, ori, fact_item, pk in zip(indices, texts, query_facts, pubkeys):
        try:
            rec_idx = int(idx)
        except Exception:
            rec_idx = idx
            
        records.append({
            "index": rec_idx,
            "ori": ori,
            "facts": fact_item,
            "pubkey": pk
        })

    # Saved with generic suffix _V{version} for seamless modularity
    json_name = f"extract-query_{lang}_V{version}.json"
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(records, jf, ensure_ascii=False, indent=2)
    print(f"Saved Query Facts: {json_path}")


def run_nli_reranking(lang: str, split: str, version: int, backend: str = "vllm", batch_size: int = 64, extractor_name: str = "nli-qq"):
    """
    Fuses extracted atomic query facts with first-stage document rankings using NLI Support Rate.
    """
    
    reranker_file = os.path.join(DATASET_PATH, f"grit_qwen_top10_{lang}.json")
    facts_file =os.path.join(BASE_PATH,"out","extract-query", f"extract-query_{lang}_V{version}.json")
    output_file = os.path.join(BASE_PATH,"out","nli-score",extractor_name, f"grit_qwen_nli_top10_{lang}_V{version}.json")

    if not os.path.exists(reranker_file):
        print(f"Warning: First stage document candidates not found: {reranker_file}")
        return
    if not os.path.exists(facts_file):
        print(f"Warning: Extracted query facts file not found under any expected path. Skipping.")
        return

    print(f"📖 Processing NLI scoring for:\n   Reranker Candidates: {reranker_file}\n   Claims: {facts_file}")
    with open(reranker_file, "r", encoding="utf-8") as f:
        rerank_data = json.load(f)
    with open(facts_file, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    facts_by_index = {}
    facts_by_text = {}
    for item in facts_data:
        idx = item.get("index")
        ori_text = item.get("ori", "").strip()
        facts_list = item.get("facts", [])
        
        if not facts_list and ori_text:
            facts_list = [ori_text]
            
        facts_by_index[idx] = facts_list
        if ori_text:
            facts_by_text[ori_text] = facts_list

    inference_tasks = []
    print("🔍 Formulating verification NLI prompts...")
    for q_idx, query_item in enumerate(rerank_data):
        query_index = query_item.get("query_index")
        query_text = query_item.get("query_text", "").strip()

        facts = facts_by_index.get(query_index)
        if facts is None:
            facts = facts_by_text.get(query_text)
        if facts is None:
            facts = [query_text] if query_text else []
        if not facts:
            continue

        candidates = query_item.get("candidates", [])
        if candidates and "reranker_rank" in candidates[0]:
            sorted_cands = sorted(candidates, key=lambda x: x.get("reranker_rank", 999))
        elif candidates and "reranker_score" in candidates[0]:
            sorted_cands = sorted(candidates, key=lambda x: x.get("reranker_score", -999), reverse=True)
        else:
            sorted_cands = candidates

        top_10_cands = sorted_cands[:10]
        query_item["candidates"] = top_10_cands

        for c_idx, cand in enumerate(top_10_cands):
            paper_text = cand.get("document_text", "")
            facts_formatted = "\n".join([f"{i}. {f}" for i, f in enumerate(facts, start=1)])
            
            user_prompt = (
                "Paper candidate:\n"
                f"{paper_text}\n\n"
                "Query facts:\n"
                f"{facts_formatted}\n\n"
                "Return JSON only."
            )
            
            inference_tasks.append({
                "query_list_idx": q_idx,
                "cand_list_idx": c_idx,
                "facts_list": facts,
                "prompt": user_prompt
            })

    total_tasks = len(inference_tasks)
    if total_tasks == 0:
        print(f"Warning: No active candidates found. Skipping NLI processing.")
        return

    print(f"Deploying NLI Batch Alignment (Total Pairs: {total_tasks})...")
    prompts_to_run = [task["prompt"] for task in inference_tasks]
    
    raw_outputs = _generate_batch(SYSTEM_PROMPT_NLI, prompts_to_run, item_label=None, temperature=0.0, backend=backend)

    print("💾 Composing outputs and scaling alignment parameters...")
    for task, output_text in zip(inference_tasks, raw_outputs):
        q_list_idx = task["query_list_idx"]
        c_list_idx = task["cand_list_idx"]
        facts_list = task["facts_list"]
        
        labels = clean_and_parse_json_nli(output_text, len(facts_list))
        support_rate = float(sum(labels)) / float(len(facts_list)) if facts_list else 0.0
        
        rerank_data[q_list_idx]["candidates"][c_list_idx]["nli_support_rate"] = support_rate

    print(f"💾 Saving unified NLI reranked document: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(rerank_data, f_out, ensure_ascii=False, indent=2)
    print(f"Finished NLI processing for language split: {lang.upper()} | Version: V{version}!\n")


# ==================== MRR Evaluation & Report Modules ====================

def calculate_mrr_at_k(preds, gts, k=MRR_K):
    scores = []
    for p_list, label in zip(preds, gts):
        target_label = str(label)
        sliced_preds = [str(p) for p in p_list[:k]]
        
        score = 0.0
        if target_label in sliced_preds:
            idx = sliced_preds.index(target_label)
            score = 1.0 / (idx + 1)
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def evaluate_query_strategies(query_item):
    candidates = query_item.get("candidates", [])
    if not candidates:
        return None

    cand_keys = [cand["pubkey"] for cand in candidates]
    s_qwen = [cand.get("reranker_score", 0.0) for cand in candidates]
    s_nli_supp_rate = [cand.get("nli_support_rate", 0.0) for cand in candidates]

    scores = {
        "baseline": s_qwen,
        "nli": s_nli_supp_rate,
        "plus": [q + n for q, n in zip(s_qwen, s_nli_supp_rate)],
        "multi": [q * n for q, n in zip(s_qwen, s_nli_supp_rate)],
        "weight": [q * (1.0 + n) for q, n in zip(s_qwen, s_nli_supp_rate)],
        "alpha": [(ALPHA * q + (1.0 - ALPHA) * n) for q, n in zip(s_qwen, s_nli_supp_rate)]
    }

    ranked = {}
    for strategy, score_array in scores.items():
        ranked_pairs = sorted(zip(cand_keys, score_array), key=lambda x: x[1], reverse=True)
        ranked[strategy] = [cand_key for cand_key, _ in ranked_pairs]
    return ranked


def evaluate_file_mrr(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    preds_by_strat = {s: [] for s in STRATEGIES}
    gts_by_strat = {s: [] for s in STRATEGIES}
    
    for item in data:
        # Compatibility check for either schema field key
        gt = item.get("ground_truth_pubkey") or item.get("true_pubkey")
        ranked = evaluate_query_strategies(item)
        if ranked is None or gt is None:
            continue
        for strategy in STRATEGIES:
            preds_by_strat[strategy].append(ranked[strategy])
            gts_by_strat[strategy].append(gt)
            
    results = {}
    for strategy in STRATEGIES:
        results[strategy] = calculate_mrr_at_k(preds_by_strat[strategy], gts_by_strat[strategy])
    return results


def run_mrr_evaluation_report(extractor_name: str = "nli-qq",version=[1]):
    """
    Aggregates MRR@5 calculations across all strategies, language splits, and versions.
    Outputs a standard academic formatted comparative markdown summary.
    """
    print(f"⏳ Compiling macro MRR@5 matrix reports across experiment partitions for: {extractor_name}")
    
    raw_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for lan in LANGUAGES:
        for v in version:
            input_file = os.path.join(
                BASE_PATH, "out","nli-score", extractor_name, f"grit_qwen_nli_top10_{lan}_V{v}.json"
            )
            if not os.path.exists(input_file):
                print(f"Warning: File missing: {input_file}. Skipping calculation.")
                continue
            
            file_scores = evaluate_file_mrr(input_file)
            for strategy, score in file_scores.items():
                raw_results[extractor_name][strategy][lan][v] = score

    report_lines = [
        f"# Pipeline Experiments Report ({extractor_name} Reranking)",
        f"Evaluation Metric: **MRR@{MRR_K}** (Mean $\\pm$ Standard Deviation across 3 runs)\n"
    ]

    headers = ["Strategy", f"{extractor_name}-en", f"{extractor_name}-de", f"{extractor_name}-fr", f"{extractor_name}-avg"]
    table_header = "| " + " | ".join(headers) + " |"
    table_divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_rows = []

    for strategy in STRATEGIES:
        row_cells = [f"**{strategy}**"]
        
        lang_means_dict = {}
        for lan in LANGUAGES:
            scores_v = []
            for v in VERSIONS:
                val = raw_results[extractor_name][strategy][lan].get(v)
                if val is not None:
                    scores_v.append(val)
            
            if len(scores_v) >= 1:
                mean_val = np.mean(scores_v)
                std_val = np.std(scores_v, ddof=1) if len(scores_v) > 1 else 0.0
                lang_means_dict[lan] = mean_val
                row_cells.append(f"{mean_val:.3f} ± {std_val:.3f}")
            else:
                row_cells.append("N/A")
                lang_means_dict[lan] = 0.0
                
        avg_scores_by_version = []
        for v in VERSIONS:
            v_scores = []
            for lan in LANGUAGES:
                if v in raw_results[extractor_name][strategy][lan]:
                    v_scores.append(raw_results[extractor_name][strategy][lan][v])
            if v_scores:
                avg_scores_by_version.append(np.mean(v_scores))
        
        if avg_scores_by_version:
            macro_mean = np.mean(avg_scores_by_version)
            macro_std = np.std(avg_scores_by_version, ddof=1) if len(avg_scores_by_version) > 1 else 0.0
            row_cells.append(f"**{macro_mean:.3f} ± {macro_std:.3f}**")
        else:
            row_cells.append("N/A")

        table_rows.append("| " + " | ".join(row_cells) + " |")

    markdown_table = "\n".join([table_header, table_divider] + table_rows)
    report_lines.append(markdown_table)
    final_report_content = "\n".join(report_lines)
    
    print("\n" + "="*40 + " FINAL REPORT " + "="*40)
    print(final_report_content)
    print("="*94)

    output_report_path = os.path.join(BASE_PATH, "out", f"{extractor_name}-final_report.md")
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(final_report_content)
    print(f"\nMRR report successfully compiled and saved to:\n{output_report_path}")


# ==================== Pipeline Main Entrypoint ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Claims Extraction and NLI Reranking Suite")
    parser.add_argument("--task", choices=["query", "nli", "mrr"], default="query",
                        help="Execution mode: 'query' (fact extraction), 'nli' (NLI validation reranking), 'mrr' (generate report)")
    parser.add_argument("--version", type=int, choices=[1, 2, 3, -1], default=1,
                        help="Target iteration version (1, 2, 3). Pass -1 to run all versions sequentially.")
    parser.add_argument("--backend", choices=["api", "vllm"], default="api",
                        help="Inference backend (default: api)")
    parser.add_argument("--split", choices=["dev", "train"], default="dev",
                        help="Dataset partition split (default: dev)")
    parser.add_argument("--vllm-model", default=None,
                        help="vLLM model override")
    parser.add_argument("--vllm-batch-size", type=int, default=None,
                        help="vLLM parallel batch size boundaries (defaults to 64)")
    parser.add_argument("--vllm-tp", type=int, default=None,
                        help="vLLM tensor parallel GPU boundaries (defaults to 1)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Number of query items to process (-1 = all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Outer iterator step size (default: 32)")
    parser.add_argument("--extractor", type=str, default="qwen-qwen",
                        help="Name of the query extractor or combo (e.g., qwen-qwen, gemma-qwen, kimi-qwen)")
    args = parser.parse_args()

    # Define default model settings based on paper specifications
    if args.task == "nli":
        # llm-as-nli is restricted to Qwen3.5 27B by default per paper setup
        VLLM_MODEL_NAME = args.vllm_model if args.vllm_model else MODEL_QWEN
    else:
        # Query extraction defaults to Gemma unless specified otherwise
        VLLM_MODEL_NAME = args.vllm_model if args.vllm_model else MODEL_GEMMA

    # Apply overrides from parameters if specified on command-line
    if args.vllm_batch_size is not None:
        VLLM_BATCH_SIZE = args.vllm_batch_size
    if args.vllm_tp is not None:
        VLLM_TENSOR_PARALLEL = args.vllm_tp

    # Run loop configurations
    target_languages = ['en', 'de', 'fr']
    target_versions = [1, 2, 3] if args.version == -1 else [args.version]

    if args.task == "query":
        for v in target_versions:
            print(f"\n🚀 === Starting Atomic Claim Extraction for Version V{v} ===")
            for lang in target_languages:
                print(f"🔄 Language Partition: {lang.upper()}")
                run_style_transfer_query(
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    backend=args.backend,
                    lang=lang,
                    split=args.split,
                    version=v
                )
                
    elif args.task == "nli":
        for v in target_versions:
            print(f"\n=== Starting NLI Reranking Calculations for Version V{v} ===")
            for lang in target_languages:
                print(f"Language Partition: {lang.upper()}")
                run_nli_reranking(
                    lang=lang,
                    split=args.split,
                    version=v,
                    backend=args.backend,
                    batch_size=args.batch_size,
                    extractor_name=args.extractor
                )
        
        # Automatically trigger MRR evaluation report compiles upon NLI completion
        print("\nAll NLI Reranking tasks completed. Generating evaluation statistics automatically...")
        run_mrr_evaluation_report(extractor_name=args.extractor)
        
    elif args.task == "mrr":
        # Standalone report generation
        run_mrr_evaluation_report(extractor_name=args.extractor,version=target_versions)