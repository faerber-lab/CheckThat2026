"""
LLM-based style transfer
python llm_requests.py --task query --backend vllm --vllm-batch-size 64 --vllm-model Qwen/Qwen3-9B
"""

from process_data import load_corpus, load_queries
from config import get_query_prompt
import os
import argparse
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import re

try:
    from vllm import LLM as VLLMEngine, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# ==============================================================================
# Dynamic Defaults & Environments
# ==============================================================================
BASE_URL = os.getenv("SCADS_API_BASE", "https://llm.scads.ai/v1")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
VLLM_MODEL_NAME = "Qwen/Qwen3.5-27B"
VLLM_TENSOR_PARALLEL = 1
VLLM_BATCH_SIZE = 64  

_vllm_engine = None
_vllm_tokenizer = None


def _get_vllm_engine():
    global _vllm_engine, _vllm_tokenizer
    if _vllm_engine is None:
        _vllm_tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_NAME)
        # Standard configuration as suggested in vLLM technical guidelines
        _vllm_engine = VLLMEngine(
            model=VLLM_MODEL_NAME,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL,
            # Fallback to Triton backend to bypass potential CUDA/SM compilation conflicts
            gdn_prefill_backend="triton", 
            attention_backend="TRITON_ATTN",
            max_num_seqs=512,
            trust_remote_code=True
        )
    return _vllm_engine


def _format_prompt(system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return _vllm_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )


def _load_api_key():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_path = os.path.join(current_dir, "api_keys")
    try:
        with open(os.path.join(api_path, "scads_llm.txt")) as keyfile:
            return keyfile.readline().strip()
    except FileNotFoundError:
        print("Error: The file 'scads_llm.txt' was not found in api_keys directory. Please create it first.")
        exit(1)

def _generate_batch(system_prompt: str, items: list[str], item_label: str, temperature: float,
                    backend: str) -> list[str]:
    if backend == "vllm":
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install it with: pip install vllm")
        engine = _get_vllm_engine()
        sampling_params = SamplingParams(temperature=temperature, max_tokens=512)
        prompts = [
            _format_prompt(system_prompt, f"{item_label}: {item}\nOutput:")
            for item in items
        ]
        all_results = []
        for i in range(0, len(prompts), VLLM_BATCH_SIZE):
            batch_prompts = prompts[i:i + VLLM_BATCH_SIZE]
            outputs = engine.generate(batch_prompts, sampling_params)
            all_results.extend([o.outputs[0].text.strip() for o in outputs])
        return all_results
    else:
        my_api_key = _load_api_key()
        client = OpenAI(base_url=BASE_URL, api_key=my_api_key)

        def process_item(item):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{item_label}: {item}\nOutput:"}
                ],
                model=MODEL_NAME,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        with ThreadPoolExecutor() as executor:
            return list(executor.map(process_item, items))


def style_transfer_query(tweets: list[str], few_shot: int = 5, temperature: float = 0.1,
                         backend: str = "api") -> list[str]:
    """
    Uses API or vLLM to generate rewritten texts for multiple tweets in parallel.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(path, "prompts")
    with open(os.path.join(prompt_path, f"prompt_{get_query_prompt()}.txt"), "r", encoding="utf-8") as file:
        system_prompt = file.read()

    return _generate_batch(system_prompt, tweets, "Tweet", temperature, backend)


def run_style_transfer_query(top_k: int = -1, batch_size: int = 5, backend: str = "api",
                             lang: str = "en", split: str = "dev", version: int = 1, prompt_id: int = 1):
    """
    Run style transfer for queries using the prompt specified in config.
    """
    df_queries = load_queries(lang=lang, split=split, k=top_k)
    corpus, cord_uids = load_corpus()
    corpus = {"cord_uid": cord_uids, "abstract": corpus}
    corpus = pd.DataFrame(data=corpus)
    test_abstracts = df_queries

    total_tweets = len(test_abstracts)
    synthetic_abstracts = [""] * total_tweets

    # Read system prompt and send requests to LLM in batches
    path = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(path, "prompts")
    prompt_file = os.path.join(prompt_path, f"prompt_{prompt_id}.txt")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    for i in range(0, total_tweets, batch_size):
        batch = test_abstracts['text'][i: i + batch_size].tolist()
        batch_results = _generate_batch(system_prompt, batch, "Tweet", 0.1, backend)

        if len(batch_results) == len(batch):
            synthetic_abstracts[i: i + batch_size] = batch_results
        else:
            print(f"Warning: Expected {len(batch)} results, but got {len(batch_results)}. Padding with empty strings.")
            synthetic_abstracts[i: i + len(batch_results)] = batch_results

        if (i + batch_size) % 10 == 0 or i + batch_size >= total_tweets:
            print(f"Processed {min(i + batch_size, total_tweets)} out of {total_tweets} tweets.")

    test_abstracts['synthetic_abstract'] = synthetic_abstracts

    out_dir = os.path.join(path, "out", "synthetic_queries")
    os.makedirs(out_dir, exist_ok=True)

    # Output JSON records format: {index, ori, query_transfer, pubkey}
    if 'index' in test_abstracts.columns:
        indices = test_abstracts['index'].tolist()
    else:
        indices = test_abstracts.index.tolist()

    if 'pubkey' in test_abstracts.columns:
        pubkeys = test_abstracts['pubkey'].tolist()
    else:
        pubkeys = [None] * len(indices)

    records = []
    texts = test_abstracts['text'].tolist()
    for idx, ori, trans, pk in zip(indices, texts, synthetic_abstracts, pubkeys):
        try:
            rec_idx = int(idx)
        except Exception:
            rec_idx = idx
        records.append({
            "index": rec_idx,
            "ori": ori,
            "query_transfer": trans,
            "pubkey": pk
        })

    json_name = f"synthetic_queries_{lang}_C{prompt_id}_V{version}.json"
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, 'w', encoding='utf-8') as jf:
        import json as _json
        _json.dump(records, jf, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM style transfer / claim extraction / reranking")
    parser.add_argument("--task", choices=["query", "corpus", "claims", "rerank"], required=True,
                        help="Which task to run: query, corpus, claims, or rerank")
    parser.add_argument("--backend", choices=["api", "vllm"], default="api",
                        help="Inference backend (default: api)")
    parser.add_argument("--vllm-model", default=None,
                        help="vLLM model name (overrides VLLM_MODEL_NAME)")
    parser.add_argument("--vllm-batch-size", type=int, default=None,
                        help="vLLM batch size (overrides VLLM_BATCH_SIZE)")
    parser.add_argument("--vllm-tp", type=int, default=None,
                        help="vLLM tensor parallel size (overrides VLLM_TENSOR_PARALLEL)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Number of items to process (-1 = all)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Outer batch size for iterating over data")
    args = parser.parse_args()

    if args.vllm_model:
        VLLM_MODEL_NAME = args.vllm_model
    if args.vllm_batch_size is not None:
        VLLM_BATCH_SIZE = args.vllm_batch_size
    if args.vllm_tp is not None:
        VLLM_TENSOR_PARALLEL = args.vllm_tp

    if args.task == "query":
        run_style_transfer_query(top_k=args.top_k, batch_size=args.batch_size, backend=args.backend)
    