#!/usr/bin/env python3
"""Run a supported reranker on cached GRITLM FAISS results.

Supported rerankers:
  - BAAI/bge-reranker-v2-m3
  - jinaai/jina-reranker-v3
  - Qwen/Qwen3-Reranker-0.6B
  - Qwen/Qwen3-Reranker-8B
  - nvidia/llama-nemotron-rerank-vl-1b-v2
"""

import argparse
import gc
import inspect
import os
import sys
import time

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_cudnn_sdp(False)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from rerankers import get_reranker
from evaluation.utils import (
    load_queries,
    evaluate_mrr,
    evaluate_hit_rate,
    evaluate_ranks,
    generate_report,
    append_rank_report,
    macro_average_results,
    get_model_slug,
    save_scores_json,
)

SUPPORTED_RERANKERS = [
    "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3",
    "jinaai/jina-reranker-v3",
    "jina-reranker-v3",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-8B",
    "Qwen3-Reranker-8B",
    "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "llama-nemotron-rerank-vl-1b-v2",
]


def cache_suffix(cache_tag: str) -> str:
    return f"_{cache_tag}" if cache_tag else ""


def resolve_faiss_paths(cache_dir: str, model_slug: str, cache_tag: str):
    tagged_slug = f"{model_slug}_{cache_tag}" if cache_tag else model_slug
    faiss_results_path = os.path.join(cache_dir, f"faiss_results_{tagged_slug}.npz")
    if cache_tag and not os.path.isfile(faiss_results_path):
        fallback = os.path.join(cache_dir, f"faiss_results_{model_slug}.npz")
        print(f"  WARNING: {faiss_results_path} not found; falling back to {fallback}")
        faiss_results_path = fallback

    faiss_scores_path = os.path.join(cache_dir, f"faiss_scores_{tagged_slug}.npz")
    if cache_tag and not os.path.isfile(faiss_scores_path):
        faiss_scores_path = os.path.join(cache_dir, f"faiss_scores_{model_slug}.npz")

    return tagged_slug, faiss_results_path, faiss_scores_path


def call_rerank_queries(
    reranker,
    query_texts,
    doc_texts,
    faiss_indices,
    reranker_top_k,
    return_scores,
    batch_size,
    query_batch_size,
):
    sig = inspect.signature(reranker.rerank_queries)
    kwargs = {"return_scores": return_scores}
    if "batch_size" in sig.parameters:
        kwargs["batch_size"] = batch_size
    if "query_batch_size" in sig.parameters:
        kwargs["query_batch_size"] = query_batch_size

    return reranker.rerank_queries(
        query_texts,
        doc_texts,
        faiss_indices,
        reranker_top_k,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description="Run a reranker on cached FAISS results")
    parser.add_argument("--reranker", type=str, required=True, help="Reranker model name")
    parser.add_argument("--reranker_top_k", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default="GRITLM_finetune/eval_cache_gritlm_translated")
    parser.add_argument("--dataset_dir", type=str, default="CT26/Dataset_translated")
    parser.add_argument("--output_dir", type=str, default="GRITLM_finetune/eval_results_rerankers")
    parser.add_argument("--model_path", type=str, default="GritLM-7B")
    parser.add_argument("--languages", nargs="+", default=["en", "de", "fr"])
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--query_prefix", type=str, default="")
    parser.add_argument("--cache_tag", type=str, default="")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--query_batch_size", type=int, default=25)
    parser.add_argument("--save_scores", action="store_true")
    parser.add_argument("--eval_shards", action="store_true", help="Evaluate per-shard results")
    args = parser.parse_args()

    if args.reranker not in SUPPORTED_RERANKERS:
        print("Unsupported reranker. Supported options:")
        for name in SUPPORTED_RERANKERS:
            print(f"  - {name}")
        raise SystemExit(2)

    os.makedirs(args.output_dir, exist_ok=True)

    model_slug = get_model_slug(args.model_path)
    tagged_slug, faiss_results_path, faiss_scores_path = resolve_faiss_paths(
        args.cache_dir, model_slug, args.cache_tag
    )

    print("Loading cached GRITLM data...")
    pubkeys = np.load(os.path.join(args.cache_dir, f"corpus_pubkeys_{model_slug}.npy"), allow_pickle=True).tolist()
    doc_texts = np.load(os.path.join(args.cache_dir, f"corpus_texts_{model_slug}.npy"), allow_pickle=True).tolist()

    if not os.path.isfile(faiss_results_path):
        raise FileNotFoundError(f"FAISS results not found at {faiss_results_path}")

    faiss_data = np.load(faiss_results_path, allow_pickle=True)
    all_faiss_indices = {lang: faiss_data[lang] for lang in faiss_data.files}
    print(f"  {len(pubkeys)} documents, languages: {list(all_faiss_indices.keys())}")

    all_faiss_scores = None
    if os.path.isfile(faiss_scores_path):
        sdata = np.load(faiss_scores_path, allow_pickle=True)
        all_faiss_scores = {lang: sdata[lang] for lang in sdata.files}
        print("  Loaded FAISS scores")

    all_query_data = {}
    for lang in args.languages:
        split_file = os.path.join(args.dataset_dir, f"{args.query_prefix}{lang}_{args.split}.json")
        if os.path.isfile(split_file):
            all_query_data[lang] = load_queries(split_file)
            print(f"  {lang.upper()}: {len(all_query_data[lang])} queries")

    print(f"\nLoading reranker: {args.reranker}")
    if args.num_shards > 1:
        print(f"  Shard {args.shard}/{args.num_shards}")
    reranker = get_reranker(args.reranker)
    reranker_slug = args.reranker.replace("/", "_")

    mrr_k = [1, 3, 5, 10]
    hr_k = [1, 5, 10, 50, 100]
    all_results = {}

    for lang in args.languages:
        if lang not in all_query_data or lang not in all_faiss_indices:
            continue

        queries = all_query_data[lang]
        faiss_indices = all_faiss_indices[lang]

        if args.max_queries is not None:
            queries = queries[:args.max_queries]
            faiss_indices = faiss_indices[:args.max_queries]

        n_total = len(queries)
        shard_start = (n_total * args.shard) // args.num_shards
        shard_end = (n_total * (args.shard + 1)) // args.num_shards
        if args.num_shards > 1:
            print(f"\n  Shard {args.shard}/{args.num_shards}: queries [{shard_start}:{shard_end}] of {n_total}")
        queries = queries[shard_start:shard_end]
        faiss_indices = faiss_indices[shard_start:shard_end]

        query_texts = [q[1] for q in queries]
        faiss_scores_lang = None
        if all_faiss_scores is not None:
            faiss_scores_lang = all_faiss_scores[lang][shard_start:shard_end]

        print(f"\n--- {lang.upper()} ({len(queries)} queries) ---")

        shard_suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
        reranker_cache = os.path.join(
            args.cache_dir,
            f"{tagged_slug}_rerank_{reranker_slug}_{lang}_{args.split}{shard_suffix}.npz",
        )
        reranker_scores_cache = reranker_cache.replace(".npz", "_scores.npz")

        cached_indices = None
        if os.path.isfile(reranker_cache):
            cached_indices = np.load(reranker_cache)["indices"]
            if cached_indices.shape[0] != len(queries):
                print(f"  Cache has {cached_indices.shape[0]} rows, need {len(queries)} — re-running")
                cached_indices = None

        reranker_scores = None
        if cached_indices is not None:
            print("  Loading cached reranked results...")
            indices = cached_indices
            if os.path.isfile(reranker_scores_cache):
                reranker_scores = np.load(reranker_scores_cache)["scores"]
        else:
            print(f"  Reranking top {args.reranker_top_k}...")
            t0 = time.time()
            return_scores = args.save_scores
            result = call_rerank_queries(
                reranker,
                query_texts,
                doc_texts,
                faiss_indices,
                min(args.reranker_top_k, faiss_indices.shape[1]),
                return_scores,
                args.batch_size,
                args.query_batch_size,
            )
            if return_scores:
                indices, reranker_scores = result
            else:
                indices = result
            indices = np.array(indices)
            np.savez_compressed(reranker_cache, indices=indices)
            if reranker_scores is not None:
                np.savez_compressed(reranker_scores_cache, scores=np.array(reranker_scores))
            print(f"  Reranking done in {time.time() - t0:.1f}s")

        if args.save_scores:
            scores_dir = os.path.join(args.output_dir, "scores")
            os.makedirs(scores_dir, exist_ok=True)
            scores_path = os.path.join(
                scores_dir,
                f"{reranker_slug}_{lang}_top{args.reranker_top_k}{shard_suffix}.json",
            )
            save_scores_json(
                scores_path,
                queries,
                faiss_indices[:len(queries)],
                faiss_scores_lang[:len(queries)] if faiss_scores_lang is not None else None,
                pubkeys,
                doc_texts,
                rerank_indices=indices[:len(queries)],
                rerank_scores=reranker_scores[:len(queries)] if reranker_scores is not None else None,
            )

        if args.num_shards > 1 and not args.eval_shards:
            print(f"  Shard {args.shard} complete for {lang.upper()} — skipping evaluation")
            continue

        mrr = evaluate_mrr(queries, indices, pubkeys, list_k=mrr_k)
        hr = evaluate_hit_rate(queries, indices, pubkeys, list_k=hr_k)
        rank_stats = evaluate_ranks(queries, indices, pubkeys)

        all_results[f"{lang.upper()} {args.split.capitalize()}"] = {
            "mrr": mrr,
            "hr": hr,
            "ranks": rank_stats,
            "n_queries": len(queries),
        }
        print(f"  MRR@5={mrr[5]:.4f}  HR@1={hr[1]:.4f}  HR@10={hr[10]:.4f}  HR@100={hr[100]:.4f}")
        if rank_stats["mean_rank"] is not None:
            print(f"  Mean Rank={rank_stats['mean_rank']:.1f}  Median Rank={rank_stats['median_rank']:.0f}")

    if all_results and (args.num_shards == 1 or args.eval_shards):
        macro = macro_average_results(all_results, args.languages)
        if macro is not None:
            all_results["Combined"] = macro
            mrr = macro["mrr"]
            hr = macro["hr"]
            print(f"\n--- Combined (Macro-Average) ---")
            print(f"  MRR@5={mrr[5]:.4f}  HR@1={hr[1]:.4f}  HR@10={hr.get(10, 0):.4f}  HR@100={hr.get(100, 0):.4f}")

        shard_suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
        report_name = (
            f"eval_{model_slug}{cache_suffix(args.cache_tag)}_rerank_{reranker_slug}_top{args.reranker_top_k}_{args.split}{shard_suffix}.md"
        )
        report_path = os.path.join(args.output_dir, report_name)
        generate_report(all_results, report_path, model_name="GritLM-7B", reranker_name=args.reranker)
        append_rank_report(all_results, report_path)

    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
