#!/usr/bin/env python3
"""
Evaluate embedding models (BM25, E5, GTR) with optional rerankers.

Query modes:
  - translated_only: use translated queries only (legacy evaluate_embeddings.py)
  - paired:          concatenate original + translated (legacy v2 behavior)
  - paired_en_once:  concatenate for non-EN only (legacy v2.5 behavior)

Usage:
    # Translated-only (legacy)
    python -m evaluation.evaluate_embeddings \
        --embedding_model bm25 \
        --dataset_dir ../Dataset_translated \
        --top_k 100 \
        --query_mode translated_only

    # Paired (original + translated)
    python -m evaluation.evaluate_embeddings \
        --embedding_model intfloat/e5-large-v2 \
        --dataset_dir ../Dataset_translated \
        --original_dataset_dir ../Dataset \
        --top_k 100 \
        --query_mode paired

    # Paired with EN once (v2.5)
    python -m evaluation.evaluate_embeddings \
        --embedding_model intfloat/e5-large-v2 \
        --dataset_dir ../Dataset_translated \
        --original_dataset_dir ../Dataset \
        --top_k 100 \
        --query_mode paired_en_once
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from embedding_models import get_embedding_model, get_embedding_model_slug
from rerankers import get_reranker
from evaluation.utils import (
    load_collection,
    load_queries,
    evaluate_mrr,
    evaluate_hit_rate,
    evaluate_ranks,
    generate_report,
    append_rank_report,
    macro_average_results,
    save_scores_json,
)

QUERY_MODE_CHOICES = ["translated_only", "paired", "paired_en_once"]


def resolve_cache_tag(cache_tag: str | None, query_mode: str) -> str:
    if cache_tag is not None:
        return cache_tag
    if query_mode == "paired":
        return "v2"
    if query_mode == "paired_en_once":
        return "v2_5"
    return ""


def cache_suffix(cache_tag: str) -> str:
    return f"_{cache_tag}" if cache_tag else ""


def use_legacy_paths(query_mode: str, cache_tag: str, split: str, query_prefix: str) -> bool:
    return query_mode == "translated_only" and not cache_tag and split == "dev" and not query_prefix


def load_paired_queries(translated_path: str, original_path: str):
    with open(translated_path, "r", encoding="utf-8") as f:
        translated = json.load(f)
    with open(original_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    orig_by_idx = {q["index"]: q["text"] for q in original}
    paired = []
    for q in translated:
        idx = q["index"]
        orig_text = orig_by_idx.get(idx, q["text"])
        paired.append((idx, orig_text, q["text"], q.get("pubkey")))
    return paired


def build_query_texts_for_embedding(queries, query_mode: str):
    if query_mode == "translated_only":
        return [q[1] for q in queries]

    combined = []
    for q in queries:
        orig, trans = q[1], q[2]
        if query_mode == "paired_en_once" and orig == trans:
            combined.append(orig)
        else:
            combined.append(f"{orig}\n{trans}")
    return combined


def to_standard_queries(queries, query_mode: str):
    if query_mode == "translated_only":
        return queries
    return [(q[0], q[2], q[3]) for q in queries]


def query_texts_for_reranker(queries, query_mode: str):
    if query_mode == "translated_only":
        return [q[1] for q in queries]
    return [q[2] for q in queries]


def query_paths(args, lang: str):
    translated_file = os.path.join(args.dataset_dir, f"{args.query_prefix}{lang}_{args.split}.json")
    original_file = None
    if args.original_dataset_dir:
        original_file = os.path.join(args.original_dataset_dir, f"{args.query_prefix}{lang}_{args.split}.json")
    return translated_file, original_file


def retrieval_cache_paths(cache_dir: str, emb_slug: str, cache_tag: str, lang: str, split: str):
    primary = os.path.join(cache_dir, f"emb_{emb_slug}{cache_suffix(cache_tag)}_retrieval_{lang}_{split}.npz")
    legacy = os.path.join(cache_dir, f"emb_{emb_slug}_retrieval_{lang}.npz")
    return primary, legacy


def reranker_cache_paths(cache_dir: str, emb_slug: str, cache_tag: str, reranker_slug: str, lang: str, split: str):
    primary = os.path.join(
        cache_dir,
        f"emb_{emb_slug}{cache_suffix(cache_tag)}_rerank_{reranker_slug}_{lang}_{split}.npz",
    )
    legacy = os.path.join(cache_dir, f"emb_{emb_slug}_rerank_{reranker_slug}_{lang}.npz")
    return primary, legacy


def choose_existing_path(primary: str, legacy: str) -> str:
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(legacy):
        return legacy
    return primary


def load_queries_for_lang(args, lang: str, allow_fallback: bool):
    translated_file, original_file = query_paths(args, lang)
    if not os.path.isfile(translated_file):
        print(f"  Skipping {lang}: {translated_file} not found")
        return None

    if args.query_mode == "translated_only":
        return load_queries(translated_file)

    if original_file and os.path.isfile(original_file):
        return load_paired_queries(translated_file, original_file)

    if allow_fallback:
        queries_raw = load_queries(translated_file)
        return [(q[0], q[1], q[1], q[2]) for q in queries_raw]

    print(f"  Skipping {lang}: original queries not found for paired mode")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models with optional rerankers"
    )
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--original_dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results_embeddings")
    parser.add_argument("--cache_dir", type=str, default="eval_cache_embeddings")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--query_prefix", type=str, default="")
    parser.add_argument("--query_mode", type=str, choices=QUERY_MODE_CHOICES, default="translated_only")
    parser.add_argument("--cache_tag", type=str, default=None)
    parser.add_argument("--reranker", type=str, default=None)
    parser.add_argument("--reranker_top_k", type=int, default=None)
    parser.add_argument("--languages", nargs="+", default=["en", "de", "fr"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--exclude_venue", action="store_true")
    parser.add_argument("--save_scores", action="store_true")
    args = parser.parse_args()

    if args.query_mode != "translated_only" and not args.original_dataset_dir:
        parser.error("--original_dataset_dir is required for paired query modes")

    args.cache_tag = resolve_cache_tag(args.cache_tag, args.query_mode)
    legacy_paths = use_legacy_paths(args.query_mode, args.cache_tag, args.split, args.query_prefix)

    if args.reranker_top_k is None:
        args.reranker_top_k = args.top_k
    if args.reranker and args.reranker_top_k > args.top_k:
        parser.error(
            f"--reranker_top_k ({args.reranker_top_k}) must be <= --top_k ({args.top_k})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    emb_slug = get_embedding_model_slug(args.embedding_model)

    print("Loading corpus...")
    result = load_collection(
        os.path.join(args.dataset_dir, "collection_data.json"),
        include_metadata=not args.no_metadata,
        exclude_venue=args.exclude_venue,
    )
    pubkeys, doc_texts = result[0], result[1]
    print(f"  {len(pubkeys)} documents")

    if not args.cached:
        emb_model = get_embedding_model(args.embedding_model)

        print(f"\nIndexing corpus with {args.embedding_model}...")
        emb_model.index_corpus(doc_texts, pubkeys, args.cache_dir, args.batch_size)

        for lang in args.languages:
            queries = load_queries_for_lang(args, lang, allow_fallback=False)
            if queries is None:
                continue

            if args.max_queries is not None and len(queries) > args.max_queries:
                queries = queries[:args.max_queries]

            retrieval_primary, retrieval_legacy = retrieval_cache_paths(
                args.cache_dir, emb_slug, args.cache_tag, lang, args.split
            )
            retrieval_path = retrieval_legacy if legacy_paths else retrieval_primary
            if os.path.isfile(retrieval_path):
                print(f"  {lang.upper()}: cached retrieval found, skipping")
                continue

            query_texts = build_query_texts_for_embedding(queries, args.query_mode)
            print(f"  {lang.upper()}: retrieving top-{args.top_k} for {len(queries)} queries...")
            result = emb_model.retrieve(
                query_texts, args.top_k, args.cache_dir, lang, args.batch_size
            )
            if isinstance(result, tuple):
                indices, scores = result
            else:
                indices, scores = result, None

            np.savez_compressed(retrieval_path, indices=indices)
            if scores is not None:
                scores_path = retrieval_path.replace(".npz", "_scores.npz")
                np.savez_compressed(scores_path, scores=scores)
            print(f"  {lang.upper()}: cached to {retrieval_path}")

        del emb_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Embedding model unloaded, GPU memory freed.")

    print("\nEvaluating...")
    mrr_k = [1, 3, 5, 10]
    hr_k = [1, 5, 10, 50, 100]
    all_results = {}
    all_queries_combined = []
    all_indices_combined = []

    reranker = None
    if args.reranker:
        print(f"\nLoading reranker: {args.reranker}")
        reranker = get_reranker(args.reranker)

    split_label = args.split.capitalize()

    for lang in args.languages:
        queries = load_queries_for_lang(args, lang, allow_fallback=True)
        if queries is None:
            continue

        if args.max_queries is not None and len(queries) > args.max_queries:
            print(f"  {lang.upper()}: limiting from {len(queries)} to {args.max_queries} queries")
            queries = queries[:args.max_queries]

        retrieval_primary, retrieval_legacy = retrieval_cache_paths(
            args.cache_dir, emb_slug, args.cache_tag, lang, args.split
        )
        retrieval_path = choose_existing_path(retrieval_primary, retrieval_legacy)
        if not os.path.isfile(retrieval_path):
            print(f"  {lang.upper()}: no cached retrieval at {retrieval_path}, skipping")
            continue

        data = np.load(retrieval_path)
        indices = data["indices"][:len(queries)]

        emb_scores_path = retrieval_path.replace(".npz", "_scores.npz")
        emb_scores = None
        if os.path.isfile(emb_scores_path):
            emb_scores = np.load(emb_scores_path)["scores"][:len(queries)]

        print(f"\n--- {lang.upper()} ({len(queries)} queries) ---")

        lang_reranker_scores = None
        if reranker is not None:
            reranker_slug = args.reranker.replace("/", "_")
            reranker_primary, reranker_legacy = reranker_cache_paths(
                args.cache_dir, emb_slug, args.cache_tag, reranker_slug, lang, args.split
            )
            reranker_cache_path = reranker_legacy if legacy_paths else reranker_primary
            reranker_cache_scores_path = reranker_cache_path.replace(".npz", "_scores.npz")

            if os.path.isfile(reranker_cache_path):
                rerank_data = np.load(reranker_cache_path)
                indices = rerank_data["indices"]
                print("  Loaded cached reranked results")
                if args.save_scores and os.path.isfile(reranker_cache_scores_path):
                    sdata = np.load(reranker_cache_scores_path)
                    lang_reranker_scores = sdata["scores"]
            else:
                print(f"  Reranking top {args.reranker_top_k}...")
                t0 = time.time()
                query_texts = query_texts_for_reranker(queries, args.query_mode)
                result = reranker.rerank_queries(
                    query_texts,
                    doc_texts,
                    indices,
                    args.reranker_top_k,
                    return_scores=args.save_scores,
                )
                if args.save_scores:
                    indices, reranker_scores = result
                    lang_reranker_scores = np.array(reranker_scores)
                    np.savez_compressed(reranker_cache_scores_path, scores=lang_reranker_scores)
                else:
                    indices = result
                np.savez_compressed(reranker_cache_path, indices=np.array(indices))
                print(f"  Reranking done in {time.time() - t0:.1f}s")

        if args.save_scores:
            scores_dir = os.path.join(args.output_dir, "scores")
            scores_base = f"{emb_slug}{cache_suffix(args.cache_tag)}_{lang}"
            if reranker is not None:
                scores_base += f"_rerank_{args.reranker.replace('/', '_')}"
            scores_path = os.path.join(scores_dir, f"{scores_base}_top{args.reranker_top_k}.json")

            queries_standard = to_standard_queries(queries, args.query_mode)
            save_scores_json(
                scores_path,
                queries_standard,
                indices,
                emb_scores,
                pubkeys,
                doc_texts,
                rerank_indices=indices if reranker else None,
                rerank_scores=lang_reranker_scores,
            )

        queries_standard = to_standard_queries(queries, args.query_mode)
        mrr = evaluate_mrr(queries_standard, indices, pubkeys, list_k=mrr_k)
        hr = evaluate_hit_rate(queries_standard, indices, pubkeys, list_k=hr_k)
        rank_stats = evaluate_ranks(queries_standard, indices, pubkeys)

        all_results[f"{lang.upper()} {split_label}"] = {
            "mrr": mrr,
            "hr": hr,
            "ranks": rank_stats,
            "n_queries": len(queries_standard),
        }
        print(f"  MRR@5={mrr[5]:.4f}  HR@1={hr[1]:.4f}  HR@10={hr[10]:.4f}  HR@100={hr[100]:.4f}")
        if rank_stats["mean_rank"] is not None:
            print(f"  Mean Rank={rank_stats['mean_rank']:.1f}  Median Rank={rank_stats['median_rank']:.0f}")

        all_queries_combined.extend(queries_standard)
        all_indices_combined.append(indices)

    macro = macro_average_results(all_results, args.languages, suffix=f" {split_label}")
    if macro is not None:
        print("\n--- Combined (Macro-Average) ---")
        all_results["Combined"] = macro
        mrr = macro["mrr"]
        hr = macro["hr"]
        rank_stats = macro.get("ranks", {})
        print(f"  MRR@5={mrr[5]:.4f}  HR@1={hr[1]:.4f}  HR@10={hr.get(10, 0):.4f}  HR@100={hr.get(100, 0):.4f}")
        if rank_stats.get("mean_rank") is not None:
            print(f"  Mean Rank={rank_stats['mean_rank']:.1f}  Median Rank={rank_stats.get('median_rank', 0):.0f}")

    if reranker is not None:
        del reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Reranker unloaded, GPU memory freed.")

    reranker_part = f"_rerank_{args.reranker.replace('/', '_')}" if args.reranker else ""
    report_tag = cache_suffix(args.cache_tag)
    report_name = f"eval_{emb_slug}{report_tag}{reranker_part}_top{args.top_k}.md"
    report_path = os.path.join(args.output_dir, report_name)

    extra_header = [f"\n**Query Mode:** {args.query_mode}"]
    if args.cache_tag:
        extra_header.append(f"\n**Cache Tag:** {args.cache_tag}")
    if args.reranker:
        extra_header.append(f"\n**Reranker Top-K:** {args.reranker_top_k}")
        extra_header.append(f"\n**Top-K:** {args.top_k}")
    else:
        extra_header.append(f"\n**Top-K:** {args.top_k}")

    generate_report(
        all_results,
        report_path,
        title="Embedding Model Evaluation Report",
        model_name=args.embedding_model,
        reranker_name=args.reranker,
        extra_header_lines=extra_header,
    )
    append_rank_report(all_results, report_path)


if __name__ == "__main__":
    main()
