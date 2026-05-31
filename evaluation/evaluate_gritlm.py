#!/usr/bin/env python3
"""
Evaluate a finetuned GritLM model on CheckThat datasets.

Query modes:
  - translated_only: use translated queries only (legacy evaluate_gritlm.py)
  - paired:          concatenate original + translated (legacy v2 behavior)
  - paired_en_once:  concatenate for non-EN only (legacy v2.5 behavior)
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import faiss
import torch

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from rerankers import get_reranker
from evaluation.utils import (
    load_collection,
    load_queries,
    get_model_slug,
    evaluate_mrr,
    evaluate_hit_rate,
    evaluate_ranks,
    generate_report,
    append_rank_report,
    macro_average_results,
    save_scores_json,
)

QUERY_MODE_CHOICES = ["translated_only", "paired", "paired_en_once"]

INSTRUCTION = "Given a scientific claim from a social media post, retrieve the source paper"


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


# ------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------

def encode_corpus(model, texts, batch_size=256):
    return model.encode(
        texts,
        batch_size=batch_size,
        instruction="<|embed|>\n",
        show_progress_bar=True,
    ).astype(np.float32)


def encode_query_texts(model, query_texts, batch_size=256):
    instruction_texts = [f"\n{INSTRUCTION}\n<|embed|>\n{t}" for t in query_texts]
    return model.encode(
        instruction_texts,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)


# ------------------------------------------------------------------
# Query loading
# ------------------------------------------------------------------

def load_paired_queries(translated_path, original_path):
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


def load_queries_for_metrics(args, lang: str):
    translated_file, original_file = query_paths(args, lang)
    if not os.path.isfile(translated_file):
        return None

    if args.query_mode == "translated_only":
        return load_queries(translated_file)

    if original_file and os.path.isfile(original_file):
        return load_paired_queries(translated_file, original_file)

    queries_raw = load_queries(translated_file)
    return [(q[0], q[1], q[1], q[2]) for q in queries_raw]


# ------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------

def _cache_paths(cache_dir, model_slug, cache_tag):
    tag = cache_suffix(cache_tag)
    return {
        "corpus_embs": os.path.join(cache_dir, f"corpus_embs_{model_slug}.npy"),
        "corpus_pubkeys": os.path.join(cache_dir, f"corpus_pubkeys_{model_slug}.npy"),
        "corpus_texts": os.path.join(cache_dir, f"corpus_texts_{model_slug}.npy"),
        "faiss_results": os.path.join(cache_dir, f"faiss_results_{model_slug}{tag}.npz"),
        "faiss_scores": os.path.join(cache_dir, f"faiss_scores_{model_slug}{tag}.npz"),
    }


def _all_cached(paths, keys):
    return all(os.path.isfile(paths[k]) for k in keys)


def query_embs_path(cache_dir, model_slug, cache_tag, lang, split):
    tag = cache_suffix(cache_tag)
    return os.path.join(cache_dir, f"query_embs_{model_slug}{tag}_{lang}_{split}.npy")


def reranker_cache_path(cache_dir, model_slug, cache_tag, reranker_slug, lang, split):
    tag = cache_suffix(cache_tag)
    return os.path.join(cache_dir, f"{model_slug}{tag}_rerank_{reranker_slug}_{lang}_{split}.npz")


def load_or_encode_corpus(model, dataset_dir, cache_dir, model_slug, batch_size,
                          include_metadata=True, exclude_venue=False, max_docs=None):
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_paths(cache_dir, model_slug, cache_tag="")

    if _all_cached(p, ["corpus_embs", "corpus_pubkeys", "corpus_texts"]):
        print("  Loading cached corpus embeddings...")
        corpus_embs = np.load(p["corpus_embs"])
        pubkeys = np.load(p["corpus_pubkeys"], allow_pickle=True).tolist()
        doc_texts = np.load(p["corpus_texts"], allow_pickle=True).tolist()
        print(f"  Loaded {len(pubkeys)} cached document embeddings")
        if max_docs is not None and len(pubkeys) > max_docs:
            pubkeys = pubkeys[:max_docs]
            doc_texts = doc_texts[:max_docs]
            corpus_embs = corpus_embs[:max_docs]
            print(f"  Truncated to {max_docs} documents")
        return pubkeys, doc_texts, corpus_embs

    pubkeys, doc_texts = load_collection(
        os.path.join(dataset_dir, "collection_data.json"),
        include_metadata=include_metadata,
        exclude_venue=exclude_venue,
    )[:2]
    if max_docs is not None and len(pubkeys) > max_docs:
        pubkeys = pubkeys[:max_docs]
        doc_texts = doc_texts[:max_docs]
    print(f"  {len(pubkeys)} documents — encoding...")
    corpus_embs = encode_corpus(model, doc_texts, batch_size=batch_size)

    np.save(p["corpus_embs"], corpus_embs)
    np.save(p["corpus_pubkeys"], np.array(pubkeys))
    np.save(p["corpus_texts"], np.array(doc_texts, dtype=object))
    print(f"  Cached to {p['corpus_embs']}")
    return pubkeys, doc_texts, corpus_embs


def load_or_encode_queries(model, args, model_slug, cache_tag):
    all_query_data = {}

    for lang in args.languages:
        translated_file, original_file = query_paths(args, lang)
        if not os.path.isfile(translated_file):
            print(f"  Skipping {lang}: {translated_file} not found")
            continue

        if args.query_mode == "translated_only":
            queries = load_queries(translated_file)
        else:
            if not original_file or not os.path.isfile(original_file):
                print(f"  Skipping {lang}: original queries not found for paired mode")
                continue
            queries = load_paired_queries(translated_file, original_file)

        qemb_path = query_embs_path(args.cache_dir, model_slug, cache_tag, lang, args.split)
        if os.path.isfile(qemb_path):
            print(f"  {lang.upper()}: {len(queries)} queries — loading cached embeddings")
            query_embs = np.load(qemb_path)
        else:
            print(f"  {lang.upper()}: {len(queries)} queries — encoding...")
            query_texts = build_query_texts_for_embedding(queries, args.query_mode)
            query_embs = encode_query_texts(model, query_texts, batch_size=args.batch_size)
            np.save(qemb_path, query_embs)
            print(f"  Cached to {qemb_path}")

        all_query_data[lang] = (queries, query_embs)

    return all_query_data


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

def build_faiss_index(corpus_embs):
    dim = corpus_embs.shape[1]
    try:
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        index_cpu = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu, config)
        print("  Using FAISS GPU index")
    except Exception:
        index = faiss.IndexFlatIP(dim)
        print("  Using FAISS CPU index")
    index.add(corpus_embs)
    return index


def batch_retrieve(index, query_embs, top_k=100):
    scores, indices = index.search(query_embs, top_k)
    return indices, scores


def load_or_retrieve(faiss_index, all_query_data, faiss_path, scores_path, top_k, save_scores=False):
    if os.path.isfile(faiss_path):
        print("  Loading cached FAISS results...")
        data = np.load(faiss_path, allow_pickle=True)
        all_faiss_indices = {lang: data[lang] for lang in data.files}
        print(f"  Loaded cached results for: {list(all_faiss_indices.keys())}")

        all_faiss_scores = None
        if save_scores and os.path.isfile(scores_path):
            sdata = np.load(scores_path, allow_pickle=True)
            all_faiss_scores = {lang: sdata[lang] for lang in sdata.files}
            print(f"  Loaded cached scores for: {list(all_faiss_scores.keys())}")
        return all_faiss_indices, all_faiss_scores

    print("  Running FAISS search...")
    all_faiss_indices = {}
    all_faiss_scores = {}
    for lang, (queries, query_embs) in all_query_data.items():
        indices, scores = batch_retrieve(faiss_index, query_embs, top_k=top_k)
        all_faiss_indices[lang] = indices
        all_faiss_scores[lang] = scores

    np.savez(faiss_path, **all_faiss_indices)
    print(f"  Cached FAISS results to {faiss_path}")

    if save_scores:
        np.savez(scores_path, **all_faiss_scores)
        print(f"  Cached FAISS scores to {scores_path}")
        return all_faiss_indices, all_faiss_scores

    return all_faiss_indices, None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GritLM on CheckThat datasets")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="../Dataset_translated")
    parser.add_argument("--original_dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--languages", nargs="+", default=["en", "de", "fr"])
    parser.add_argument("--cache_dir", type=str, default="./eval_cache")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--reranker", type=str, default=None)
    parser.add_argument("--reranker_top_k", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--exclude_venue", action="store_true")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--save_scores", action="store_true")
    parser.add_argument("--query_prefix", type=str, default="")
    parser.add_argument("--query_mode", type=str, choices=QUERY_MODE_CHOICES, default="translated_only")
    parser.add_argument("--cache_tag", type=str, default=None)
    args = parser.parse_args()

    if args.query_mode != "translated_only" and not args.original_dataset_dir:
        parser.error("--original_dataset_dir is required for paired query modes")

    args.cache_tag = resolve_cache_tag(args.cache_tag, args.query_mode)

    if args.reranker:
        if args.reranker_top_k is None:
            args.reranker_top_k = args.top_k
        if args.reranker_top_k > args.top_k:
            raise ValueError(
                f"--reranker_top_k ({args.reranker_top_k}) must be <= --top_k ({args.top_k})"
            )
    else:
        args.reranker_top_k = args.top_k

    model_slug = get_model_slug(args.model_path)
    p = _cache_paths(args.cache_dir, model_slug, args.cache_tag)

    os.makedirs(args.cache_dir, exist_ok=True)
    corpus_cached = _all_cached(p, ["corpus_embs", "corpus_pubkeys", "corpus_texts"])

    if args.cached and not corpus_cached:
        raise FileNotFoundError(
            f"--cached requested but corpus not found in {args.cache_dir}. "
            "Run once without --cached first."
        )

    include_metadata = not args.no_metadata

    if corpus_cached:
        print("Loading cached corpus embeddings...")
        corpus_embs = np.load(p["corpus_embs"])
        pubkeys = np.load(p["corpus_pubkeys"], allow_pickle=True).tolist()
        doc_texts = np.load(p["corpus_texts"], allow_pickle=True).tolist()
        print(f"  {len(pubkeys)} documents loaded from cache")

        if args.max_docs is not None and len(pubkeys) > args.max_docs:
            pubkeys = pubkeys[:args.max_docs]
            doc_texts = doc_texts[:args.max_docs]
            corpus_embs = corpus_embs[:args.max_docs]
            print(f"  Truncated to {args.max_docs} documents")
    else:
        from gritlm import GritLM

        print(f"Loading model: {args.model_path}")
        model = GritLM(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            mode="embedding",
            attn_implementation="eager",
        )
        pubkeys, doc_texts, corpus_embs = load_or_encode_corpus(
            model,
            args.dataset_dir,
            args.cache_dir,
            model_slug,
            args.batch_size,
            include_metadata=include_metadata,
            exclude_venue=args.exclude_venue,
            max_docs=args.max_docs,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("  GritLM unloaded to free VRAM")

    faiss_cached = os.path.isfile(p["faiss_results"])
    if args.cached and not faiss_cached:
        raise FileNotFoundError(
            f"--cached requested but query/FAISS data not found. Run once without --cached first."
        )

    if faiss_cached:
        print("Loading cached FAISS results...")
        faiss_data = np.load(p["faiss_results"], allow_pickle=True)
        all_faiss_indices = {lang: faiss_data[lang] for lang in faiss_data.files}
        print(f"  Loaded results for: {list(all_faiss_indices.keys())}")

        all_faiss_scores = None
        if args.save_scores:
            if os.path.isfile(p["faiss_scores"]):
                sdata = np.load(p["faiss_scores"], allow_pickle=True)
                all_faiss_scores = {lang: sdata[lang] for lang in sdata.files}
                print(f"  Loaded cached scores for: {list(all_faiss_scores.keys())}")
            else:
                print("  WARNING: --save_scores set but no cached scores found.")

        all_query_data = {}
        for lang in args.languages:
            queries = load_queries_for_metrics(args, lang)
            if queries is not None:
                all_query_data[lang] = (queries, None)
    else:
        query_embs_cached = all(
            os.path.isfile(query_embs_path(args.cache_dir, model_slug, args.cache_tag, lang, args.split))
            for lang in args.languages
        )

        if not query_embs_cached:
            from gritlm import GritLM

            print(f"Loading model for query encoding: {args.model_path}")
            model = GritLM(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                mode="embedding",
                attn_implementation="eager",
            )
            all_query_data = load_or_encode_queries(model, args, model_slug, args.cache_tag)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("  GritLM unloaded to free VRAM")
        else:
            all_query_data = {}
            for lang in args.languages:
                translated_file, original_file = query_paths(args, lang)
                if not os.path.isfile(translated_file):
                    continue

                qemb_path = query_embs_path(args.cache_dir, model_slug, args.cache_tag, lang, args.split)
                query_embs = np.load(qemb_path)

                if args.query_mode == "translated_only":
                    queries = load_queries(translated_file)
                else:
                    if original_file and os.path.isfile(original_file):
                        queries = load_paired_queries(translated_file, original_file)
                    else:
                        queries_raw = load_queries(translated_file)
                        queries = [(q[0], q[1], q[1], q[2]) for q in queries_raw]

                all_query_data[lang] = (queries, query_embs)

        print("Building FAISS index...")
        faiss_index = build_faiss_index(corpus_embs)
        all_faiss_indices, all_faiss_scores = load_or_retrieve(
            faiss_index,
            all_query_data,
            p["faiss_results"],
            p["faiss_scores"],
            args.top_k,
            save_scores=args.save_scores,
        )

    mrr_k = [1, 3, 5, 10]
    hr_k = [1, 5, 10, 50, 100]
    all_results = {}

    reranker = None
    if args.reranker:
        print(f"\nLoading reranker: {args.reranker}")
        reranker = get_reranker(args.reranker)

    split_label = args.split.capitalize()
    rshort = os.path.basename(args.model_path).replace("/", "_")

    for lang in args.languages:
        query_data = all_query_data.get(lang)
        if query_data is None:
            continue
        queries = query_data[0]

        if args.max_queries is not None and len(queries) > args.max_queries:
            queries = queries[:args.max_queries]

        indices = all_faiss_indices[lang]
        if args.max_queries is not None and len(indices) > args.max_queries:
            indices = indices[:args.max_queries]
        faiss_indices_for_lang = indices

        print(f"\n--- {lang.upper()} ({len(queries)} queries) ---")

        lang_faiss_scores = all_faiss_scores[lang] if all_faiss_scores is not None else None
        lang_reranker_scores = None

        if reranker is not None:
            reranker_slug = args.reranker.replace("/", "_")
            reranker_path = reranker_cache_path(args.cache_dir, model_slug, args.cache_tag, reranker_slug, lang, args.split)
            reranker_scores_path = reranker_path.replace(".npz", "_scores.npz")

            if os.path.isfile(reranker_path):
                rerank_data = np.load(reranker_path)
                indices = rerank_data["indices"]
                print("  Loaded cached reranked results")
                if args.save_scores and os.path.isfile(reranker_scores_path):
                    sdata = np.load(reranker_scores_path)
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
                    np.savez_compressed(reranker_scores_path, scores=lang_reranker_scores)
                else:
                    indices = result
                np.savez_compressed(reranker_path, indices=np.array(indices))
                print(f"  Reranking done in {time.time() - t0:.1f}s")

        if args.save_scores:
            scores_dir = os.path.join(args.output_dir, "scores")
            scores_base = f"{rshort}{cache_suffix(args.cache_tag)}_{lang}"
            if reranker is not None:
                scores_base += f"_rerank_{args.reranker.replace('/', '_')}"
            scores_path = os.path.join(scores_dir, f"{scores_base}_top{args.reranker_top_k}.json")

            emb_scores_for_lang = lang_faiss_scores[:len(queries)] if lang_faiss_scores is not None else None
            rerank_scores_for_lang = lang_reranker_scores[:len(queries)] if lang_reranker_scores is not None else None

            queries_standard = to_standard_queries(queries, args.query_mode)
            save_scores_json(
                scores_path,
                queries_standard,
                faiss_indices_for_lang,
                emb_scores_for_lang,
                pubkeys,
                doc_texts,
                rerank_indices=indices if reranker else None,
                rerank_scores=rerank_scores_for_lang,
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

    reranker_part = f"_rerank_{args.reranker.replace('/', '_')}" if args.reranker else ""
    if args.cache_tag:
        report_base = f"{rshort}_{args.cache_tag}"
    else:
        report_base = model_slug
    report_name = f"eval_{report_base}{reranker_part}_top{args.reranker_top_k}.md"
    report_path = os.path.join(args.output_dir, report_name)

    extra_header = [f"\n**Query Mode:** {args.query_mode}"]
    if args.cache_tag:
        extra_header.append(f"\n**Cache Tag:** {args.cache_tag}")

    generate_report(
        all_results,
        report_path,
        model_name=args.model_path,
        reranker_name=args.reranker,
        extra_header_lines=extra_header,
    )
    append_rank_report(all_results, report_path)


if __name__ == "__main__":
    main()
