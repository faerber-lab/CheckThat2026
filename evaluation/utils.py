"""
Shared utilities for evaluation scripts.

Contains common data loading, metric computation, and report generation
functions used across evaluate_embeddings.py, evaluate_gritlm.py,
ensemble_rerankers.py, and llm_entity_reranker.py.
"""

import json
import os
from datetime import datetime

import numpy as np


def _json_default(obj):
    """Handle numpy types that aren't JSON serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_scores_json(
    output_path,
    queries,
    emb_indices,
    emb_scores,
    pubkeys,
    doc_texts,
    rerank_indices=None,
    rerank_scores=None,
):
    """Save per-query retrieval + reranker scores as a JSON file.

    Args:
        output_path:  path to write the .json file.
        queries:      list of query tuples [(index, text, pubkey), ...].
        emb_indices:  np.ndarray (n_queries, top_k) — corpus indices from embedding retrieval.
        emb_scores:   np.ndarray (n_queries, top_k) — embedding similarity scores (inner product).
                      Can be None if scores are not available.
        pubkeys:      list of document pubkeys (indexed by corpus position).
        doc_texts:    list of document texts (indexed by corpus position).
        rerank_indices: optional np.ndarray (n_queries, rerank_top_k) — after reranking.
        rerank_scores:   optional np.ndarray (n_queries, rerank_top_k) — reranker scores.
    """
    pubkeys_arr = np.array(pubkeys)

    # Pre-compute pubkey lookups for all queries at once
    emb_pubkeys = pubkeys_arr[emb_indices]
    rerank_pubkeys = pubkeys_arr[rerank_indices] if rerank_indices is not None else None

    records = []
    top_k = emb_indices.shape[1]

    for qi, q in enumerate(queries):
        q_idx, q_text, q_gt = q[0], q[1], q[2]

        candidates = []
        for rank in range(top_k):
            cidx = int(emb_indices[qi, rank])
            entry = {
                "rank": rank + 1,
                "corpus_index": cidx,
                "pubkey": str(emb_pubkeys[qi, rank]),
                "document_text": str(doc_texts[cidx]),
                "embedding_score": round(float(emb_scores[qi, rank]), 6) if emb_scores is not None else None,
            }
            candidates.append(entry)

        # Add reranker scores by matching corpus_index
        if rerank_indices is not None and rerank_scores is not None:
            rerank_map = {}
            for rr in range(rerank_indices.shape[1]):
                cidx = int(rerank_indices[qi, rr])
                rerank_map[cidx] = {
                    "reranker_rank": rr + 1,
                    "reranker_score": round(float(rerank_scores[qi, rr]), 6),
                }
            for c in candidates:
                info = rerank_map.get(c["corpus_index"])
                if info is not None:
                    c["reranker_rank"] = info["reranker_rank"]
                    c["reranker_score"] = info["reranker_score"]
                else:
                    c["reranker_rank"] = None
                    c["reranker_score"] = None

        records.append({
            "query_index": int(q_idx),
            "query_text": str(q_text),
            "ground_truth_pubkey": str(q_gt),
            "candidates": candidates,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, default=_json_default)
    print(f"  Scores saved to {output_path} ({len(records)} queries, "
          f"{len(records[0]['candidates'])} candidates each)", flush=True)


def macro_average_results(all_results, languages, suffix=" Dev"):
    """Compute macro-averaged Combined metrics from per-language results.

    For each metric key (mrr, hr, ranks), computes the arithmetic mean
    of the per-language values.  For rank dictionaries (mean_rank, median_rank)
    the mean of language means is used.  For 'n_queries', sums are returned.

    Args:
        all_results: dict {split_name: result_dict} — must contain entries
                     for each f"{lang.upper()}{suffix}".
        languages:   list of language codes, e.g. ["en", "de", "fr"].
        suffix:      suffix used in split names (default " Dev").

    Returns:
        dict resembling a single split result: {"mrr": {k: float}, "hr": {k: float},
               "ranks": {"mean_rank": float, ...}, "n_queries": int}
    """
    lang_keys = [f"{lang.upper()}{suffix}" for lang in languages]
    present = [k for k in lang_keys if k in all_results]

    if len(present) < 2:
        return None  # nothing meaningful to average

    n = len(present)

    # ── MRR ──
    mrr_keys = list(all_results[present[0]].get("mrr", {}).keys())
    mrr_macro = {k: float(np.mean([all_results[lk]["mrr"][k] for lk in present])) for k in mrr_keys}

    # ── HR ──
    hr_keys = list(all_results[present[0]].get("hr", {}).keys())
    hr_macro = {k: float(np.mean([all_results[lk]["hr"][k] for lk in present])) for k in hr_keys}

    # ── Ranks ──
    ranks_macro = {}
    any_has_ranks = any("ranks" in all_results[lk] for lk in present)
    if any_has_ranks:
        rank_fields = ["mean_rank", "median_rank",
                       "rank_1", "rank_2_5", "rank_6_10",
                       "rank_11_50", "rank_51_100", "not_found"]
        # derive from per-split distribution or stored fields
        for field in ["mean_rank", "median_rank"]:
            vals = [all_results[lk]["ranks"].get(field)
                    for lk in present
                    if "ranks" in all_results[lk] and all_results[lk]["ranks"].get(field) is not None]
            if vals:
                ranks_macro[field] = float(np.mean(vals))

        # Distribution counts — just sum them; report generation handles formatting
        if "ranks" in all_results[present[0]] and "distribution" in all_results[present[0]]["ranks"]:
            dist_macro = {}
            for bucket in all_results[present[0]]["ranks"]["distribution"]:
                dist_macro[bucket] = sum(
                    all_results[lk]["ranks"]["distribution"].get(bucket, 0) for lk in present
                )
            ranks_macro["distribution"] = dist_macro

    n_queries = sum(all_results[lk].get("n_queries", 0) for lk in present)

    return {"mrr": mrr_macro, "hr": hr_macro, "ranks": ranks_macro, "n_queries": n_queries}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_collection(path, include_metadata=True, exclude_venue=False, return_all_metadata=False):
    """Load collection data from a JSON file.

    Args:
        path: Path to collection_data.json.
        include_metadata: If True, append metadata to document text.
            If False, use only title + abstract.
        exclude_venue: If True and include_metadata is True, include authors
            but exclude venue from the document text.
        return_all_metadata: If True, return separate lists for titles, venues,
            authors, abstracts in addition to pubkeys and texts.
            (Default False for backward compatibility.)

    Returns:
        If return_all_metadata is False (default):
            tuple: (pubkeys, texts, [titles]) – titles only if present in data.
        If return_all_metadata is True:
            tuple: (pubkeys, texts, titles, venues, authors, abstracts)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pubkeys = []
    texts = []
    titles = []
    venues = []
    authors = []
    abstracts = []

    has_titles = False
    for doc in data:
        pubkeys.append(doc["pubkey"])
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")

        # Build the 'texts' entry according to existing logic
        if include_metadata:
            author_str = doc.get("authors", "")
            if exclude_venue:
                doc_text = f"{title} {abstract} Authors: {author_str}"
            else:
                venue = doc.get("venue", "")
                doc_text = f"{title} {abstract} Venue: {venue} Authors: {author_str}"
        else:
            doc_text = f"{title} {abstract}"
        texts.append(doc_text)

        # Collect separate metadata if requested
        if return_all_metadata:
            titles.append(title)
            venues.append(doc.get("venue", ""))
            authors.append(doc.get("authors", ""))
            abstracts.append(abstract)
        elif "title" in doc:
            # For backward compatibility, still collect titles if present
            titles.append(title)
            has_titles = True

    if return_all_metadata:
        return pubkeys, texts, titles, venues, authors, abstracts
    else:
        if has_titles or any(titles):
            return pubkeys, texts, titles
        return pubkeys, texts


def load_queries(path):
    """Load queries from a JSON file.

    Returns:
        list of tuples: [(index, text, pubkey), ...]
            pubkey is None if not present (e.g. test set without ground truth).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(q["index"], q["text"], q.get("pubkey")) for q in data]


def extract_query_pubkeys(queries):
    """Extract ground-truth pubkeys from query tuples or dicts."""
    if not queries:
        return []
    if isinstance(queries[0], dict):
        return [q.get("pubkey") or q.get("pub_key") for q in queries]
    return [q[2] for q in queries]


def get_model_slug(model_path):
    """Create a filesystem-safe slug from a model path for cache file names."""
    return model_path.replace("/", "_").replace(".", "_")


# ------------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------------

def evaluate_mrr(queries, indices, pubkeys, list_k=(1, 3, 5, 10)):
    """Compute MRR@K.

    Args:
        queries: list of query tuples or dicts with ground-truth pubkeys.
        indices: np.ndarray (n_queries, top_k) or list[list[int]].
        pubkeys: list of document pubkeys (ordered by corpus index).
        list_k: which K values to compute.

    Returns:
        dict: {k: float} mapping K to mean reciprocal rank.
    """
    pubkeys_arr = np.array(pubkeys)
    gt = np.array(extract_query_pubkeys(queries))
    results = {}

    if isinstance(indices, np.ndarray):
        for k in list_k:
            top_k_pks = pubkeys_arr[indices[:, :k]]
            found = top_k_pks == gt[:, None]
            rank = np.argmax(found, axis=1).astype(float)
            hit = found.any(axis=1)
            rr = np.where(hit, 1.0 / (rank + 1), 0.0)
            results[k] = rr.mean()
    else:
        for k in list_k:
            rrs = []
            for i, top_ids in enumerate(indices):
                top_pks = [pubkeys_arr[idx] for idx in top_ids[:k]]
                try:
                    pos = top_pks.index(gt[i])
                    rrs.append(1.0 / (pos + 1))
                except ValueError:
                    rrs.append(0.0)
            results[k] = np.mean(rrs)
    return results


def evaluate_hit_rate(queries, indices, pubkeys, list_k=(1, 5, 10, 50, 100)):
    """Compute HR@K.

    Args:
        queries: list of query tuples or dicts with ground-truth pubkeys.
        indices: np.ndarray (n_queries, top_k) or list[list[int]].
        pubkeys: list of document pubkeys (ordered by corpus index).
        list_k: which K values to compute.

    Returns:
        dict: {k: float} mapping K to hit rate.
    """
    pubkeys_arr = np.array(pubkeys)
    gt = np.array(extract_query_pubkeys(queries))
    results = {}

    if isinstance(indices, np.ndarray):
        for k in list_k:
            top_k_pks = pubkeys_arr[indices[:, :k]]
            found = (top_k_pks == gt[:, None]).any(axis=1)
            results[k] = found.mean()
    else:
        for k in list_k:
            hits = []
            for i, top_ids in enumerate(indices):
                top_pks = [pubkeys_arr[idx] for idx in top_ids[:k]]
                hits.append(gt[i] in top_pks)
            results[k] = np.mean(hits)
    return results


def evaluate_ranks(queries, indices, pubkeys):
    """Find the rank position of the ground truth for each query.

    Returns dict with:
      - 'ranks': list of per-query GT ranks (0 = rank 1, -1 = not found in top-K)
      - 'mean_rank': mean rank (excluding not-found), or None if none found
      - 'median_rank': median rank (excluding not-found), or None
      - 'distribution': dict of bucket -> count
    """
    pubkeys_arr = np.array(pubkeys)
    gt = np.array(extract_query_pubkeys(queries))
    ranks = []

    if isinstance(indices, np.ndarray):
        # Fully vectorized rank computation
        top_pks = pubkeys_arr[indices]
        matches = (top_pks == gt[:, None])
        first_match_idx = np.argmax(matches, axis=1)
        hit = matches.any(axis=1)
        ranks = np.where(hit, first_match_idx, -1).astype(int).tolist()
    else:
        for i, top_ids in enumerate(indices):
            top_pks = [pubkeys_arr[idx] for idx in top_ids]
            try:
                ranks.append(top_pks.index(gt[i]))
            except ValueError:
                ranks.append(-1)

    dist = {
        "rank 1": 0, "rank 2-5": 0, "rank 6-10": 0,
        "rank 11-50": 0, "rank 51-100": 0, "rank 101-500": 0,
        "rank 501-1000": 0, "not in top-1000": 0,
    }
    for r in ranks:
        if r == 0:
            dist["rank 1"] += 1
        elif 1 <= r <= 4:
            dist["rank 2-5"] += 1
        elif 5 <= r <= 9:
            dist["rank 6-10"] += 1
        elif 10 <= r <= 49:
            dist["rank 11-50"] += 1
        elif 50 <= r <= 99:
            dist["rank 51-100"] += 1
        elif 100 <= r <= 499:
            dist["rank 101-500"] += 1
        elif 500 <= r <= 999:
            dist["rank 501-1000"] += 1
        else:
            dist["not in top-1000"] += 1

    found_ranks = [r + 1 for r in ranks if r >= 0]  # 1-indexed
    return {
        "ranks": ranks,
        "mean_rank": float(np.mean(found_ranks)) if found_ranks else None,
        "median_rank": float(np.median(found_ranks)) if found_ranks else None,
        "distribution": dist,
    }


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def generate_report(all_results, output_path, title="Retrieval Evaluation Report",
                    model_name=None, reranker_name=None, extra_header_lines=None,
                    mrr_k=(1, 3, 5, 10), hr_k=(1, 5, 10, 50, 100)):
    """Generate a markdown evaluation report with MRR and HR tables.

    Args:
        all_results: dict {split_name: {'mrr': {k: float}, 'hr': {k: float}, 'n_queries': int}}
        output_path: path to write the .md file.
        title: report heading.
        model_name: model name for the report header.
        reranker_name: reranker name for the report header.
        extra_header_lines: list of additional "**Key:** value" header lines.
        mrr_k: which MRR@K values to show.
        hr_k: which HR@K values to show.
    """
    lines = [f"# {title}\n"]
    if model_name:
        lines.append(f"\n**Model:** `{model_name}`")
    if reranker_name:
        lines.append(f"\n**Reranker:** `{reranker_name}`")
    if extra_header_lines:
        lines.extend(extra_header_lines)
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # MRR table
    lines.append("## Mean Reciprocal Rank (MRR@K)\n")
    header = "| Split | " + " | ".join(f"MRR@{k}" for k in mrr_k) + " | Queries |"
    sep = "|---" + "|---" * len(mrr_k) + "|---|"
    lines.append(header)
    lines.append(sep)
    for split_name, r in all_results.items():
        row = f"| {split_name} | " + " | ".join(
            f"{r['mrr'].get(k, 0.0):.4f}" for k in mrr_k
        ) + f" | {r['n_queries']} |"
        lines.append(row)

    # HR table
    lines.append("\n## Hit Rate (HR@K)\n")
    header = "| Split | " + " | ".join(f"HR@{k}" for k in hr_k) + " | Queries |"
    sep = "|---" + "|---" * len(hr_k) + "|---|"
    lines.append(header)
    lines.append(sep)
    for split_name, r in all_results.items():
        row = f"| {split_name} | " + " | ".join(
            f"{r['hr'].get(k, 0.0):.4f}" for k in hr_k
        ) + f" | {r['n_queries']} |"
        lines.append(row)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {output_path}")


def append_rank_report(all_results, output_path):
    """Append a detailed per-query rank distribution to the existing report file."""
    lines = []
    lines.append("\n## Ground Truth Rank Distribution\n")
    header = ("| Split | Mean Rank | Median Rank | Rank 1 | Rank 2-5 | Rank 6-10 | "
              "Rank 11-50 | Rank 51-100 | Rank 101-500 | Rank 501-1000 | Not Found |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for split_name, r in all_results.items():
        rs = r.get("ranks", {})
        dist = rs.get("distribution", {})
        mean_r = f"{rs['mean_rank']:.1f}" if rs.get("mean_rank") is not None else "N/A"
        med_r = f"{rs['median_rank']:.0f}" if rs.get("median_rank") is not None else "N/A"
        row = (f"| {split_name} | {mean_r} | {med_r} "
               f"| {dist.get('rank 1', 0)} | {dist.get('rank 2-5', 0)} "
               f"| {dist.get('rank 6-10', 0)} | {dist.get('rank 11-50', 0)} "
               f"| {dist.get('rank 51-100', 0)} | {dist.get('rank 101-500', 0)} "
               f"| {dist.get('rank 501-1000', 0)} | {dist.get('not in top-1000', 0)} |")
        lines.append(row)

    lines.append("\n## Per-Query Ground Truth Ranks\n")
    for split_name, r in all_results.items():
        rs = r.get("ranks", {})
        rank_list = rs.get("ranks", [])
        if not rank_list:
            continue
        lines.append(f"### {split_name}\n")
        lines.append("```")
        for i, rank in enumerate(rank_list):
            pos = rank + 1 if rank >= 0 else "not found"
            lines.append(f"  Query {i}: GT at position {pos}")
        lines.append("```\n")

    with open(output_path, "a") as f:
        f.write("\n".join(lines) + "\n")