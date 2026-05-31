"""Reranker modules. Add new rerankers to RERANKER_REGISTRY as you create them."""

# Lazy imports — each reranker is only loaded when actually requested.
# This avoids ImportErrors when optional deps (sentence-transformers, etc.)
# are missing but the reranker isn't being used.
RERANKER_REGISTRY = {
    "BAAI/bge-reranker-v2-m3":       "rerankers.bge_reranker_v2_m3:BGERerankerV2M3",
    "bge-reranker-v2-m3":            "rerankers.bge_reranker_v2_m3:BGERerankerV2M3",
    "Qwen/Qwen3-Reranker-0.6B":      "rerankers.qwen3_reranker_06b:Qwen3Reranker06B",
    "Qwen3-Reranker-0.6B":           "rerankers.qwen3_reranker_06b:Qwen3Reranker06B",
    "Qwen/Qwen3-Reranker-8B":        "rerankers.qwen3_reranker_8b:Qwen3Reranker8B",
    "Qwen3-Reranker-8B":             "rerankers.qwen3_reranker_8b:Qwen3Reranker8B",
    "nvidia/llama-nemotron-rerank-vl-1b-v2":      "rerankers.nemotron_rerank_vl_1b_v2:NemotronRerankVL1BV2",
    "llama-nemotron-rerank-vl-1b-v2":             "rerankers.nemotron_rerank_vl_1b_v2:NemotronRerankVL1BV2",
    "jinaai/jina-reranker-v3":                    "rerankers.jina_reranker_v3:JinaRerankerV3",
    "jina-reranker-v3":                           "rerankers.jina_reranker_v3:JinaRerankerV3",
}


def _import_class(dotted_path: str):
    """Lazy import: 'module.path:ClassName' -> the class."""
    module_path, class_name = dotted_path.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_reranker(name: str):
    """Return an instantiated reranker given a model name or shorthand."""
    entry = RERANKER_REGISTRY.get(name)
    if entry is None:
        for key, val in RERANKER_REGISTRY.items():
            if key in name:
                entry = val
                break
    if entry is None:
        raise ValueError(
            f"Unknown reranker '{name}'. Available: {list(RERANKER_REGISTRY.keys())}"
        )
    cls = _import_class(entry)
    return cls(name)
