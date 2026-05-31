"""Embedding model registry for retrieval evaluation."""

EMBEDDING_MODEL_REGISTRY = {
    "bm25":                    "embedding_models.bm25:BM25Model",
    "intfloat/e5-large-v2":    "embedding_models.e5_large_v2:E5LargeV2Model",
    "e5-large-v2":             "embedding_models.e5_large_v2:E5LargeV2Model",
    "gtr-t5-xl":               "embedding_models.gtr_t5_xl:GTRT5XLModel",
    "sentence-transformers/gtr-t5-xl": "embedding_models.gtr_t5_xl:GTRT5XLModel",
}

EMBEDDING_MODEL_SLUGS = {
    "bm25":                    "bm25",
    "intfloat/e5-large-v2":    "e5-large-v2",
    "e5-large-v2":             "e5-large-v2",
    "gtr-t5-xl":               "gtr-t5-xl",
    "sentence-transformers/gtr-t5-xl": "gtr-t5-xl",
}


def _import_class(dotted_path: str):
    """Lazy import: 'module.path:ClassName' -> the class."""
    import importlib
    module_path, class_name = dotted_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_embedding_model(name: str):
    """Return an instantiated embedding model given a model name or shorthand."""
    entry = EMBEDDING_MODEL_REGISTRY.get(name)
    if entry is None:
        for key, val in EMBEDDING_MODEL_REGISTRY.items():
            if key in name:
                entry = val
                break
    if entry is None:
        raise ValueError(
            f"Unknown embedding model '{name}'. Available: {list(EMBEDDING_MODEL_REGISTRY.keys())}"
        )
    cls = _import_class(entry)
    return cls(name)


def get_embedding_model_slug(name: str) -> str:
    """Return the cache slug for a model without loading it."""
    slug = EMBEDDING_MODEL_SLUGS.get(name)
    if slug is not None:
        return slug
    for key, s in EMBEDDING_MODEL_SLUGS.items():
        if key in name:
            return s
    raise ValueError(
        f"Unknown embedding model '{name}'. Available: {list(EMBEDDING_MODEL_SLUGS.keys())}"
    )
