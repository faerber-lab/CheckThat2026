"""intfloat/e5-large-v2 embedding model using sentence-transformers."""

import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class E5LargeV2Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)
        self.index = None
        self.corpus_pubkeys = None

    @property
    def slug(self):
        return "e5-large-v2"

    def index_corpus(self, corpus_texts, corpus_pubkeys, cache_dir=None, batch_size=32):
        """Encode corpus with 'passage: ' prefix and build FAISS index."""
        self.corpus_pubkeys = list(corpus_pubkeys)

        emb_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            emb_path = os.path.join(cache_dir, f"emb_{self.slug}_corpus_embs.npy")
            if os.path.isfile(emb_path):
                corpus_embs = np.load(emb_path)
                self._build_index(corpus_embs)
                print(f"  Loaded cached corpus embeddings from {emb_path}")
                return

        # E5 requires "passage: " prefix for documents
        prefixed = [f"passage: {text}" for text in corpus_texts]
        corpus_embs = self.model.encode(
            prefixed, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        corpus_embs = np.array(corpus_embs, dtype=np.float32)

        if cache_dir:
            np.save(emb_path, corpus_embs)

        self._build_index(corpus_embs)
        print(f"  Corpus encoded: {corpus_embs.shape}")

    def _build_index(self, corpus_embs):
        dim = corpus_embs.shape[1]
        try:
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index_cpu = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu, config)
            print("  Using FAISS GPU index")
        except Exception:
            self.index = faiss.IndexFlatIP(dim)
            print("  Using FAISS CPU index")
        self.index.add(corpus_embs)

    def retrieve(self, query_texts, top_k, cache_dir=None, lang=None, batch_size=32):
        """Encode queries with 'query: ' prefix and search FAISS index."""
        # E5 requires "query: " prefix for queries
        prefixed = [f"query: {text}" for text in query_texts]
        query_embs = self.model.encode(
            prefixed, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_embs = np.array(query_embs, dtype=np.float32)

        scores, indices = self.index.search(query_embs, top_k)
        return indices, scores
