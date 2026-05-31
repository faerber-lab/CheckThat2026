import os
import json
import numpy as np
import torch
import faiss
import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. Base Configurations and Dynamic Workspace Path Resolution
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to the sibling "data" directory relative to this script's parent folder
DATA_PATH = os.path.join(BASE_DIR, os.path.abspath(os.path.join(BASE_DIR, "..", "sample_data")))
SYNTHETIC_DIR = os.path.join(BASE_DIR,"out", "synthetic_queries")
CORPUS_PATH = os.path.join(DATA_PATH, "collection_data_process.json")
CACHE_DIR = os.path.join(BASE_DIR, "eval_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "mrr_results")

# Model checkpoints directory paths (Hugging Face local snapshots paths)
MODELS = {
    "bm25": "bm25",
    "gtr": os.path.join(BASE_DIR, "hf_home", "hub", "models--sentence-transformers--gtr-t5-xl"),
    "e5": os.path.join(BASE_DIR, "hf_home", "hub", "models--intfloat--e5-large-v2"),
    "gritlm": os.path.join(BASE_DIR, "hf_home", "hub", "models--GritLM--GritLM-7B", "snapshots", "cb7f7ffb99c0c24ca2c325d798d7ef5c455d5339")
}

# Fallback IDs to HF Hub if local directories are absent
FALLBACK_IDS = {
    "gtr": "sentence-transformers/gtr-t5-xl",
    "e5": "intfloat/e5-large-v2",
    "gritlm": "GritLM/GritLM-7B"
}


# ==============================================================================
# 1.5. Dynamic Cache Path Resolver (For structured HF Hub Snapshot directories)
# ==============================================================================
def get_model_path_or_id(model_key):
    path = MODELS.get(model_key)
    if not path or path == "bm25":
        return "bm25"
    
    # Resolve snapshot commits within standard huggingface cache folder
    if os.path.isdir(path):
        snapshots_dir = os.path.join(path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                resolved_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"🎯 [Path Resolution] Redirected {model_key} cache directory to correct snapshot commitment path:")
                print(f"   --> {resolved_path}")
                return resolved_path
        return path
    
    # Fallback to official repo identifier
    fallback_id = FALLBACK_IDS.get(model_key, path)
    print(f"⚠️ [Path Warning] Local model path not found: {path}. Using official Hugging Face ID: {fallback_id}")
    return fallback_id


# ==============================================================================
# 2. Evaluation Metric Computations
# ==============================================================================
def calculate_mrr(target_pubkey, retrieved_pubkeys, k=5):
    """Computes MRR@K for a single query."""
    for i, pubkey in enumerate(retrieved_pubkeys[:k]):
        if str(pubkey) == str(target_pubkey):
            return 1.0 / (i + 1)
    return 0.0


# ==============================================================================
# 3. Data Loading Helpers
# ==============================================================================
def load_corpus():
    print(f"📦 Loading dataset corpus file: {CORPUS_PATH}")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f) 
    
    print(f"📊 Parsing JSON completed. Total documents: {len(data)}. Concatenating fields...")
    texts = []
    pubkeys = []
    for i, item in enumerate(data):
        full_text = f"{item.get('title', '')} {item.get('abstract', '')} {item.get('venue', '')} {item.get('authors', '')}"
        texts.append(full_text.strip())
        pubkeys.append(str(item['pubkey']))
        if (i + 1) % 50000 == 0:
            print(f"   Processed {i + 1} documents...")
            
    print(f"✅ Text fields merged. Actual corpus size: {len(texts)}")
    return texts, pubkeys


def load_queries(lang, condition, version):
    """Loads query data based on condition (C0 is baseline, C1-C4 are styles)."""
    if condition == 0:
        path = os.path.join(DATA_DIR, "trans", f"{lang}_dev.json")
        text_field = "text"
    else:
        path = os.path.join(SYNTHETIC_DIR, f"synthetic_queries_{lang}_C{condition}_V{version}.json")
        text_field = "query_transfer"
    
    if not os.path.exists(path):
        print(f"⚠️ Query file does not exist: {path}")
        return [], []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item.get(text_field, "").strip() for item in data]
    target_pubkeys = [str(item.get('pubkey')) for item in data]
    return texts, target_pubkeys


# ==============================================================================
# 4. Retrieval Runner Interfaces (CPU/GPU-agnostic safe wrappers)
# ==============================================================================

# --- BM25 ---
def _score_query_bm25(args):
    bm25, query, top_k = args
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(-scores)[:top_k]
    return top_indices


class BM25Runner:
    def __init__(self):
        self.bm25 = None
        self.pubkeys = None
    
    def index(self, corpus_texts, pubkeys):
        print("⚡ Tokenizing BM25 corpus...")
        tokenized_corpus = [doc.split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.pubkeys = pubkeys
        print(f"✅ BM25 indexing complete ({len(pubkeys)} documents indexed)")

    def retrieve(self, queries, top_k=100):
        args = [(self.bm25, q, top_k) for q in queries]
        # Multi-threaded parallel retrieval scoring
        with ThreadPoolExecutor(max_workers=4) as pool:
            indices_list = list(pool.map(_score_query_bm25, args))
        results = []
        for indices in indices_list:
            results.append([self.pubkeys[i] for i in indices])
        return results


# --- Dense Retrieval (E5 / GTR) ---
class DenseRunner:
    def __init__(self, model_type, model_path):
        self.type = model_type  # 'e5' or 'gtr'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        self.faiss_index = None  
        self.pubkeys = None

    def index(self, texts, pubkeys):
        self.pubkeys = pubkeys
        cache_file = os.path.join(CACHE_DIR, f"corpus_{self.type}.npy")
        if os.path.exists(cache_file):
            print(f"📍 [{self.type.upper()} Runner] Cache found. Direct-loading numpy array...")
            embs = np.load(cache_file)
        else:
            print(f"📍 [{self.type.upper()} Runner] Cache absent. Computing dense corpus embeddings...")
            if self.type == 'e5':
                texts = [f"passage: {t}" for t in texts]
            embs = self.model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
            embs = np.array(embs, dtype=np.float32)
            np.save(cache_file, embs)
        
        dim = embs.shape[1]
        
        # Load FAISS index with automatic GPU-to-CPU downgrade on missing dependencies
        try:
            res = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            print(f"📍 [{self.type.upper()} Runner] Initialized FAISS GPU index successfully")
        except (AttributeError, Exception) as e:
            self.faiss_index = faiss.IndexFlatIP(dim)
            print(f"📍 [{self.type.upper()} Runner] FAISS GPU unavailable. Defaulted to IndexFlatIP CPU index. Info: {e}")
            
        self.faiss_index.add(embs.astype('float32'))
        print(f"✅ {self.type.upper()} indexing completed successfully.")


    def retrieve(self, queries, top_k=100):
        if self.type == 'e5':
            queries = [f"query: {q}" for q in queries]
        q_embs = self.model.encode(queries, batch_size=256, normalize_embeddings=True)
        scores, indices = self.faiss_index.search(q_embs.astype('float32'), top_k)
        return [[self.pubkeys[idx] for idx in row] for row in indices]


# --- GritLM Dense Retrieval (Safe Wrapper Interface) ---
def load_gritlm_model_safe(model_path, **kwargs):
    from gritlm import GritLM
    try:
        return GritLM(model_path, **kwargs)
    except TypeError as e:
        msg = str(e)
        if 'dtype' in kwargs and 'unexpected keyword argument' in msg:
            alt = kwargs.copy()
            alt['torch_dtype'] = alt.pop('dtype')
            return GritLM(model_path, **alt)
        if 'torch_dtype' in kwargs and 'unexpected keyword argument' in msg:
            alt = kwargs.copy()
            alt['dtype'] = alt.pop('torch_dtype')
            return GritLM(model_path, **alt)
        raise


class GritLMRunner:
    def __init__(self, model_path):
        print("📍 [GritLM Runner] Instantiating GritLM with safe wrappers...")
        self.model = load_gritlm_model_safe(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            mode="embedding",
            attn_implementation="eager"
        )
        self.faiss_index = None  
        self.pubkeys = None

    def index(self, texts, pubkeys):
        self.pubkeys = pubkeys
        cache_file = os.path.join(CACHE_DIR, "corpus_gritlm.npy")
        if os.path.exists(cache_file):
            print("📍 [GritLM Runner] Cache found. Direct-loading GritLM numpy array...")
            embs = np.load(cache_file)
        else:
            print("📍 [GritLM Runner] Cache absent. Encoding GritLM document corpus...")
            embs = self.model.encode(texts, batch_size=128, instruction="<|embed|>\n", show_progress_bar=True)
            embs = np.array(embs, dtype=np.float32)
            np.save(cache_file, embs)
        
        dim = embs.shape[1]
        
        # Load FAISS index with automatic GPU-to-CPU downgrade on missing dependencies
        try:
            res = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            print("📍 [GritLM Runner] Initialized FAISS GPU index successfully")
        except (AttributeError, Exception) as e:
            self.faiss_index = faiss.IndexFlatIP(dim)
            print(f"📍 [GritLM Runner] FAISS GPU unavailable. Defaulted to IndexFlatIP CPU index. Info: {e}")
            
        self.faiss_index.add(embs.astype('float32'))
        print("✅ GritLM indexing completed successfully.")

    def retrieve(self, queries, top_k=100):
        instruction = "Given a scientific claim from a social media post, retrieve the source paper"
        formatted = [f"icie\n{instruction}\n<|embed|>\n{q}" for q in queries]
        q_embs = self.model.encode(formatted, batch_size=128)
        scores, indices = self.faiss_index.search(q_embs.astype('float32'), top_k)
        return [[self.pubkeys[idx] for idx in row] for row in indices]


# ==============================================================================
# 5. Main Retrieval and Scoring Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["bm25", "gtr", "e5", "gritlm"])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("📍 [Step 1] Loading document corpus data...")
    t_start = time.time()
    
    corpus_texts, corpus_pubkeys = load_corpus()
    print(f"📍 [Step 2] Corpus load successfully in {time.time() - t_start:.2f} seconds. Initializing model configuration...")

    # Initialize retriever
    if args.model == "bm25":
        runner = BM25Runner()
    elif args.model == "gtr":
        resolved_path = get_model_path_or_id("gtr")
        runner = DenseRunner("gtr", resolved_path)
    elif args.model == "e5":
        resolved_path = get_model_path_or_id("e5")
        runner = DenseRunner("e5", resolved_path)
    elif args.model == "gritlm":
        resolved_path = get_model_path_or_id("gritlm")
        runner = GritLMRunner(resolved_path)
    
    runner.index(corpus_texts, corpus_pubkeys)

    # Perform retrieval evaluation loops
    results = {}  # {lang: {cond: score}}
    langs = ["en", "de", "fr"]
    conditions = [0, 1, 2, 3, 4]

    for lang in langs:
        results[lang] = {}
        for cond in conditions:
            scores_for_versions = []
            versions = [1] if cond == 0 else [1, 2, 3]
            
            for v in versions:
                print(f"🔍 Evaluating: {args.model} | {lang} | C{cond} | V{v}")
                query_texts, target_keys = load_queries(lang, cond, v)
                
                if not query_texts:
                    continue
                
                retrieved_keys_list = runner.retrieve(query_texts, top_k=5)
                
                # Compute batch MRR@5
                batch_mrr = []
                for target, retrieved in zip(target_keys, retrieved_keys_list):
                    batch_mrr.append(calculate_mrr(target, retrieved, k=5))
                
                scores_for_versions.append(np.mean(batch_mrr))
            
            # C1-C4 score is the average of version V1-V3
            results[lang][cond] = np.mean(scores_for_versions) if scores_for_versions else 0.0

    # Save output summary results to final results directory
    output_json = os.path.join(OUTPUT_DIR, f"mrr_{args.model}.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ MRR Evaluation finished. Results written to: {output_json}")


if __name__ == "__main__":
    main()
