import pandas as pd
import os
import json
from pandas import DataFrame
from config import get_query_prompt, get_corpus_prompt

# ==============================================================================
# 1. Path Resolutions (Dynamically relative to project root)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to the sibling "data" directory relative to this script's parent folder
DATASET_PATH = os.getenv("DATASET_PATH", os.path.abspath(os.path.join(CURRENT_DIR, "..", "sample_data")))

def load_corpus() -> tuple[list[str], list[int]]:
    """
    Loads and processes the document corpus by combining title, abstract, venue, and authors.
    :return: A tuple of concatenated text list and the corresponding pubkeys list.
    """
    corpus_path = os.path.join(DATASET_PATH, 'collection_data_process.json')
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Cannot find corpus at: {corpus_path}")
        
    df_collection = pd.read_json(corpus_path)
    
    def combine_fields(row):
        title = str(row.get('title', '')).strip()
        abstract = str(row.get('abstract', '')).strip()
        venue = str(row.get('venue', '')).strip()
        authors = str(row.get('authors', '')).strip()
        return f"{title} {abstract} {venue} {authors}".strip()

    corpus = df_collection.apply(combine_fields, axis=1).tolist()
    pubkeys = df_collection['pubkey'].tolist()
    
    return corpus, pubkeys


def load_queries(lang: str = 'en', split: str = 'dev', k: int = -1, t: int = 0) -> pd.DataFrame:
    """
    Loads queries from the raw JSON file (e.g. en_dev.json).
    :param lang: Language code ('en', 'de', 'fr').
    :param split: Dataset split ('dev', 'train').
    :param k: Number of samples to load (-1 to load all).
    :param t: Number of test samples to exclude for experimental splits.
    :return: Queries represented as a pandas DataFrame.
    """
    file_name = f"{lang}_{split}.json"
    query_path = os.path.join(DATASET_PATH, file_name)

    if not os.path.exists(query_path):
        print(f"Error: {query_path} not found.")
        return pd.DataFrame()

    df_query: DataFrame = pd.read_json(query_path)

    # Exclude t test samples and save them for model selection verification
    if t > 0:
        df_test_samples = df_query.sample(n=t, random_state=42)
        df_query = df_query.drop(df_test_samples.index)

        # Save test samples for subsequent load_test_queries
        test_file_path = os.path.join(DATASET_PATH, f'test_samples_{lang}_{split}.csv')
        if not os.path.exists(test_file_path):
            df_test_samples.to_csv(test_file_path, index=False)

    # If k is specified, randomly select k samples
    if k > 0:
        k = min(k, len(df_query))
        df_query = df_query.sample(n=k, random_state=42)

    return df_query


def load_test_queries(lang: str = 'en', split: str = 'dev') -> pd.DataFrame:
    """
    Loads the previously excluded test queries.
    """
    test_file_path = os.path.join(DATASET_PATH, f'test_samples_{lang}_{split}.csv')

    if os.path.exists(test_file_path):
        return pd.read_csv(test_file_path)
    else:
        print(f"Test queries file {test_file_path} does not exist.")
        return pd.DataFrame()


def load_synthetic_queries(lang: str = 'en', version: int = 1, k: int = -1, llm: str = 'qwen9B') -> pd.DataFrame:
    """
    Loads LLM-generated style-transfer queries.
    :param lang: Language code.
    :param version: Evaluation run version.
    :param k: Number of samples to load.
    :param llm: LLM model name.
    :return: DataFrame containing synthetic queries.
    """
    # Dynamically resolve synthetic directory in workspace root
    path = os.path.join(os.path.dirname(CURRENT_DIR), "out", "synthetic_queries", llm)
    file_name = f"synthetic_queries_{lang}_C{get_query_prompt()}_V{version}.json"
    full_path = os.path.join(path, file_name)
    
    if not os.path.exists(full_path):
        print(f"Synthetic queries file not found at: {full_path}")
        return pd.DataFrame()

    # Reading JSON format: [ {index, ori, query_transfer, pubkey}, ... ]
    with open(full_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    queries = pd.DataFrame(records)
    if k > 0:
        k = min(k, len(queries))
        queries = queries.iloc[:k]

    return queries


def load_claim_corpus() -> tuple[list[str], list[int]]:
    """
    Loads extracted claims from the corpus.
    """
    path = os.path.join(os.path.dirname(CURRENT_DIR), "out", "scientific_claims")
    file_path = os.path.join(path, f"scientific_claims_{get_corpus_prompt()}.csv")
    
    if not os.path.exists(file_path):
        print(f"Claim corpus file not found: {file_path}")
        return [], []
        
    df_collection = pd.read_csv(file_path)
    corpus = df_collection["claim"].tolist()
    pubkeys = df_collection["pubkey"].tolist()
    return corpus, pubkeys


def load_synthetic_abstracts() -> tuple[list[str], list[int]]:
    """
    Loads synthetic paper abstracts from output cache.
    """
    path = os.path.join(os.path.dirname(CURRENT_DIR), "out", "synthetic_abstracts")
    file_path = os.path.join(path, f"synthetic_abstracts_{get_corpus_prompt()}.csv")
    
    if not os.path.exists(file_path):
        print(f"Synthetic abstracts file not found: {file_path}")
        return [], []
        
    df_collection = pd.read_csv(file_path)
    corpus = df_collection["abstract"].tolist()
    pubkeys = df_collection["pubkey"].tolist()
    return corpus, pubkeys