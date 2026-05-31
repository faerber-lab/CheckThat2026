Attribution-Based Reranking via Natural Language Inference (NLI)This directory implements the second-stage reranking module for claim verification and scientific literature retrieval. Based on the top-10 candidate papers retrieved from the first-stage models (GritLM-Top100 and Qwen8B-Reranker), this module leverages Natural Language Inference (NLI) to compute an objective attribution alignment metric, significantly enhancing retrieval and ranking precision.🛠️ Pipeline Overview[Query Input] ➔ [Atomic Fact Decomposition] (LLM as Fact Extractor)
                      │
                      ▼
               [Atomic Facts] ──┐
                                ├─► [NLI Support Rate] ➔ [Score Fusion Strategies] ➔ [Evaluate MRR@5]
            [Top-10 Papers] ────┘   (YES/NO Entailment)
Atomic Fact Decomposition: Breaks down high-context, unstructured queries (such as scientific tweets) into multiple independent, stand-alone factual sentences with all pronouns resolved to explicit entities.NLI Verification: Pairs each decomposed atomic fact with the candidate paper's title and abstract. It utilizes an LLM as an NLI reasoning engine to determine if the paper content "entails" the fact, outputting YES (supported) or NO (not supported/irrelevant).Attribution Score Generation: Calculates $S_{\text{nli\_supp}}$ (NLI Support Rate), representing the ratio of verified facts.Fusion Reranking: Merges the first-stage scores ($S_{\text{qwen}}$) with the NLI verification support rates ($S_{\text{nli\_supp}}$) using various fusion strategies.Evaluation: Evaluates and outputs the Mean Reciprocal Rank (MRR@5) for each fusion strategy to compare the reranking performance.🧮 Score Fusion StrategiesWe evaluate the following formulas to combine the original Qwen score $S_{\text{qwen}}$ and the NLI support rate $S_{\text{nli\_supp}}$:Baseline:$$S_{\text{baseline}} = S_{\text{qwen}}$$NLI-Only:$$S_{\text{nli}} = S_{\text{nli\_supp}}$$Additive Fusion (Plus):$$S_{\text{plus}} = S_{\text{qwen}} + S_{\text{nli\_supp}}$$Multiplicative Fusion (Multi):$$S_{\text{multi}} = S_{\text{qwen}} \times S_{\text{nli\_supp}}$$Weighted Increment (Weight):$$S_{\text{weight}} = S_{\text{qwen}} \times (1.0 + S_{\text{nli\_supp}})$$Convex Combination / Linear Interpolation (Alpha):$$S_{\text{alpha}} = \alpha \cdot S_{\text{qwen}} + (1.0 - \alpha) \cdot S_{\text{nli\_supp}}$$(where default $\alpha = 0.5$)📂 Input SpecificationsThe reranker expects candidate files to be located under the ./out/qwen_rerank/ directory with the naming format qwen_top10_{lang}_{split}.json.File Schema Example:
[
  {
    "query_id": 0,
    "query_text": "Is high-dose vitamin C effective against influenza?",
    "true_pubkey": "paper_key_99",
    "candidates": [
      {
        "pubkey": "paper_key_99",
        "qwen_score": 0.942,
        "title": "Efficacy of Ascorbic Acid in Respiratory Viral Models",
        "abstract": "We studied clinical impact of high concentration vitamin C..."
      }
    ]
  }
]
(If the input files are not detected, process_data.py will automatically generate simulated Mock data to ensure the main program runs and tests without errors)🚀 Execution & Usage1. Install DependenciesEnsure you have the required Python libraries installed in your environment:pip install pandas numpy openai
(Optional: To run offline local inference using GPU acceleration, please install vllm)2. Configure Environment VariablesBefore running the scripts, configure your LLM API keys and endpoint properties:export LLM_API_KEY="your_api_key"
export OPENAI_API_BASE="[https://api.yourllmprovider.com/v1](https://api.yourllmprovider.com/v1)"
export OPENAI_MODEL_NAME="moonshotai/Kimi-K2.6"
3. Launch Reranking PipelineRun using the Online API Backend:python main.py --backend api --split dev --batch-size 32
Run using the Offline Local vLLM Engine:python main.py --backend vllm --split dev
4. Review Experimental ResultsOnce reranking finishes, the system will output a comparative performance table (MRR@5) in the terminal. Detailed inference logs and reranked JSON predictions are saved to:./out/rerank_logs/attribution_rerank_{lang}_{split}.json