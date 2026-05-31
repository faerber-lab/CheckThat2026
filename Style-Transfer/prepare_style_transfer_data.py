import argparse
import json
import os
from datetime import datetime
import llm_requests as llm_requests_vllm

# ==============================================================================
# Executable wrapper script to run query style transfer across multiple versions,
# styles, and target languages in parallel batch runs.
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True, help="v1, v2, or v3 evaluation run")
    parser.add_argument("--backend", choices=["api", "vllm"], default="vllm",
                        help="Inference backend (default: vllm)")
    parser.add_argument("--vllm-model", default=None,
                        help="vLLM model name (overrides default Qwen/Qwen3.5-9B)")
    parser.add_argument("--vllm-tp", type=int, default=None,
                        help="vLLM tensor parallel size (optional)")
    parser.add_argument("--split", choices=["dev", "train"], default="dev",
                        help="Dataset split (default: dev)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Number of items to process (-1 = all)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for parallel pipeline runs")
    args = parser.parse_args()

    languages = ['en', 'de', 'fr']  # Target evaluation languages
    style_prompts = [1, 2, 3, 4]    # Styles/Condition mappings
    version = args.version

    if args.vllm_model:
        llm_requests_vllm.VLLM_MODEL_NAME = args.vllm_model
    if args.vllm_tp is not None:
        llm_requests_vllm.VLLM_TENSOR_PARALLEL = args.vllm_tp

    print(f"🌟 Starting Generation Run - Version V{version}")
    print(f"   Backend: {args.backend}")
    print(f"   Split: {args.split}")
    print(f"   Total Tasks: {len(languages)} × {len(style_prompts)} = {len(languages) * len(style_prompts)}")
    print(f"   Total Target Files: {len(languages) * len(style_prompts)}\n")

    results = {
        "version": version,
        "backend": args.backend,
        "split": args.split,
        "timestamp": datetime.now().isoformat(),
        "tasks": []
    }
    completed = 0
    failed = 0

    for lang in languages:
        for prompt_id in style_prompts:
            task_idx = completed + failed + 1
            total_tasks = len(languages) * len(style_prompts)
            print(f"[{task_idx}/{total_tasks}] Processing: {lang} | C{prompt_id}")
            try:
                llm_requests_vllm.run_style_transfer_query(
                    lang=lang,
                    split=args.split,
                    version=version,
                    prompt_id=prompt_id,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    backend=args.backend
                )
                completed += 1
                print(f"Progress Success: {completed}/{total_tasks}")
            except Exception as e:
                failed += 1
                print(f"Error executing {lang}_C{prompt_id}_V{version}: {e}")

    
    print(f"\nExecution Summary completed!")
    print(f"   Success Tasks: {completed}")
    print(f"   Failed Tasks: {failed}")

if __name__ == "__main__":
    main()