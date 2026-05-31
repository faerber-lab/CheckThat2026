"""
Fine-tune GritLM-7B with LoRA on a prepared JSONL dataset.

Merged models (base + LoRA) are saved after each epoch via SaveMergedModelCallback.
The final epoch's merged model is the ready-to-use output.

Usage:
    python train.py \
        --data_path ./gritlm_train_en.jsonl \
        --output_dir ./output/gritlm_en \
        --model_name GritLM/GritLM-7B \
        --num_train_epochs 4
"""
import sys
import os
import logging
import argparse

import datasets
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Trainer, TrainerCallback, set_seed

# Local GritLM imports – the gritlm pip package or a local clone.
# If using a local clone placed next to this script, uncomment the next line:
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gritlm"))
from gritlm.training.arguments import CustomTrainingArguments, DataArguments
from gritlm.training.data import CustomCollator, CustomDataset
from gritlm.training.model import GritLMTrainModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolve model path: check HF cache before downloading
# ---------------------------------------------------------------------------
def _resolve_model_path(model_name):
    """Use local snapshot if cached, otherwise return the HF repo id."""
    if os.path.isdir(model_name):
        print(f"  Using local model path: {model_name}")
        return model_name

    # Try huggingface_hub's cache scanner
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_name:
                for rev in repo.revisions:
                    sp = str(rev.snapshot_path)
                    if os.path.isfile(os.path.join(sp, "config.json")):
                        print(f"  Found cached model: {sp}")
                        return sp
    except Exception:
        pass

    # Fallback: check common HF cache dirs manually
    repo_folder = "models--" + model_name.replace("/", "--")
    for cache_base in [
        os.environ.get("HF_HOME", ""),
        os.environ.get("TRANSFORMERS_CACHE", ""),
        os.path.expanduser("~/.cache/huggingface"),
    ]:
        if not cache_base:
            continue
        snapshots_dir = os.path.join(cache_base, "hub", repo_folder, "snapshots")
        if os.path.isdir(snapshots_dir):
            versions = sorted(os.listdir(snapshots_dir))
            if versions:
                sp = os.path.join(snapshots_dir, versions[-1])
                if os.path.isfile(os.path.join(sp, "config.json")):
                    print(f"  Found cached model: {sp}")
                    return sp

    print(f"  No cached model found, will download: {model_name}")
    return model_name


# ---------------------------------------------------------------------------
# Callback: merge LoRA → save full model → unmerge (after each epoch)
# ---------------------------------------------------------------------------
class SaveMergedModelCallback(TrainerCallback):
    """After each epoch: merge LoRA -> save clean merged state dict -> unmerge."""

    def __init__(self, gritlm_model, output_dir, tokenizer, base_model_path):
        self.gritlm_model = gritlm_model
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self._epoch = 0
        self.base_model_path = base_model_path

    def on_epoch_end(self, args, state, control, **kwargs):
        self._epoch += 1
        epoch_dir = os.path.join(self.output_dir, f"merged_epoch_{self._epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        logger.info(
            f"Epoch {self._epoch}: merging LoRA weights and saving to {epoch_dir} ..."
        )

        peft_model = self.gritlm_model.model

        # 1. Merge adapter weights into the base weights in-place
        peft_model.merge_adapter()

        # 2. Retrieve the state dict and clean the keys
        state_dict = peft_model.state_dict()
        clean_state_dict = {}

        for key, value in state_dict.items():
            # Skip LoRA-specific adapter weights (A/B matrices)
            if "lora_A" in key or "lora_B" in key or "lora_magnitude_vector" in key:
                continue

            new_key = key

            # Remove the PeftModel wrapper prefix
            prefix = "base_model.model."
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

            # Remove the '.base_layer' wrapper in the key name
            if ".base_layer." in new_key:
                new_key = new_key.replace(".base_layer.", ".")

            clean_state_dict[new_key] = value

        # 3. Save the cleaned state dict (with metadata for transformers compatibility)
        safetensors_path = os.path.join(epoch_dir, "model.safetensors")
        save_file(clean_state_dict, safetensors_path, metadata={"format": "pt"})

        # 4. Save config: copy the BASE model's config.json to preserve auto_map
        #    (PEFT's config.save_pretrained strips auto_map, breaking custom code loading)
        import shutil
        base_config = os.path.join(self.base_model_path, "config.json")
        if os.path.isfile(base_config):
            shutil.copy2(base_config, os.path.join(epoch_dir, "config.json"))
            logger.info(f"  Copied config.json (with auto_map) to {epoch_dir}")
        else:
            peft_model.base_model.model.config.save_pretrained(epoch_dir)

        # 5. Generate model.safetensors.index.json (required for device_map="auto")
        safetensors_size = os.path.getsize(safetensors_path)
        weight_map = {k: "model.safetensors" for k in clean_state_dict.keys()}
        index_content = {
            "metadata": {"total_size": safetensors_size},
            "weight_map": weight_map,
        }
        with open(os.path.join(epoch_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index_content, f, indent=2)
        logger.info(f"  Generated model.safetensors.index.json")

        # 5. Save tokenizer
        self.tokenizer.save_pretrained(epoch_dir)

        # 6. Copy custom modeling code files from base model (required for loading)
        for fname in os.listdir(self.base_model_path):
            if fname.endswith(".py"):
                src = os.path.join(self.base_model_path, fname)
                dst = os.path.join(epoch_dir, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    logger.info(f"  Copied {fname} to {epoch_dir}")

        # 7. Unmerge to restore original state for continued training
        peft_model.unmerge_adapter()
        logger.info(f"✓ Epoch {self._epoch} merged model saved to {epoch_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune GritLM with LoRA")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the JSONL training data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and merged models")
    parser.add_argument("--model_name", type=str, default="GritLM/GritLM-7B",
                        help="HuggingFace model name or local path")
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--query_max_len", type=int, default=256)
    parser.add_argument("--passage_max_len", type=int, default=256)
    parser.add_argument("--train_group_size", type=int, default=9,
                        help="1 pos + num_hard_neg + num_random_neg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Resolve model path: check HF cache before downloading
    model_path = _resolve_model_path(args.model_name)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if not tokenizer.pad_token and tokenizer.bos_token:
        tokenizer.pad_token = tokenizer.bos_token

    # Model
    logger.info("Loading GritLM model...")
    model = GritLMTrainModel(
        model_name_or_path=model_path,
        normalized=True,
        pooling_method="mean",
        negatives_cross_device=False,
        temperature=args.temperature,
        mode="embedding",
        attn="bbcc",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        low_cpu_mem_usage=True,
    )

    # LoRA
    from peft import get_peft_model, LoraConfig

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "o_proj", "v_proj", "k_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
    )
    model.model.enable_input_require_grads()
    model.model = get_peft_model(model.model, peft_config)
    model.model.print_trainable_parameters()

    # Data arguments
    data_args = DataArguments(
        train_data=args.data_path,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        train_group_size=args.train_group_size,
    )

    # Training arguments
    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        mode="embedding",
        temperature=args.temperature,
        negatives_cross_device=False,
        lora=True,
        torch_compile=False,
        dataloader_num_workers=0,
    )

    # Dataset & collator
    train_dataset = datasets.load_dataset(
        "json", data_files=args.data_path, split="train"
    )
    full_bs = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    train_dataset_obj = CustomDataset(
        dataset=train_dataset,
        args=data_args,
        tokenizer=tokenizer,
        mode="embedding",
        full_bs=full_bs,
    )

    data_collator = CustomCollator(tokenizer=tokenizer)
    data_collator.query_max_len = data_args.query_max_len
    data_collator.passage_max_len = data_args.passage_max_len

    # Callback
    save_merged_cb = SaveMergedModelCallback(
        gritlm_model=model,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        base_model_path=model_path,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_obj,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[save_merged_cb],
    )

    # Train
    logger.info("Starting training...")
    logger.info(
        f"Config: {training_args.num_train_epochs} epochs | "
        f"per_device_bs={training_args.per_device_train_batch_size} | "
        f"grad_accum={training_args.gradient_accumulation_steps} | "
        f"group_size={data_args.train_group_size} | "
        f"lr={training_args.learning_rate}"
    )
    trainer.train()

    print(f"\n✓ Training complete. Merged models saved to {args.output_dir}/merged_epoch_{{1..{args.num_train_epochs}}}")


if __name__ == "__main__":
    main()
