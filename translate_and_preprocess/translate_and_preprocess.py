#!/usr/bin/env python3
"""Combined translation and preprocessing tools.

Subcommands:
  - clean-corpus
  - translate-corpus-de-fr
  - translate-de-queries
  - translate-queries
"""

import argparse
import json
import os
import shutil
import sys


SYSTEM_PROMPT_GENERIC = (
    "You are a professional translator. "
    "Translate the following text to English. "
    "Output ONLY the English translation - nothing else. "
    "Do not add explanations, notes, or quotes."
)

SYSTEM_PROMPT_DE = (
    "You are an expert scientific translator working from German into English. "
    "Your translations will be used to search for academic papers, so every detail matters. "
    "Follow these rules strictly:\n"
    "1. Preserve all named entities exactly (genes, diseases, drugs, methods, institutions, percentages).\n"
    "2. Never add or remove any scientific facts - only translate the language.\n"
    "3. Keep the informal tone of social media, but ensure medical/scientific terms are translated "
    "precisely (e.g., \"Herzinsuffizienz\" -> \"heart failure\", not \"cardiac insufficiency\").\n"
    "4. If a term has no direct English equivalent, keep the original German and add an explanation "
    "in brackets.\n"
    "5. Output ONLY the English translation. No extra text, no commentary, no quotation marks.\n"
    "\n"
    "Examples:\n"
    "User: \"Neue Studie zeigt: COVID-19-Impfung reduziert das Risiko fuer Long-COVID um 50 % bei "
    "Geimpften.\"\n"
    "Assistant: New study shows: COVID-19 vaccination reduces the risk of Long COVID by 50% in "
    "vaccinated individuals.\n"
    "\n"
    "User: \"Die Arbeitsgruppe um Prof. Dr. Mueller von der Charite Berlin hat in einer "
    "Kohortenstudie mit 10.000 Patienten nachgewiesen, dass Ivermectin keinen Nutzen bei "
    "ambulanten COVID-19-Patienten hat.\"\n"
    "Assistant: The research group led by Prof. Dr. Mueller from Charite Berlin demonstrated in a "
    "cohort study with 10,000 patients that ivermectin has no benefit in outpatient COVID-19 "
    "patients.\n"
    "\n"
    "User: \"SARS-CoV-2 kann ueber Aerosole uebertragen werden - das belegt eine aktuelle "
    "Untersuchung des RKI.\"\n"
    "Assistant: SARS-CoV-2 can be transmitted via aerosols - this is shown by a recent "
    "investigation of the Robert Koch Institute (RKI).\n"
    "\n"
    "Now translate this German tweet:"
)


def get_corpus_system_prompt(source_lang_name):
    return (
        f"You are a professional scientific translator working from {source_lang_name} into English. "
        "Your translation will be used for academic search, so accuracy of terminology is crucial. "
        "Rules:\n"
        "1. Translate ALL {source_lang_name} text into natural, fluent English.\n"
        "2. Do not change any scientific facts, numbers, units, variable names, or equations.\n"
        "3. Preserve named entities (university names, city names, person names) exactly as given.\n"
        "4. If a sentence is already partly in English, translate only the non-English parts.\n"
        "5. Output ONLY the English translation. No extra text, no quotes, no commentary.\n"
        "\n"
        "Now translate the following scientific text from {source_lang_name} to English:"
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def copy_json(src_path, dst_path):
    if not os.path.isfile(src_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"  Copied {src_path} -> {dst_path}")


def clean_authors(author_str, first_n=3, last_n=3):
    if not author_str:
        return ""
    entries = author_str.split(";")
    names = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        comma_idx = entry.find(",")
        name = entry[:comma_idx].strip() if comma_idx != -1 else entry
        if name:
            names.append(name)
    if len(names) <= first_n + last_n:
        return ", ".join(names)
    return ", ".join(names[:first_n] + [n for n in names[-last_n:] if n not in names[:first_n]])


def run_clean_corpus(args):
    data = load_json(args.collection_path)
    for doc in data:
        doc["authors"] = clean_authors(doc.get("authors", ""), args.first_n, args.last_n)
    save_json(data, args.collection_path)
    print(f"Cleaned authors in {args.collection_path}")


def load_translation_model(model_name, device, use_fast=None, attn_implementation="eager"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs = {
        "trust_remote_code": True,
        "padding_side": "left",
    }
    if use_fast is not None:
        tokenizer_kwargs["use_fast"] = use_fast

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return tokenizer, model


def translate_batch(
    texts,
    tokenizer,
    model,
    device,
    system_prompt,
    max_new_tokens=512,
    max_input_length=2048,
    temperature=0.7,
    top_p=0.8,
):
    messages_list = []
    for text in texts:
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ])

    applied = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for msgs in messages_list
    ]

    inputs = tokenizer(
        applied,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    import torch

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    translations = []
    for output_ids in outputs:
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[input_len:]
        translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        translations.append(translation)

    return translations


def translate_queries_file(src_path, dst_path, system_prompt, tokenizer, model, device, batch_size):
    if not os.path.isfile(src_path):
        print(f"  Skipping: {src_path} not found")
        return

    data = load_json(src_path)
    print(f"  {os.path.basename(src_path)}: {len(data)} queries to translate")

    texts = [q["text"] for q in data]
    translated = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_translated = translate_batch(
            batch,
            tokenizer,
            model,
            device,
            system_prompt=system_prompt,
            max_new_tokens=512,
            max_input_length=2048,
            temperature=0.7,
            top_p=0.8,
        )
        translated.extend(batch_translated)

        done = min(i + batch_size, len(texts))
        if done % max(1, len(texts) // 10) < batch_size or done == len(texts):
            print(f"    Translated {done}/{len(texts)}")

    for q, t in zip(data, translated):
        q["text"] = t

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    save_json(data, dst_path)
    print(f"  Saved to {dst_path}")


def run_translate_de_queries(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading translation model: {args.model_name}")
    tokenizer, model = load_translation_model(args.model_name, device)
    print("Model loaded.\n")

    print("Copying collection data...")
    copy_json(
        os.path.join(args.src_dataset_dir, "collection_data.json"),
        os.path.join(args.dst_dataset_dir, "collection_data.json"),
    )

    print("\nCopying English files...")
    for split in args.splits:
        copy_json(
            os.path.join(args.src_dataset_dir, f"en_{split}.json"),
            os.path.join(args.dst_dataset_dir, f"en_{split}.json"),
        )

    print("\nTranslating DE...")
    for split in args.splits:
        src = os.path.join(args.src_dataset_dir, f"de_{split}.json")
        dst = os.path.join(args.dst_dataset_dir, f"de_{split}.json")
        translate_queries_file(
            src,
            dst,
            system_prompt=SYSTEM_PROMPT_DE,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )

    print("\nTranslation complete.")


def resolve_test_dirs(args):
    if args.src_test_dir:
        src_test_dir = args.src_test_dir
    else:
        candidate = args.src_dataset_dir
        probe = os.path.join(candidate, f"{args.test_prefix}de_test.json")
        src_test_dir = candidate if os.path.isfile(probe) else os.path.join(candidate, "test_files")

    if args.dst_test_dir:
        dst_test_dir = args.dst_test_dir
    else:
        if src_test_dir == args.src_dataset_dir:
            dst_test_dir = args.dst_dataset_dir
        else:
            dst_test_dir = os.path.join(args.dst_dataset_dir, "test_files")

    return src_test_dir, dst_test_dir


def run_translate_queries(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading translation model: {args.model_name}")
    tokenizer, model = load_translation_model(args.model_name, device)
    print("Model loaded.\n")

    print("Copying collection data...")
    copy_json(
        os.path.join(args.src_dataset_dir, "collection_data.json"),
        os.path.join(args.dst_dataset_dir, "collection_data.json"),
    )

    print("\nCopying English files...")
    for split in args.splits:
        copy_json(
            os.path.join(args.src_dataset_dir, f"en_{split}.json"),
            os.path.join(args.dst_dataset_dir, f"en_{split}.json"),
        )

    for lang in ["de", "fr"]:
        print(f"\nTranslating {lang.upper()}...")
        for split in args.splits:
            src = os.path.join(args.src_dataset_dir, f"{lang}_{split}.json")
            dst = os.path.join(args.dst_dataset_dir, f"{lang}_{split}.json")
            translate_queries_file(
                src,
                dst,
                system_prompt=SYSTEM_PROMPT_GENERIC,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )

    if args.process_test:
        print("\n" + "=" * 60)
        print("Processing test files...")
        print("=" * 60)

        src_test_dir, dst_test_dir = resolve_test_dirs(args)

        en_test_file = f"{args.test_prefix}en_test.json"
        copy_json(
            os.path.join(src_test_dir, en_test_file),
            os.path.join(dst_test_dir, en_test_file),
        )

        for lang in ["de", "fr"]:
            test_file = f"{args.test_prefix}{lang}_test.json"
            src = os.path.join(src_test_dir, test_file)
            dst = os.path.join(dst_test_dir, test_file)
            print(f"\nTranslating {lang.upper()} test file...")
            translate_queries_file(
                src,
                dst,
                system_prompt=SYSTEM_PROMPT_GENERIC,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )

    print("\nTranslation complete.")


def translate_corpus_documents(corpus, indices, tokenizer, model, device, source_lang_name, batch_size):
    from tqdm import tqdm

    texts_to_translate = []
    for i in indices:
        doc = corpus[i]
        src_text = f"Title: {doc.get('title','')}\nAbstract: {doc.get('abstract','')}"
        texts_to_translate.append(src_text)

    translated = []
    for start in tqdm(range(0, len(texts_to_translate), batch_size), desc=f"Translating {source_lang_name}"):
        batch = texts_to_translate[start : start + batch_size]
        batch_trans = translate_batch(
            batch,
            tokenizer,
            model,
            device,
            system_prompt=get_corpus_system_prompt(source_lang_name),
            max_new_tokens=1024,
            max_input_length=1536,
            temperature=0.3,
            top_p=0.9,
        )
        translated.extend(batch_trans)

    for idx, trans in zip(indices, translated):
        doc = corpus[idx]
        doc["original_title"] = doc.get("title", "")
        doc["original_abstract"] = doc.get("abstract", "")

        lines = trans.strip().split("\n")
        if len(lines) >= 2 and lines[0].lower().startswith("title:"):
            new_title = lines[0].replace("Title:", "", 1).strip()
            new_abstract = " ".join(lines[1:]).replace("Abstract:", "", 1).strip()
        else:
            new_title = doc.get("title", "")
            new_abstract = trans.strip()

        doc["title"] = new_title
        doc["abstract"] = new_abstract

        venue = doc.get("venue", "")
        authors = doc.get("authors", "")
        doc["doc_text"] = f"{new_title} {new_abstract} Venue: {venue} Authors: {authors}"


def run_translate_corpus_de_fr(args):
    import torch
    from langdetect import detect, DetectorFactory
    from tqdm import tqdm

    DetectorFactory.seed = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    tokenizer, model = load_translation_model(
        args.model_name,
        device,
        use_fast=False,
        attn_implementation="eager",
    )

    print(f"Loading corpus from {args.src_corpus}")
    corpus = load_json(args.src_corpus)
    print(f"Total documents: {len(corpus)}")

    def detect_language(text):
        if not text or len(text.strip()) < 20:
            return None
        try:
            return detect(text)
        except Exception:
            return None

    de_indices = []
    fr_indices = []
    for i, doc in enumerate(tqdm(corpus, desc="Detecting languages")):
        text_for_detection = f"{doc.get('title','')} {doc.get('abstract','')}"
        lang = detect_language(text_for_detection)
        if lang == "de":
            de_indices.append(i)
        elif lang == "fr":
            fr_indices.append(i)

    print(f"German documents: {len(de_indices)}")
    print(f"French documents: {len(fr_indices)}")

    if de_indices:
        translate_corpus_documents(
            corpus,
            de_indices,
            tokenizer,
            model,
            device,
            source_lang_name="German",
            batch_size=args.batch_size,
        )
    else:
        print("No German documents to translate.")

    if fr_indices:
        translate_corpus_documents(
            corpus,
            fr_indices,
            tokenizer,
            model,
            device,
            source_lang_name="French",
            batch_size=args.batch_size,
        )
    else:
        print("No French documents to translate.")

    save_json(corpus, args.dst_corpus)
    print(f"Translated corpus saved to {args.dst_corpus}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Translation and preprocessing utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_clean = subparsers.add_parser("clean-corpus", help="Clean authors in collection_data.json")
    p_clean.add_argument("collection_path", type=str, help="Path to collection_data.json")
    p_clean.add_argument("--first_n", type=int, default=3, help="Keep first N authors")
    p_clean.add_argument("--last_n", type=int, default=3, help="Keep last N authors")
    p_clean.set_defaults(func=run_clean_corpus)

    p_corpus = subparsers.add_parser(
        "translate-corpus-de-fr",
        help="Translate German and French corpus docs to English",
    )
    p_corpus.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p_corpus.add_argument("--src_corpus", type=str, required=True)
    p_corpus.add_argument("--dst_corpus", type=str, required=True)
    p_corpus.add_argument("--batch_size", type=int, default=4)
    p_corpus.set_defaults(func=run_translate_corpus_de_fr)

    p_de = subparsers.add_parser(
        "translate-de-queries",
        help="Translate German queries to English",
    )
    p_de.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p_de.add_argument("--src_dataset_dir", type=str, default="../Dataset")
    p_de.add_argument("--dst_dataset_dir", type=str, default="../Dataset_translated")
    p_de.add_argument("--batch_size", type=int, default=8)
    p_de.add_argument("--splits", nargs="+", default=["dev", "train"])
    p_de.set_defaults(func=run_translate_de_queries)

    p_all = subparsers.add_parser(
        "translate-queries",
        help="Translate German and French queries to English",
    )
    p_all.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p_all.add_argument("--src_dataset_dir", type=str, default="../Dataset")
    p_all.add_argument("--dst_dataset_dir", type=str, default="../Dataset_translated")
    p_all.add_argument("--batch_size", type=int, default=8)
    p_all.add_argument("--splits", nargs="+", default=["dev", "train"])
    p_all.add_argument("--process_test", action="store_true")
    p_all.add_argument("--src_test_dir", type=str, default=None)
    p_all.add_argument("--dst_test_dir", type=str, default=None)
    p_all.add_argument("--test_prefix", type=str, default="final_")
    p_all.set_defaults(func=run_translate_queries)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
