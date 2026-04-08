from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm
from vllm import LLM, SamplingParams

MODEL_PATH = Path("/mnt/phwfile/medai_p/LLMModels/LLMs/Llama-3.1-8B-Instruct")
DEFAULT_DATASET = Path(os.path.expanduser("~/datasets/medical_train/medmcqa.json"))
DEFAULT_CACHE = Path(os.path.expanduser("~/datasets/medical_train/llama318b_greedy_medmcqa.jsonl"))
FILTER_OUTPUT = "medmcqa_llama31_filter.json"


def load_dataset(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_cache(path: Path) -> Dict[str, dict]:
    processed: Dict[str, dict] = {}
    if not path.exists():
        return processed
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            processed[entry["question_id"]] = entry
    return processed


def append_cache(path: Path, entries: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # ensure resume data persists immediately


def save_incorrect_questions(
    dataset: List[dict],
    cache: Dict[str, dict],
    dataset_path: Path,
    output_name: str = FILTER_OUTPUT,
) -> Path:
    incorrect_ids = {
        qid for qid, entry in cache.items() if entry and not entry.get("is_correct")
    }
    filtered_dataset = [item for item in dataset if item["id"] in incorrect_ids]
    output_path = dataset_path.with_name(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_dataset, f, ensure_ascii=False)
    print(
        f"Saved {len(filtered_dataset)} incorrect questions to "
        f"{output_path}."
    )
    return output_path


def build_prompt(question: str) -> str:
    return (
        "You are a medical expert that answers multiple-choice questions.\n"
        "Read the question and select the single best option letter without explanation.\n\n"
        f"{question.strip()}\n\n"
        "Final answer (just the letter and short answer):"
    )


def extract_choice(text: str) -> str | None:
    normalized = text.strip().upper()
    letter = re.search(r"\b([A-H])\b", normalized)
    if letter:
        return letter.group(1)
    letter = re.search(r"OPTION\\s*([A-H])", normalized)
    if letter:
        return letter.group(1)
    return None


def batched(seq: List[dict], size: int) -> Iterable[List[dict]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def dataset_progress(cache: Dict[str, dict], dataset_ids: List[str]) -> tuple[int, int]:
    processed = 0
    correct = 0
    for qid in dataset_ids:
        entry = cache.get(qid)
        if entry is None:
            continue
        processed += 1
        if entry.get("is_correct"):
            correct += 1
    return processed, correct


def run_inference(
    dataset_path: Path,
    cache_path: Path,
    batch_size: int,
    max_tokens: int,
    sample_limit: int | None = None,
) -> None:
    dataset = load_dataset(dataset_path)
    if sample_limit is not None:
        if sample_limit <= 0:
            print("Sample limit is non-positive; nothing to process.")
            return
        dataset = dataset[:sample_limit]
        print(f"Debug mode enabled: limiting inference to {len(dataset)} samples.")
    if not dataset:
        print("Dataset is empty after applying the sample limit.")
        return
    dataset_ids = [item["id"] for item in dataset]
    cache = load_cache(cache_path)
    pending = [item for item in dataset if item["id"] not in cache]
    total_processed, correct_processed = dataset_progress(cache, dataset_ids)

    if not pending:
        accuracy = (
            correct_processed / len(dataset) if dataset else 0.0
        )
        print(f"All {len(dataset)} questions already processed. Accuracy: {accuracy:.4f}")
        save_incorrect_questions(dataset, cache, dataset_path)
        return

    llm = LLM(model=str(MODEL_PATH), trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_seq_len_to_capture=3072,
            max_num_batched_tokens=4000,
            enable_prefix_caching=True,
            max_model_len=3072,
            enable_chunked_prefill=False,
            enable_sleep_mode=True,)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)

    new_entries: List[dict] = []
    total_batches = (len(pending) + batch_size - 1) // batch_size
    for batch in tqdm(
        batched(pending, batch_size),
        total=total_batches,
        desc="Running MedMCQA",
    ):
        prompts = [build_prompt(q["conversations"][0]["value"]) for q in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        batch_entries = []
        for sample, output in zip(batch, outputs, strict=True):
            response = output.outputs[0].text.strip()
            predicted_idx = extract_choice(response)
            ground_truth_idx = sample.get("answer_idx")
            ground_truth = sample["conversations"][1]["value"].strip()
            is_correct = predicted_idx == ground_truth_idx
            entry = {
                "question_id": sample["id"],
                "question": sample["conversations"][0]["value"],
                "ground_truth": ground_truth,
                "ground_truth_idx": ground_truth_idx,
                "prediction": response,
                "predicted_idx": predicted_idx,
                "is_correct": bool(is_correct),
            }
            cache[entry["question_id"]] = entry
            batch_entries.append(entry)
        append_cache(cache_path, batch_entries)
        total_processed += len(batch_entries)
        correct_processed += sum(1 for entry in batch_entries if entry["is_correct"])
        running_accuracy = (
            correct_processed / total_processed if total_processed else 0.0
        )
        print(
            f"Processed {total_processed}/{len(dataset)} questions | "
            f"running accuracy: {running_accuracy:.4f}"
        )
        new_entries.extend(batch_entries)

    final_accuracy = correct_processed / len(dataset)
    print(f"Finished. Accuracy over {len(dataset)} questions: {final_accuracy:.4f}")
    save_incorrect_questions(dataset, cache, dataset_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MedMCQA using vLLM.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of questions to process for debugging (e.g., 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        dataset_path=args.dataset_path,
        cache_path=args.cache_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        sample_limit=args.sample_limit,
    )

if __name__ == "__main__":
    main()
