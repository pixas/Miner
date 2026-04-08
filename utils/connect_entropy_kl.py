#!/usr/bin/env python3
"""
Compute token-wise KL statistics between a tuned policy and a reference policy.

Given a cache.jsonl containing trajectories, this script recomputes per-token log
probabilities/entropies with both models, aggregates the requested statistics, and
plots entropy histograms for the token groups of interest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.utils.prompt import get_prepare_func


@dataclass
class TokenizedSample:
    prompt_text: str
    response_text: str
    input_ids: list[int]
    prompt_len: int
    response_len: int
    is_correct: bool


def _best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _render_prompt(prompt_obj, tokenizer: AutoTokenizer) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, list):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(prompt_obj, tokenize=False, add_generation_prompt=True)
        pieces = []
        for turn in prompt_obj:
            role = turn.get("role", "").strip().title() or "User"
            pieces.append(f"{role}: {turn.get('content', '')}")
        pieces.append("Assistant:")
        return "\n".join(pieces)
    raise TypeError(f"Unsupported prompt type: {type(prompt_obj)}")


def _load_result_metadata(cache_path: Path, explicit_result: Optional[Path]) -> tuple[Optional[str], Optional[str]]:
    result_path = explicit_result
    if result_path is None:
        candidate = cache_path.with_name("result.json")
        if candidate.exists():
            result_path = candidate
    prompt_type = None
    data_source = None
    if result_path and result_path.exists():
        with open(result_path, "r") as f:
            payload = json.load(f)
        args_dict = payload.get("args", {})
        prompt_type = args_dict.get("prompt_type")
        data_path = args_dict.get("data_path")
        if data_path:
            data_source = Path(data_path).stem
    return prompt_type, data_source


def load_samples(
    cache_path: Path,
    tokenizer: AutoTokenizer,
    prompt_type: str,
    data_source: Optional[str],
    sample_limit: Optional[int],
    prompt_model_hint: str,
) -> list[TokenizedSample]:
    prepare_fn = get_prepare_func(prompt_model_hint, prompt_type, data_source)
    samples: list[TokenizedSample] = []
    prompt_cache: dict[str, list[int]] = {}

    with open(cache_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            task = obj.get("task", {})
            prompt_bundle = prepare_fn(task)
            prompt_raw = prompt_bundle.get("prompt", task.get("input", ""))
            prompt_text = _render_prompt(prompt_raw, tokenizer)
            if prompt_text not in prompt_cache:
                enc = tokenizer(prompt_text, add_special_tokens=False, return_attention_mask=False)
                ids = enc["input_ids"]
                if not ids:
                    special = tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id
                    if special is None:
                        raise ValueError("Tokenizer lacks special tokens to seed an empty prompt.")
                    ids = [special]
                prompt_cache[prompt_text] = ids
            prompt_ids = prompt_cache[prompt_text]

            responses: Iterable[str] = obj.get("trajectory") or obj.get("predictions") or []
            acc_list = obj.get("all_acc")
            if not responses or not acc_list:
                continue

            for response, is_correct in zip(responses, acc_list):
                if not response:
                    continue
                resp_enc = tokenizer(response, add_special_tokens=False, return_attention_mask=False)
                response_ids = resp_enc["input_ids"]
                if not response_ids:
                    continue
                full_ids = list(prompt_ids + response_ids)
                samples.append(
                    TokenizedSample(
                        prompt_text=prompt_text,
                        response_text=response,
                        input_ids=full_ids,
                        prompt_len=len(prompt_ids),
                        response_len=len(response_ids),
                        is_correct=bool(is_correct),
                    )
                )
                if sample_limit and len(samples) >= sample_limit:
                    return samples
    return samples


@torch.inference_mode()
def compute_batched_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[TokenizedSample],
    batch_size: int,
    desc: Optional[str] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not samples:
        return [], []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.bos_token_id or 0
    device = next(model.parameters()).device
    logprob_storage: list[Optional[np.ndarray]] = [None] * len(samples)
    entropy_storage: list[Optional[np.ndarray]] = [None] * len(samples)

    total_steps = max(1, (len(samples) + batch_size - 1) // batch_size)
    loop = range(0, len(samples), batch_size)
    loop = tqdm(loop, total=total_steps, desc=desc or "logprob pass")

    for start in loop:
        chunk = samples[start : start + batch_size]
        max_len = max(len(s.input_ids) for s in chunk)
        input_ids = torch.full(
            (len(chunk), max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(input_ids)
        for row, sample in enumerate(chunk):
            ids = torch.tensor(sample.input_ids, dtype=torch.long, device=device)
            input_ids[row, : ids.numel()] = ids
            attention[row, : ids.numel()] = 1

        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        shift_entropy = -(shift_logprobs.exp() * shift_logprobs).sum(dim=-1)

        for row, sample in enumerate(chunk):
            start_idx = max(0, sample.prompt_len - 1)
            end_idx = start_idx + sample.response_len
            if end_idx <= start_idx or start_idx >= shift_logprobs.shape[1]:
                logprob_storage[start + row] = np.empty(0, dtype=np.float32)
                entropy_storage[start + row] = np.empty(0, dtype=np.float32)
                continue
            end_idx = min(end_idx, shift_logprobs.shape[1])
            lp_slice = shift_logprobs[row, start_idx:end_idx, :]
            label_slice = shift_labels[row, start_idx:end_idx]
            resp_lp = lp_slice.gather(-1, label_slice.unsqueeze(-1)).squeeze(-1)
            logprob_storage[start + row] = resp_lp.float().detach().cpu().numpy()
            entropy_storage[start + row] = shift_entropy[row, start_idx:end_idx].float().detach().cpu().numpy()

    return logprob_storage, entropy_storage  # type: ignore[return-value]


def summarize_statistics(
    samples: list[TokenizedSample],
    tuned_logps: list[np.ndarray],
    ref_logps: list[np.ndarray],
    tuned_entropy: list[np.ndarray],
    ref_entropy: list[np.ndarray],
) -> dict[str, object]:
    total_tokens = 0
    kl_sum = 0.0
    correct_tokens = 0
    incorrect_tokens = 0
    correct_condition = 0
    incorrect_condition = 0
    correct_entropy_tuned: list[np.ndarray] = []
    correct_entropy_ref: list[np.ndarray] = []
    wrong_entropy_tuned: list[np.ndarray] = []
    wrong_entropy_ref: list[np.ndarray] = []

    for sample, lp_tuned, lp_ref, ent_tuned, ent_ref in zip(samples, tuned_logps, ref_logps, tuned_entropy, ref_entropy):
        limit = min(sample.response_len, lp_tuned.shape[0], lp_ref.shape[0])
        if limit <= 0:
            continue
        lp_tuned = lp_tuned[:limit]
        lp_ref = lp_ref[:limit]
        ent_tuned = ent_tuned[:limit]
        ent_ref = ent_ref[:limit]
        log_ratio = lp_ref - lp_tuned
        # log_ratio = lp_tuned - lp_ref
        log_ratio = np.clip(log_ratio, -20, 20)
        ratio = np.exp(log_ratio)
        kl = ratio - log_ratio - 1
        kl = np.clip(kl, -10, 10)
        kl_sum += float(kl.sum())
        total_tokens += limit

        if sample.is_correct:
            correct_tokens += limit
            mask = log_ratio < 0
            correct_condition += int(mask.sum())
            if mask.any():
                correct_entropy_tuned.append(ent_tuned[mask])
                correct_entropy_ref.append(ent_ref[mask])
        else:
            incorrect_tokens += limit
            
            mask = log_ratio > 0 | np.array(limit == 8192)
            incorrect_condition += int(mask.sum())
            if mask.any():
                wrong_entropy_tuned.append(ent_tuned[mask])
                wrong_entropy_ref.append(ent_ref[mask])

    stats = {
        "total_tokens": total_tokens,
        "mean_kl": kl_sum / max(total_tokens, 1),
        "correct_tokens": correct_tokens,
        "incorrect_tokens": incorrect_tokens,
        "correct_condition": correct_condition,
        "incorrect_condition": incorrect_condition,
        "fraction_correct": correct_condition / max(correct_tokens, 1),
        "fraction_incorrect": incorrect_condition / max(incorrect_tokens, 1),
        "correct_entropy_tuned": np.concatenate(correct_entropy_tuned) if correct_entropy_tuned else np.empty(0),
        "correct_entropy_ref": np.concatenate(correct_entropy_ref) if correct_entropy_ref else np.empty(0),
        "wrong_entropy_tuned": np.concatenate(wrong_entropy_tuned) if wrong_entropy_tuned else np.empty(0),
        "wrong_entropy_ref": np.concatenate(wrong_entropy_ref) if wrong_entropy_ref else np.empty(0),
    }
    return stats


def _plot_group(ax, ref_values: np.ndarray, tuned_values: np.ndarray, title: str) -> None:
    ax.set_title(title)
    if ref_values.size == 0 and tuned_values.size == 0:
        ax.text(0.5, 0.5, "No tokens match condition", ha="center", va="center")
        ax.set_xlabel("Entropy (nats)")
        ax.set_ylabel("Count")
        return
    max_val = 0.0
    if ref_values.size:
        max_val = max(max_val, float(ref_values.max()))
    if tuned_values.size:
        max_val = max(max_val, float(tuned_values.max()))
    max_val = max(max_val, 1e-3)
    bins = np.linspace(0.0, max_val, 40)
    if ref_values.size:
        ax.hist(ref_values, bins=bins, alpha=0.5, label="pi_ref entropy")
    if tuned_values.size:
        ax.hist(tuned_values, bins=bins, alpha=0.5, label="pi_tuned entropy")
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Count")
    ax.legend()


def plot_entropy_histograms(stats: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _plot_group(
        axes[0],
        stats["correct_entropy_ref"],  # type: ignore[arg-type]
        stats["correct_entropy_tuned"],  # type: ignore[arg-type]
        "Correct tokens (pi_tuned > pi_ref)",
    )
    _plot_group(
        axes[1],
        stats["wrong_entropy_ref"],  # type: ignore[arg-type]
        stats["wrong_entropy_tuned"],  # type: ignore[arg-type]
        "Incorrect tokens (pi_tuned < pi_ref)",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect entropy and KL gaps between tuned and reference policies.")
    parser.add_argument("--cache-path", required=True, type=Path, help="Path to cache.jsonl containing trajectories.")
    parser.add_argument("--tuned-model", required=True, type=str, help="Path or name of the tuned model (pi_tuned).")
    parser.add_argument("--ref-model", required=True, type=str, help="Path or name of the reference model (pi_ref).")
    parser.add_argument("--tokenizer", default=None, type=str, help="Optional tokenizer path (defaults to tuned model).")
    parser.add_argument("--prompt-type", default=None, type=str, help="Prompt type override (e.g., base_math).")
    parser.add_argument("--data-source", default=None, type=str, help="Dataset hint used for prompt construction.")
    parser.add_argument("--result-json", default=None, type=Path, help="Optional result.json to infer metadata from.")
    parser.add_argument("--batch-size", default=2, type=int, help="Batch size for log-prob computation.")
    parser.add_argument("--sample-limit", default=None, type=int, help="Optional limit on number of trajectories.")
    parser.add_argument("--figure-path", default=Path("connect_entropy_kl.png"), type=Path, help="Output figure path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache_path
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    inferred_prompt, inferred_source = _load_result_metadata(cache_path, args.result_json)
    prompt_type = args.prompt_type or inferred_prompt or "base_math"
    data_source = args.data_source or inferred_source

    tokenizer_path = args.tokenizer or args.tuned_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    dtype = _best_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tuned_model = AutoModelForCausalLM.from_pretrained(args.tuned_model, torch_dtype=dtype).to(device).eval()
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model, torch_dtype=dtype).to(device).eval()

    samples = load_samples(
        cache_path=cache_path,
        tokenizer=tokenizer,
        prompt_type=prompt_type,
        data_source=data_source,
        sample_limit=args.sample_limit,
        prompt_model_hint=args.tuned_model,
    )
    if not samples:
        raise SystemExit("No valid samples found in cache file.")

    print(f"[1/2] Computing pi_tuned logprobs for {len(samples)} samples...")
    tuned_logps, tuned_entropy = compute_batched_logprobs(
        tuned_model,
        tokenizer,
        samples,
        args.batch_size,
        desc="pi_tuned logprobs",
    )
    print("Finished computing pi_tuned logprobs.")

    print(f"[2/2] Computing pi_ref logprobs for {len(samples)} samples...")
    ref_logps, ref_entropy = compute_batched_logprobs(
        ref_model,
        tokenizer,
        samples,
        args.batch_size,
        desc="pi_ref logprobs",
    )
    print("Finished computing pi_ref logprobs.")

    stats = summarize_statistics(samples, tuned_logps, ref_logps, tuned_entropy, ref_entropy)
    plot_path = plot_entropy_histograms(stats, args.figure_path)

    print(f"Processed {stats['total_tokens']} response tokens across {len(samples)} trajectories.")
    print(f"Average KL(pi_tuned || pi_ref): {stats['mean_kl']:.4f} nats.")
    print(
        f"Correct trajectories: {stats['correct_condition']} / {stats['correct_tokens']} "
        f"tokens improved ({stats['fraction_correct']:.2%})."
    )
    print(
        f"Incorrect trajectories: {stats['incorrect_condition']} / {stats['incorrect_tokens']} "
        f"tokens suppressed ({stats['fraction_incorrect']:.2%})."
    )
    print(f"Entropy histograms saved to {plot_path}")


if __name__ == "__main__":
    main()
