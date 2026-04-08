#!/usr/bin/env python3
"""
Visualize trajectory-level advantages and token-wise pi_new/pi_ref ratios for fully-correct math problems.

The script loads the tuned (pi_new) and base (pi_ref) checkpoints, recomputes log-probabilities for every
trajectory belonging to questions whose cached generations are all correct, and emits an interactive HTML
viewer. Each page shows one problem alongside up to N trajectories (default 16). Tokens are color-coded by
log(pi_new/pi_ref) and display the clipped ratio, while per-trajectory advantages are derived from the
negative z-score of their sequence log-probabilities under pi_new.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.utils.prompt import get_prepare_func

def _best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_and_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_best_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def clean_tokens(toks: Iterable[str]) -> List[str]:
    cleaned = []
    for t in toks:
        if t is None:
            cleaned.append("")
            continue
        cleaned.append(t.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " "))
    return cleaned

# todo: 其实不用展示ref的entropy，我想要同时可以比较

@dataclass
class QuestionEntry:
    idx: int
    task_id: str
    dataset: str
    instruction: str
    input_text: str
    eval_answer: str
    prompt: str
    sample_indices: list[int] = field(default_factory=list)


@dataclass
class TokenizedSample:
    question_index: int
    response_index: int
    prompt_text: str
    response_text: str
    input_ids: list[int]
    response_ids: list[int]
    prompt_len: int
    response_len: int
    tokens: list[str]
    logprobs_new: np.ndarray | None = None
    logprobs_ref: np.ndarray | None = None
    entropy_new: np.ndarray | None = None
    entropy_ref: np.ndarray | None = None
    seq_logprob: float | None = None
    advantage: float | None = None


def prepare_questions_and_samples(
    cache_path: str,
    tokenizer: AutoTokenizer,
    per_question_limit: int,
    max_questions: int,
    sample_limit: int,
) -> tuple[list[QuestionEntry], list[TokenizedSample]]:
    prompt_builder = get_prepare_func(None, "base_math")
    questions: list[QuestionEntry] = []
    samples: list[TokenizedSample] = []
    prompt_cache: dict[str, list[int]] = {}
    sample_limit = max(0, sample_limit)
    stop = False
    with open(cache_path, "r", encoding="utf-8") as f:
        for raw in f:
            if stop:
                break
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            acc = obj.get("all_acc") or []
            if not acc or not all(bool(x) for x in acc):
                continue

            task = obj.get("task", {})
            prompt = prompt_builder(task)["prompt"]
            prompt_ids = prompt_cache.get(prompt)
            if prompt_ids is None:
                prompt_ids = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                if not prompt_ids and tokenizer.bos_token_id is not None:
                    prompt_ids = [tokenizer.bos_token_id]
                prompt_cache[prompt] = prompt_ids

            q_entry = QuestionEntry(
                idx=len(questions),
                task_id=task.get("id", ""),
                dataset=task.get("dataset", ""),
                instruction=task.get("instruction", ""),
                input_text=task.get("input", ""),
                eval_answer=task.get("eval", ""),
                prompt=prompt,
            )

            trajectories = obj.get("trajectory", []) or []
            for resp_idx, response in enumerate(trajectories[:per_question_limit]):
                resp_ids = tokenizer(response, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                if not resp_ids:
                    continue
                sample = TokenizedSample(
                    question_index=q_entry.idx,
                    response_index=resp_idx,
                    prompt_text=prompt,
                    response_text=response,
                    input_ids=list(prompt_ids) + resp_ids,
                    response_ids=resp_ids,
                    prompt_len=len(prompt_ids),
                    response_len=len(resp_ids),
                    tokens=clean_tokens(tokenizer.convert_ids_to_tokens(resp_ids)),
                )
                q_entry.sample_indices.append(len(samples))
                samples.append(sample)
                if sample_limit and len(samples) >= sample_limit:
                    stop = True
                    break

            if not q_entry.sample_indices:
                continue

            questions.append(q_entry)
            if max_questions and len(questions) >= max_questions:
                stop = True
            if sample_limit and len(samples) >= sample_limit:
                stop = True
            if stop:
                break

    return questions, samples


def compute_batched_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[TokenizedSample],
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not samples:
        return [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    outputs: list[np.ndarray] = [None] * len(samples)  # type: ignore
    entropies: list[np.ndarray] = [None] * len(samples)  # type: ignore

    for start in range(0, len(samples), batch_size):
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

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
            log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)

        for row, sample in enumerate(chunk):
            start_pos = max(sample.prompt_len - 1, 0)
            end_pos = start_pos + sample.response_len
            available = log_probs.shape[1]
            if end_pos > available:
                delta = end_pos - available
                end_pos = available
                # Trim labels if logits truncated.
                resp_ids = sample.response_ids[: sample.response_len - delta]
            else:
                resp_ids = sample.response_ids

            if end_pos <= start_pos or not resp_ids:
                gathered = torch.empty((0,), dtype=torch.float32)
                ent = torch.empty((0,), dtype=torch.float32)
            else:
                labels = torch.tensor(resp_ids, dtype=torch.long, device=device)
                window = log_probs[row, start_pos:end_pos, :]
                gathered = window.gather(-1, labels.unsqueeze(-1)).squeeze(-1).detach().to("cpu")
                probs = window.exp()
                ent = torch.sum((-window) * probs, dim=-1).detach().to("cpu")
            outputs[start + row] = gathered.float().numpy()
            entropies[start + row] = ent.float().numpy()

    return outputs, entropies


def compute_advantages(questions: list[QuestionEntry], samples: list[TokenizedSample]):
    for question in questions:
        question_seq_scores = np.array([samples[sample_idx].seq_logprob for sample_idx in question.sample_indices if samples[sample_idx].seq_logprob is not None], dtype=np.float64)
        mean = float(question_seq_scores.mean())
        std = float(question_seq_scores.std())
        
        for sample_idx in question.sample_indices:
            samples[sample_idx].advantage = float(-(samples[sample_idx].seq_logprob - mean) / (std + 1e-6))
            


def color_from_delta(delta: float, vmax: float) -> str:
    vmax = max(vmax, 1e-6)
    x = min(max(delta, 0.0), vmax) / vmax
    r = 255
    g = int((1 - x) * 255)
    b = int((1 - x) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def format_token_span(
    token: str,
    delta: float,
    logp_new: float,
    logp_ref: float,
    color_limit: float,
    ratio_clip: float,
) -> str:
    ratio_clip = max(1.0, ratio_clip)
    clip_log = math.log(ratio_clip)
    clipped_delta = max(-clip_log, min(clip_log, delta))
    ratio_display = math.exp(clipped_delta)

    
    # display the KL loss = ratio_display - clipped_delta - 1
    ratio_display = ratio_display - clipped_delta - 1
    
    sign = "-" if ratio_display > 1 else "+"
    
    color = color_from_delta(ratio_display, color_limit)
    safe = html.escape(token).replace("\n", "<br/>")
    if not safe.strip():
        safe = "&nbsp;"
    title = (
        f"Δlogp={delta:.3f} | logp_new={logp_new:.3f} | "
        f"logp_ref={logp_ref:.3f} | kl≈{ratio_display:.2f}"
    )
    return (
        f"<span class='token' style='background-color:{color}' title='{title}'>"
        f"{safe}<sub>{sign}{ratio_display:.2f}</sub></span>"
    )

def color_from_entropy(value: float, vmax: float) -> str:
    vmax = max(vmax, 1e-6)
    x = max(0.0, min(1.0, value / vmax))
    light = (230, 249, 230)   # #e6f9e6
    dark = (46, 125, 50)      # #2e7d32
    r = int(light[0] + (dark[0] - light[0]) * x)
    g = int(light[1] + (dark[1] - light[1]) * x)
    b = int(light[2] + (dark[2] - light[2]) * x)
    return f"#{r:02x}{g:02x}{b:02x}"


def format_entropy_span(token: str, entropy: float, entropy_limit: float) -> str:
    color = color_from_entropy(entropy, entropy_limit)
    safe = html.escape(token).replace("\n", "<br/>")
    if not safe.strip():
        safe = "&nbsp;"
    title = f"Entropy≈{entropy:.3f} nats"
    return (
        f"<span class='entropy-token' style='background-color:{color}' title='{title}'>"
        f"{safe}<sub>H={entropy:.2f}</sub></span>"
    )


def render_entropy_block(sample: TokenizedSample, entropy_limit: float) -> str:
    tokens = sample.tokens
    rows: list[str] = []
    if sample.entropy_new is not None and len(sample.entropy_new):
        spans = [
            format_entropy_span(t, float(h), entropy_limit) for t, h in zip(tokens, sample.entropy_new)
        ]
        rows.append(
            "<div class='entropy-row'><span class='entropy-label'>pi_new</span>"
            f"<div class='entropy-grid'>{''.join(spans)}</div></div>"
        )
    if sample.entropy_ref is not None and len(sample.entropy_ref):
        spans = [
            format_entropy_span(t, float(h), entropy_limit) for t, h in zip(tokens, sample.entropy_ref)
        ]
        rows.append(
            "<div class='entropy-row'><span class='entropy-label'>pi_ref</span>"
            f"<div class='entropy-grid'>{''.join(spans)}</div></div>"
        )
    if not rows:
        return ""
    return (
        "<div class='entropy-block'>"
        "<div class='entropy-title'>Token entropy (higher = darker)</div>"
        f"{''.join(rows)}"
        "</div>"
    )


def render_sample_block(
    sample: TokenizedSample,
    ordinal: int,
    color_limit: float,
    ratio_clip: float,
    entropy_limit: float,
) -> str:
    tokens = sample.tokens
    if sample.logprobs_new is None or sample.logprobs_ref is None:
        return ""
    min_len = min(len(tokens), len(sample.logprobs_new), len(sample.logprobs_ref))
    tokens = tokens[:min_len]
    lp_new = sample.logprobs_new[:min_len]
    lp_ref = sample.logprobs_ref[:min_len]
    deltas = lp_new - lp_ref
    token_spans = [
        format_token_span(t, float(d), float(n), float(r), color_limit, ratio_clip)
        for t, d, n, r in zip(tokens, deltas, lp_new, lp_ref)
    ]
    advantage = sample.advantage if sample.advantage is not None else 0.0
    seq_lp = sample.seq_logprob if sample.seq_logprob is not None else 0.0
    entropy_section = render_entropy_block(sample, entropy_limit)
    return (
        "<div class='trajectory'>"
        "<div class='trajectory-header'>"
        f"<span>Trajectory #{ordinal + 1}</span>"
        f"<span>Advantage: {advantage:.3f}</span>"
        f"<span>Seq logp: {seq_lp:.2f}</span>"
        f"<span>Tokens: {len(tokens)}</span>"
        "</div>"
        f"<div class='token-grid'>{''.join(token_spans)}</div>"
        f"{entropy_section}"
        "</div>"
    )


def render_html(
    questions: list[QuestionEntry],
    samples: list[TokenizedSample],
    output_path: str,
    color_limit: float,
    ratio_clip: float,
    entropy_limit: float,
):
    if not questions:
        raise SystemExit("No fully-correct questions found in cache.jsonl.")
    total_pages = len(questions)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'/>"
        "<title>pi_new vs pi_ref visualization</title>"
        "<style>",
        "body {font-family: 'Inter', 'Segoe UI', sans-serif; background:#f5f7fb; margin:0; padding:0;}",
        ".controls {position:sticky; top:0; background:#111827; color:#fff; padding:12px 18px; "
        "display:flex; align-items:center; gap:12px; z-index:10;}",
        ".controls button {background:#2563eb; color:#fff; border:none; border-radius:6px; padding:6px 16px;"
        "font-size:15px; cursor:pointer;}",
        ".controls button:disabled {opacity:0.45; cursor:not-allowed;}",
        ".page-indicator {font-weight:600; font-size:15px;}",
        ".note {font-size:13px; color:#cbd5f5; margin-left:auto;}",
        ".question-page {display:none; padding:24px; max-width:1200px; margin:0 auto;}",
        ".question-card {background:#fff; border-radius:16px; box-shadow:0 12px 40px rgba(15,23,42,0.13);"
        "padding:24px; margin-bottom:28px;}",
        ".question-header {display:flex; flex-wrap:wrap; gap:8px; align-items:baseline;}",
        ".question-header h2 {margin:0; font-size:22px; color:#0f172a;}",
        ".question-header span {color:#475569; font-size:14px;}",
        ".question-body pre {white-space:pre-wrap; font-size:15px; background:#f1f5f9; padding:14px;"
        "border-radius:12px;}",
        ".trajectory {background:#fff; border-left:4px solid #2563eb; margin-bottom:20px; border-radius:12px;"
        "padding:16px; box-shadow:0 4px 18px rgba(15,23,42,0.08);}",
        ".trajectory-header {display:flex; flex-wrap:wrap; gap:14px; font-size:13px; color:#0f172a; "
        "margin-bottom:10px; font-weight:600;}",
        ".token-grid {display:flex; flex-wrap:wrap; gap:4px; font-size:13px; line-height:1.35; "
        "background:#f8fafc; border-radius:10px; padding:10px;}",
        ".token {border-radius:6px; padding:2px 4px; display:inline-flex; flex-direction:column; "
        "align-items:center; min-width:18px; word-break:break-word;}",
        ".token sub {font-size:10px; color:#111;}",
        ".entropy-block {background:#fff7ed; border-radius:12px; padding:12px; margin-top:12px; "
        "display:flex; flex-direction:column; gap:8px;}",
        ".entropy-title {font-size:13px; font-weight:600; color:#7c2d12;}",
        ".entropy-row {display:flex; gap:10px; align-items:flex-start;}",
        ".entropy-label {font-weight:600; font-size:12px; color:#7c2d12; min-width:60px;}",
        ".entropy-grid {flex:1; display:flex; flex-wrap:wrap; gap:4px; background:#fff; border-radius:10px; "
        "padding:6px;}",
        ".entropy-token {border-radius:6px; padding:2px 4px; display:inline-flex; flex-direction:column; "
        "align-items:center; min-width:18px; word-break:break-word;}",
        ".entropy-token sub {font-size:10px; color:#1f2937;}",
        ".legend {font-size:13px; color:#475569; margin-bottom:14px;}",
        "</style></head><body>",
        "<div class='controls'>"
        "<button id='prevBtn' onclick='prevPage()'>&larr; Prev</button>"
        "<div class='page-indicator' id='pageIndicator'></div>"
        "<button id='nextBtn' onclick='nextPage()'>Next &rarr;</button>"
        "<div class='note'>Advantage = -zscore(seq logp under pi_new). Token colors encode log(pi_new/pi_ref); "
        "entropy rows darken as uncertainty grows.</div>"
        "</div>",
    ]

    token_ratio_note = (
        "<div class='legend'>Token labels show the clipped ratio (pi_new/pi_ref) per token. "
        "Blue indicates pi_new underweights the token vs. pi_ref, red indicates the opposite.</div>"
    )
    entropy_note = (
        "<div class='legend'>Entropy grids show per-token entropy for pi_new and pi_ref. "
        "Darker cells indicate higher uncertainty.</div>"
    )

    for question in questions:
        parts.append(f"<section class='question-page' id='page-{question.idx}'>")
        parts.append("<div class='question-card'>")
        parts.append("<div class='question-header'>")
        parts.append(f"<h2>Problem {question.idx + 1}</h2>")
        meta_bits = []
        if question.task_id:
            meta_bits.append(f"ID: {html.escape(question.task_id)}")
        if question.dataset:
            meta_bits.append(f"Dataset: {html.escape(question.dataset)}")
        if question.eval_answer:
            meta_bits.append(f"Answer: {html.escape(str(question.eval_answer))}")
        if meta_bits:
            parts.append(f"<span>{' | '.join(meta_bits)}</span>")
        parts.append("</div>")  # header
        if question.instruction:
            parts.append(f"<strong>Instruction:</strong><pre>{html.escape(question.instruction)}</pre>")
        if question.input_text:
            parts.append(f"<strong>Problem:</strong><pre>{html.escape(question.input_text)}</pre>")
        parts.append(token_ratio_note)
        parts.append(entropy_note)
        parts.append("</div>")  # question-card

        for local_idx, sample_idx in enumerate(question.sample_indices):
            sample = samples[sample_idx]
            parts.append(render_sample_block(sample, local_idx, color_limit, ratio_clip, entropy_limit))

        parts.append("</section>")

    parts.append(
        "<script>"
        "let currentPage = 0;"
        "const pages = document.querySelectorAll('.question-page');"
        "function showPage(idx){"
        "if(!pages.length)return;"
        "if(idx<0) idx = pages.length - 1;"
        "if(idx>=pages.length) idx = 0;"
        "pages.forEach((el,i)=>{el.style.display = i===idx ? 'block':'none';});"
        "document.getElementById('pageIndicator').textContent = `Page ${idx+1} / ${pages.length}`;"
        "currentPage = idx;"
        "document.getElementById('prevBtn').disabled = !pages.length;"
        "document.getElementById('nextBtn').disabled = !pages.length;"
        "}"
        "function nextPage(){showPage(currentPage+1);}"
        "function prevPage(){showPage(currentPage-1);}"
        "document.addEventListener('DOMContentLoaded',()=>showPage(0));"
        "</script>"
        "</body></html>"
    )

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[visualize_entropy] Saved {total_pages} pages to {out_path}")


def _extract_dataset_name(cache_path: str) -> str:
    path = Path(cache_path)
    parts = path.parts
    dataset = None
    for idx, part in enumerate(parts[:-1]):
        if part == "results" and idx + 1 < len(parts):
            dataset = parts[idx + 1]
    if dataset:
        return dataset
    parent = path.parent.name
    return parent or path.name


def _sanitize_segment(segment: str, fallback: str) -> str:
    segment = (segment or "").strip()
    if not segment:
        segment = fallback
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", segment)
    sanitized = sanitized.strip("_")
    return sanitized or fallback


def _default_output_path(tuned_model: str, cache_path: str) -> str:
    dataset_name = _sanitize_segment(_extract_dataset_name(cache_path), "dataset")
    tuned_name = _sanitize_segment(Path(tuned_model).name or tuned_model, "tuned")
    output_dir = Path("outputs") / dataset_name
    return str(output_dir / f"{dataset_name}__{tuned_name}.html")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize fully-correct math trajectories with token ratios.")
    parser.add_argument(
        "--tuned_model",
        default="/nvme/jiangshuyang/s3_mount/checkpoints/verl_math/deepscaler_qwen34b_grpo_bsz128_global_step_314",
        help="Path to pi_new (tuned) checkpoint.",
    )
    parser.add_argument(
        "--base_model",
        default="/mnt/phwfile/medai_p/LLMModels/LLMs/Qwen3-4B-Base",
        help="Path to pi_ref (base) checkpoint.",
    )
    parser.add_argument(
        "--cache_path",
        default="/mnt/petrelfs/jiangshuyang/repo/efficient_RL/results/math/"
        "deepscaler_qwen34b_grpo_bsz128_global_step_314_sc128/cache.jsonl",
        help="Path to cache.jsonl containing trajectories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination HTML path. Defaults to outputs/<dataset>/<dataset>__<tuned_model>.html when omitted.",
    )
    parser.add_argument(
        "--per_question_limit",
        type=int,
        default=16,
        help="Number of trajectories to visualize per fully-correct problem.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=0,
        help="Optional cap on number of fully-correct questions to visualize (0 = all).",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="Only process the first K valid trajectories overall (0 = all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for log-prob computation per model.",
    )
    parser.add_argument(
        "--ratio_clip",
        type=float,
        default=20.0,
        help="Maximum pi_new/pi_ref ratio shown in token subtitles (values beyond are clipped).",
    )
    return parser.parse_args()

def kl_from_diff(logp_diff):
    ratio = np.exp(logp_diff)
    return ratio - logp_diff - 1

def main():
    args = parse_args()
    if not args.output:
        args.output = _default_output_path(args.tuned_model, args.cache_path)
    tuned_model, tuned_tok = load_model_and_tokenizer(args.tuned_model)
    base_model, _ = load_model_and_tokenizer(args.base_model)

    questions, samples = prepare_questions_and_samples(
        args.cache_path,
        tuned_tok,
        args.per_question_limit,
        args.max_questions,
        args.sample_limit,
    )
    if not samples:
        raise SystemExit("No fully-correct samples found to visualize.")

    tuned_logps, tuned_entropies = compute_batched_logprobs(tuned_model, tuned_tok, samples, args.batch_size)
    base_logps, base_entropies = compute_batched_logprobs(base_model, tuned_tok, samples, args.batch_size)

    for sample, lp_new, lp_ref, ent_new, ent_ref in zip(
        samples, tuned_logps, base_logps, tuned_entropies, base_entropies
    ):
        min_len = min(len(lp_new), len(lp_ref), len(ent_new), len(ent_ref), sample.response_len)
        lp_new = lp_new[:min_len]
        lp_ref = lp_ref[:min_len]
        sample.logprobs_new = lp_new
        sample.logprobs_ref = lp_ref
        sample.entropy_new = ent_new[:min_len]
        sample.entropy_ref = ent_ref[:min_len]

        sample.seq_logprob = float(lp_new.mean()) if len(lp_new) else float("-inf")
        sample.response_len = min_len
        sample.tokens = sample.tokens[:min_len]

    compute_advantages(questions, samples)

    kl_arrays = [
        kl_from_diff(sample.logprobs_new - sample.logprobs_ref)
        for sample in samples 
        if sample.logprobs_new is not None and sample.logprobs_ref is not None
    ]
    # delta_arrays = [
    #     sample.logprobs_new - sample.logprobs_ref
    #     for sample in samples
    #     if sample.logprobs_new is not None and sample.logprobs_ref is not None
    # ]
    if kl_arrays:
        deltas = np.concatenate(kl_arrays)
        color_limit = float(np.percentile(np.abs(deltas), 95))
        if color_limit < 1e-3:
            color_limit = 1.0
    else:
        color_limit = 1.0

    entropy_arrays = [
        arr
        for sample in samples
        for arr in (sample.entropy_new, sample.entropy_ref)
        if arr is not None and len(arr)
    ]
    if entropy_arrays:
        entropy_vals = np.concatenate(entropy_arrays)
        entropy_limit = float(np.percentile(entropy_vals, 95))
        if entropy_limit < 1e-4:
            entropy_limit = float(np.max(entropy_vals))
        if entropy_limit < 1e-6:
            entropy_limit = 1.0
    else:
        entropy_limit = 1.0

    render_html(questions, samples, args.output, color_limit, args.ratio_clip, entropy_limit)


if __name__ == "__main__":
    main()
