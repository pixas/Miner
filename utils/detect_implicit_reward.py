import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from evaluation.utils.prompt import get_prepare_func

def _best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bfloat16 if available, otherwise float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_and_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        # Ensure we have a pad token for batching convenience
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_best_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tok


def _extract_pred_answer(text: str) -> Optional[str]:
    """Extract a final answer from text, preferring \\boxed{...}. Returns string if found."""
    # Prefer boxed{} style
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    # Fallback: look for 'Final Answer:' patterns
    m = re.search(r"Final\s*Answer\s*[:：]\s*([\-+]?\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: trailing = number
    m = re.search(r"=\s*([\-+]?\d+)\s*$", text)
    if m:
        return m.group(1).strip()
    return None


def iter_samples(path: Path, only_correct: bool = False, only_incorrect: bool = False) -> Iterable[tuple[str, str]]:
    """
    Iterate samples as (prompt, response).

    Supported formats:
    - .jsonl: objects with keys 'prompt' and 'response'.
    - .tsv: two columns per line: prompt<TAB>response.
    - .txt: each line is treated as response (empty prompt).
    """
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Support custom cache.jsonl schema: {"task": {instruction, input, ...}, "trajectory": ["...", ...]}
                if "task" in obj and isinstance(obj["task"], dict):

                    prompt = get_prepare_func(None, "base_math")(obj["task"])['prompt']
                    traj = obj.get("trajectory", [])
                    # if isinstance(traj, list) and len(traj) > 0:
                    #     # Use the last step as response
                    #     response = traj[-1] if isinstance(traj[-1], str) else json.dumps(traj[-1], ensure_ascii=False)
                    # else:
                    #     response = obj.get("response", obj.get("output", obj.get("text", "")))

                    # Optional filtering by correctness
                    if only_correct or only_incorrect:
                        acc_list = obj.get("all_acc")
                        response = None
                        for i, t in enumerate(traj):
                            
                        # Prefer explicit accuracy signal from cache: obj['all_acc'][-1]

                            is_correct = bool(acc_list[i])

                            if only_correct and not is_correct:
                                continue
                            if only_incorrect and is_correct:
                                continue
                            response = traj[i]
                            break
                        if response is None:
                            continue
                    else:
                        response = traj[-1]
                    
                else:
                    prompt = obj.get("prompt", "")
                    response = obj.get("response", obj.get("output", obj.get("text", "")))
                yield prompt, response
        elif suffix in {".tsv", ".csv"}:  # treat CSV as TSV if no commas expected
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                # Prefer tab split; if no tab, fall back to comma once
                parts = line.split("\t") if "\t" in line else line.split(",", 1)
                if len(parts) == 1:
                    yield "", parts[0]
                else:
                    yield parts[0], parts[1]
        else:
            # default: .txt or any plain text → each line is a response
            for line in f:
                line = line.strip()
                if line:
                    yield "", line


@torch.inference_mode()
@torch.inference_mode()
def compute_response_logprob_and_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """
    Compute per-token logprob of the response tokens given (prompt + response).

    Returns tokens (decoded response token strings) and a tensor of logprobs per response token.
    """
    # Tokenize separately to know the response span
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    resp_enc = tokenizer(response, add_special_tokens=False, return_tensors="pt")
    response_ids = resp_enc.input_ids[0]
    # If no prompt, prepend BOS if available to enable log-prob for first token
    if prompt_ids.numel() == 0 and tokenizer.bos_token_id is not None:
        prompt_ids = torch.tensor([tokenizer.bos_token_id], dtype=response_ids.dtype)

    # Build joint sequence
    input_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    # Forward pass
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits  # [1, T, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Identify positions corresponding to response tokens in shifted positions
    # Response token positions in the original input are indices [len(prompt_ids), ..., len(prompt_ids)+len(response_ids)-1]
    # After shift, those are one step earlier
    start = len(prompt_ids)
    end = start + len(response_ids)  # exclusive
    # Corresponding in shifted arrays: positions [start-1, end-2] produce token indices [start, end-1]
    # Handle edge case when start == 0
    start_shift = max(0, start - 1)
    end_shift = end - 1
    # Slice the relevant window
    window_logits = shift_logits[:, start_shift:end_shift, :]
    window_labels = shift_labels[:, start_shift:end_shift]

    logprobs = torch.nn.functional.log_softmax(window_logits, dim=-1)
    probs = logprobs.exp()
    # token-wise entropy H(p) = -sum p log p
    ent = -(probs * logprobs).sum(dim=-1).squeeze(0)
    resp_logprobs = torch.gather(logprobs, dim=-1, index=window_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
    
    

    # Decode response tokens for visualization (best-effort; use the fast tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(response_ids.tolist())
    return tokens, resp_logprobs.detach().to("cpu"), ent.detach().to("cpu")


def color_from_value(v: float, vmax: float) -> str:
    """Map value in [-vmax, vmax] to a red-white-green gradient color hex."""
    vmax = max(vmax, 1e-6)
    x = max(-vmax, min(v, vmax)) / vmax  # in [-1, 1]
    # Negative → red, positive → green. Interpolate toward white at 0.
    if x >= 0:
        r, g, b = int((1 - x) * 255), 255, int((1 - x) * 255)
    else:
        r, g, b = 255, int((1 + x) * 255), int((1 + x) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def color_from_delta(v: float, vmax: float) -> str:
    """Map signed delta in [-vmax, vmax] to blue-white-red.

    Negative (decrease) → blue, positive (increase) → red, near 0 → white.
    """
    vmax = max(vmax, 1e-6)
    x = max(-vmax, min(v, vmax)) / vmax  # in [-1, 1]
    # Negative → red, positive → green. Interpolate toward white at 0.
    if x >= 0:
        r, g, b = int((1 - x) * 255), 255, int((1 - x) * 255)
    else:
        r, g, b = 255, int((1 + x) * 255), int((1 + x) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_html(prompt: str, tokens: list[str], values: torch.Tensor, title: Optional[str] = None) -> str:
    vmax = float(values.abs().max().item() if values.numel() > 0 else 1.0)
    spans = []
    for t, v in zip(tokens, values.tolist()):
        color = color_from_value(v, vmax)
        safe_t = (t.replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("&", "&amp;"))
        # make control/newline characters visible to avoid rendering issues
        safe_t = safe_t.replace("\n", "⏎")
        spans.append(
            f'<span style="background:{color}; padding:1px 2px; margin:1px; border-radius:3px; display:inline-block;">{safe_t}</span>'
        )

    prompt_html = (prompt.replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace("&", "&amp;")
                        .replace("\n", "<br>"))

    title_html = f"<h3>{(title or '').replace('<','&lt;').replace('>','&gt;')}</h3>" if title else ""
    legend = (
        "<div style=\"margin:6px 0; font-size:12px;\">"
        "Color = log pi_tuned - log pi_ref per token (red=lower, green=higher)."
        "</div>"
    )
    return (
        f"<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;\">"
        f"{title_html}"
        f"<div style=\"margin-bottom:4px; color:#666;\"><b>Prompt</b>: {prompt_html}</div>"
        f"{legend}"
        f"<div style=\"line-height:1.8; white-space: pre-wrap; word-break: break-word;\">{''.join(spans)}</div>"
        f"</div>"
    )


def render_dual_html(prompt: str, tokens: list[str], logratio: torch.Tensor, entropy_change: torch.Tensor, title: Optional[str] = None) -> str:
    # Normalize ranges
    vmax_lr = float(logratio.abs().max().item() if logratio.numel() > 0 else 1.0)
    # Symmetric robust scaling for entropy change using 95th percentile of |delta|
    if entropy_change.numel() > 0:
        eabs = np.abs(entropy_change.detach().float().cpu().numpy())
        vmax_e = float(np.quantile(eabs, 0.95)) if eabs.size > 0 else 1.0
        vmax_e = max(vmax_e, 1e-6)
    else:
        vmax_e = 1.0

    left_spans = []
    right_spans = []
    for t, v_lr, v_e in zip(tokens, logratio.tolist(), entropy_change.tolist()):
        color_lr = color_from_value(v_lr, vmax_lr)
        color_e = color_from_delta(v_e, vmax_e)
        safe_t = (t.replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("&", "&amp;")
                    .replace("\n", "⏎"))
        left_spans.append(
            f'<span style="background:{color_lr}; padding:1px 2px; margin:1px; border-radius:3px; display:inline-block;">{safe_t}</span>'
        )
        right_spans.append(
            f'<span style="background:{color_e}; padding:1px 2px; margin:1px; border-radius:3px; display:inline-block;">{safe_t}</span>'
        )

    prompt_html = (prompt.replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace("&", "&amp;")
                        .replace("\n", "<br>"))
    title_html = f"<h3>{(title or '').replace('<','&lt;').replace('>','&gt;')}</h3>" if title else ""

    return (
        "<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin-bottom: 16px;\">"
        f"{title_html}"
        f"<div style=\"margin-bottom:6px; color:#666;\"><b>Prompt</b>: {prompt_html}</div>"
        "<div style=\"display:flex; gap:16px; align-items:flex-start;\">"
        "  <div style=\"flex:1;\">"
        "    <div style=\"font-size:12px; margin-bottom:4px;\">log pi_tuned - log pi_ref</div>"
        f"    <div style=\"line-height:1.8; white-space: pre-wrap; word-break: break-word;\">{''.join(left_spans)}</div>"
        "  </div>"
        "  <div style=\"flex:1;\">"
        "    <div style=\"font-size:12px; margin-bottom:4px;\">entropy change (tuned - base)</div>"
        f"    <div style=\"line-height:1.8; white-space: pre-wrap; word-break: break-word;\">{''.join(right_spans)}</div>"
        "  </div>"
        "</div>"
        "</div>"
    )


def tokens_for_display(tokenizer: AutoTokenizer, text: str) -> list[str]:
    """Split original text by token offsets to get human-readable pieces.

    Falls back to sanitizing token strings if offsets are unavailable.
    """
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping")
        if offsets is not None and len(offsets) > 0:
            offs = offsets[0]
            pieces = []
            for s, e in offs:
                if s is None or e is None:
                    pieces.append("")
                else:
                    pieces.append(text[s:e])
            return pieces
    except Exception:
        pass
    # Fallback: sanitize common space markers from BPE/SentencePiece
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    cleaned = []
    for t in toks:
        t = t.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")
        cleaned.append(t)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Visualize implicit reward and entropy for responses")
    parser.add_argument("--tuned_model", required=False, help="Path to tuned HF model")
    parser.add_argument("--base_model", required=False, help="Path to base/reference HF model")
    parser.add_argument("--responses", required=False, help="Path to responses file (.jsonl/.tsv/.txt)")
    parser.add_argument("--result_json", required=False, help="Path to result.json to infer models and cache.jsonl")
    parser.add_argument("--output", default="outputs/implicit_reward.html", help="Output HTML file path")
    parser.add_argument(
        "--reward_mode",
        choices=["token", "cumsum"],
        default="token",
        help="How to compute implicit reward: per-token log pi_tuned - log pi_ref ('token') or cumulative cumsum over tokens ('cumsum')",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max number of samples to process (0=all)")
    parser.add_argument("--only_correct", action="store_true", help="Only iterate correct responses (when label present)")
    parser.add_argument("--only_incorrect", action="store_true", help="Only iterate incorrect responses (when label present)")
    args = parser.parse_args()

    if args.only_correct and args.only_incorrect:
        raise SystemExit("Cannot set both --only_correct and --only_incorrect")

    # Resolve models and responses from result.json if provided
    tuned_model_path = args.tuned_model
    base_model_path = args.base_model
    responses_path_arg = args.responses

    if args.result_json:
        rpath = Path(args.result_json)
        if not rpath.exists():
            raise SystemExit(f"result.json not found: {rpath}")
        with rpath.open("r", encoding="utf-8") as rf:
            robj = json.load(rf)
        # Allow both top-level and nested under 'args'
        tuned_model_path = tuned_model_path or robj.get("model_name_or_path") or robj.get("tuned_model") or robj.get("args", {}).get("model_name_or_path")
        base_model_path = base_model_path or robj.get("hf_model_path") or robj.get("base_model") or robj.get("args", {}).get("hf_model_path")
        # If responses not provided, infer sibling cache.jsonl
        if responses_path_arg is None:
            sibling = rpath.parent / "cache.jsonl"
            responses_path_arg = str(sibling)

    if not tuned_model_path or not base_model_path:
        raise SystemExit("Missing tuned/base model paths. Provide --tuned_model and --base_model, or --result_json containing them.")
    if not responses_path_arg:
        raise SystemExit("Missing --responses and could not infer from --result_json.")

    tuned_model, tuned_tok = load_model_and_tokenizer(tuned_model_path)
    base_model, base_tok = load_model_and_tokenizer(base_model_path)

    # Prefer tuned tokenizer for both to match tokenization used during fine-tuning
    tok = tuned_tok

    in_path = Path(responses_path_arg)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"<h2 style='font-family: sans-serif;'>Implicit Reward "
        f"({'per-token' if args.reward_mode=='token' else 'cumulative cumsum'})"
        f" (log pi_tuned - log pi_ref)</h2>"
    )
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Implicit Reward Visualization</title></head><body>",
        header,
    ]

    count = 0
    for idx, (prompt, response) in enumerate(iter_samples(in_path, args.only_correct, args.only_incorrect)):
        if args.limit and count >= args.limit:
            break

        # Compute with tuned model (logprob and entropy)
        _, tuned_lp, tuned_ent = compute_response_logprob_and_entropy(tuned_model, tok, prompt, response)
        # Compute with base model
        _, base_lp, ref_ent = compute_response_logprob_and_entropy(base_model, tok, prompt, response)

        # Safety: lengths should match response tokenization
        # Tokenize response once for display using offsets
        resp_ids = tok(response, add_special_tokens=False, return_tensors="pt").input_ids[0]
        resp_tokens = tokens_for_display(tok, response)

        # Align sizes if any off-by-one occurs (shouldn't, but guard)
        n = min(len(resp_tokens), tuned_lp.shape[0], base_lp.shape[0])
        resp_tokens = resp_tokens[:n]
        # Implicit reward as log-ratio; optionally cumulative over tokens
        reward = (tuned_lp[:n] - base_lp[:n]).detach().cpu()
        if args.reward_mode == "cumsum":
            reward = torch.cumsum(reward, dim=0)
        entropy_diff = (tuned_ent[:n] - ref_ent[:n]).detach().cpu()
        # entropy = tuned_ent[:n].detach().cpu()

        html_parts.append(render_dual_html(prompt, resp_tokens, reward, entropy_diff, title=f"Sample {idx}"))
        html_parts.append("<hr>")
        count += 1

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Wrote visualization to {out_path}")


if __name__ == "__main__":
    main()
