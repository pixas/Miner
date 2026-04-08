
import os
import sys
import json

import re
from collections import defaultdict
from typing import List, Optional

from tqdm import tqdm
import pandas as pd



import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class GeminiServer:
    def __init__(self):
        self.api_key = 'sk-HWFg9xacp788T8sfuJTiaXEaAffS4nGJn9dEcI8Gjfc0NuMZ'
        self.base_url = "http://192.154.241.225:3000/v1beta/models/gemini-2.5-pro:generateContent"
        # os.environ['http_proxy'] = os.environ['https_proxy'] = os.environ['GPT_PROXY']
    def __call__(self, messages, thinking=False, **gen):
        headers = {
            "Content-Type": "application/json"
        }
        temperature = gen.pop("temperature", 1.0)
        maxOutputTokens = gen.pop("max_new_tokens", 8)
        
        generation_config = {"temperature": temperature, 
                             "maxOutputTokens": maxOutputTokens,
                             "ThinkingConfig": {"thinkingBudget":0}}
        generation_config.update(gen)
        
        url = f"{self.base_url}?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": messages[-1]["content"]}
                    ]
                }
            ],
            "generationConfig": generation_config
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            # 尝试提取生成的文本（根据 Gemini API 的响应结构）
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text
            except (KeyError, IndexError, TypeError) as e:
                print("Unexpected response format:")
                print(json.dumps(result, indent=2))
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)
        
        return None

    def batch_call(self, prompts: List, max_workers: int = 16, thinking: bool = False, **gen) -> List:
        """
        Concurrently call the Gemini server for a batch of prompts.

        - Accepts a list of prompts where each item can be either a raw string
          or a message list compatible with __call__ (list of dicts with "content").
        - Returns a list of responses aligned with the input order.
        - Optimized for speed using ThreadPoolExecutor.
        """
        def to_messages(item):
            if isinstance(item, str):
                return [{"content": item, "role": "user"}]
            return item

        items = [to_messages(p) for p in prompts]

        results = [None] * len(items)

        def worker(idx_item):
            idx, msg = idx_item
            try:
                res = self(msg, thinking=thinking, **gen)
            except Exception:
                res = None
            return idx, res

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, (i, m)) for i, m in enumerate(items)]
            # Add a progress bar that updates as each future completes
            with tqdm(total=len(futures), desc="batch", unit="req", leave=False, disable=not sys.stdout.isatty()) as pbar:
                for fut in as_completed(futures):
                    i, res = fut.result()
                    results[i] = res
                    pbar.update(1)

        return results
from rollout.vllm_rollout import VLLMRollout
model = VLLMRollout("/mnt/phwfile/medai_p/LLMModels/LLMs/Qwen3-8B", )

OPTION_PATTERN = re.compile(r"^\s*([A-Z])[\)\.]\s*(.+)$", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"^```(?:[a-zA-Z0-9_+-]+)?\n([\s\S]*?)```$", re.MULTILINE)
PROMPT_TEMPLATE = """You are given a text fragment that may contain scientific expressions from physics, chemistry, biology, or related fields. Your task is to **normalize only the mathematical or quantitative parts** so they conform to standard LaTeX syntax that can be reliably parsed by mathematical evaluation tools, while leaving non-mathematical content unchanged.

Apply the following rules:

1. **Mathematical Standardization**:  
   - Convert any scientific notation (e.g., `2.6e5`, `3.14 \\times 1e-3`) into proper LaTeX using `10^{...}` and `\\times`.  
   - Notations that are not in scientific format should not be transformed into scientific notations.
   - Ensure all mathematical expressions are valid LaTeX (e.g., use `^{-2}` for negative exponents, `\\frac{}{}` for fractions, etc.).  
   - Do **not** add extra formatting unless required for syntactic correctness.

2. **Unit Handling**:  
   - If units are present (e.g., m, kg, mol, J, s), wrap them in `\\text{}` to ensure they render as text in math mode.  
   - **Do not** insert LaTeX spacing commands such as `\\,`, `\\ `, or `\\:`. Use only a single literal space between the numerical expression and the unit (e.g., `2.6 \\times 10^{5} \\text{m}`, not `2.6 \\times 10^{5}\,\\text{m}`).

3. **Non-Mathematical Content**:  
   - If the input is not related to math expressions, **leave it exactly as is**—do not attempt to wrap or modify it.

4. **Output Format**:  
   - Return **only** the normalized expression—no explanations, prefixes, or markdown.

**Examples:**

- Input: 2.6e5 m
  Output: 2.6 \\times 10^{5} \\text{m}

- Input: -4.7 \\times 1e-3 kg  
  Output: -4.7 \\times 10^{-3} \\text{kg}

- Input: 0.77c
  Output: 0.77c

- Input: glucose
  Output: glucose

- Input: 5.0 N 
  Output: 5.0 \\text{N}

- Input: The enzyme catalyzes the reaction.  
  Output: The enzyme catalyzes the reaction.

Now process the following input: """
DEFAULT_PARQUET_PATH = "data/gpqa_diamond/test.parquet"


def _extract_math_from_response(response: Optional[str]) -> Optional[str]:
    if not response:
        return None
    text = response.strip()
    fence_match = CODE_FENCE_PATTERN.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    lowered = text.lower()
    for marker in ("normalized expression", "expression", "result"):
        if lowered.startswith(marker):
            colon = text.find(":")
            if colon != -1:
                text = text[colon + 1 :].strip()
                break
    if text.startswith("$$") and text.endswith("$$") and len(text) > 4:
        return text[2:-2].strip()
    if text.startswith("$") and text.endswith("$") and len(text) > 2:
        return text[1:-1].strip()
    inline_match = re.search(r"\${1,2}([^$]+)\${1,2}", text, re.DOTALL)
    if inline_match:
        return inline_match.group(1).strip()
    stripped = text.strip("$ ").strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    if not first_line:
        return None
    return first_line.strip("$ ").strip("`")


def _normalize_plain_text(text: str) -> Optional[str]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    return stripped.strip("$ ")


def normalize_gpqa_ground_truth(parquet_path: str = DEFAULT_PARQUET_PATH) -> None:
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} GPQA entries from {parquet_path}")
    text_to_entries = defaultdict(list)
    reward_models = df["reward_model"].tolist()
    for idx, reward_model in tqdm(
        enumerate(reward_models),
        total=total_rows,
        desc="Scanning answers",
    ):
        if not isinstance(reward_model, dict):
            continue
        answer = reward_model.get("ground_truth")
        if not isinstance(answer, str):
            continue
        match = OPTION_PATTERN.match(answer.strip())
        if not match:
            continue
        option_letter, option_text = match.groups()
        option_text = option_text.strip()
        if not option_text:
            continue
        text_to_entries[option_text].append((idx, option_letter))
    if not text_to_entries:
        return

    unique_choices = list(text_to_entries.keys())
    print(f"Found {len(unique_choices)} unique answer texts to normalize")
    prompts = [PROMPT_TEMPLATE + choice for choice in unique_choices]
    prompts = [model.tokenizer.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in prompts]
    # server = GeminiServer()
    print("Requesting Qwen normalization...")
    responses = model.rollout(prompts, wrap_chat=False, temperature=0.7, top_p=0.8, top_k=20, do_sample=True, n=1)['response_text']
    # responses = server.batch_call(
    #     prompts,
    #     max_workers=min(32, max(1, os.cpu_count() or 1)),
    #     temperature=0.0,
    #     max_new_tokens=128,
    # )

    normalized_text = {}
    for original, response in tqdm(
        zip(unique_choices, responses),
        total=len(unique_choices),
        desc="Parsing responses",
    ):
        # print(response)
        cleaned = _extract_math_from_response(response)
        if "no math" in cleaned.lower() or "The theory is not renormalizable" in cleaned:
            cleaned = original 
            
        print(original, cleaned, sep='\t')
        if not cleaned:
            cleaned = _normalize_plain_text(original)
        if not cleaned:
            cleaned = original
        
        normalized_text[original] = cleaned

    for option_text, entries in tqdm(
        text_to_entries.items(),
        total=len(text_to_entries),
        desc="Writing updates",
    ):
        cleaned = normalized_text.get(option_text)
        if not cleaned:
            continue
        for row_idx, option_letter in entries:
            reward_model = df.at[row_idx, "reward_model"]
            if not isinstance(reward_model, dict):
                continue
            updated_model = dict(reward_model)
            updated_model["cleaned_ground_truth"] = f"{option_letter}) {cleaned}"
            df.at[row_idx, "reward_model"] = updated_model

    df.to_parquet(parquet_path.replace("test.parquet", "test_debug.parquet"))
    print(f"Saved normalized dataset to {parquet_path.replace('test.parquet', 'test_debug.parquet')}")


if __name__ == "__main__":
    normalize_gpqa_ground_truth()
    # df = pd.read_parquet(DEFAULT_PARQUET_PATH.replace("test.parquet", "test_debug.parquet"))
    # print(df.iloc[1]['reward_model'])
