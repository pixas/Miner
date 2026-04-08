from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path(
    "/mnt/petrelfs/jiangshuyang/datasets/medical_test/MedMCQA_cot.json"
)
DEFAULT_REFERENCE = Path("data/medqa/test.parquet")


DEFAULT_OUTPUT = Path("data/medmcqa/test.parquet")

OPTION_LINE_RE = re.compile(r"^\s*([A-Z])[\.\)]\s*(.*)$")
ANSWER_LINE_RE = re.compile(
    r"^\s*(?:ANS(?:WER)?|CORRECT\s*(?:OPTION|ANSWER)?)[:\s\-\.]*([A-Z])",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedMCQA test JSON into the MedQA Parquet schema."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to MedMCQA test JSON file.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference MedQA Parquet file (schema + metadata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Parquet file.",
    )
    return parser.parse_args()


def read_source_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}, got {type(payload)}")
    return payload


def read_reference_metadata(
    path: Path,
) -> Tuple[pa.Schema, str, str, str]:
    schema = pq.read_schema(path)
    parquet_file = pq.ParquetFile(path)
    sample_table = parquet_file.read_row_group(
        0, columns=["prompt", "reward_model", "extra_info"]
    )
    sample_row = sample_table.slice(0, 1).to_pylist()[0]

    prompt_messages = sample_row.get("prompt") or []
    system_prompt = ""
    for message in prompt_messages:
        if isinstance(message, dict) and message.get("role") == "system":
            system_prompt = message.get("content", "")
            break

    reward_style = (sample_row.get("reward_model") or {}).get("style") or "rule"
    default_split = (sample_row.get("extra_info") or {}).get("split") or "train"
    return schema, system_prompt, reward_style, default_split


def strip_trailing_answer_lines(lines: List[str]) -> Tuple[List[str], str]:
    stripped_lines = list(lines)
    declared_answer = ""
    while stripped_lines:
        tail = stripped_lines[-1].strip()
        if not tail:
            stripped_lines.pop()
            continue
        match = ANSWER_LINE_RE.match(tail)
        if match:
            declared_answer = match.group(1).upper()
            stripped_lines.pop()
            continue
        break
    return stripped_lines, declared_answer


def split_question_and_options(
    text: str,
) -> Tuple[str, Dict[str, str], str]:
    lines = text.replace("\r\n", "\n").split("\n")
    lines, trailing_answer = strip_trailing_answer_lines(lines)

    question_lines: List[str] = []
    options: Dict[str, str] = {}
    current_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() and not current_key:
            continue
        match = OPTION_LINE_RE.match(line)
        if match:
            current_key = match.group(1).upper()
            options[current_key] = match.group(2).strip()
            continue
        if current_key:
            options[current_key] = (
                options[current_key] + " " + line.strip()
            ).strip()
        else:
            question_lines.append(line.strip())

    question_part = "\n".join(l for l in question_lines if l).strip()
    if not question_part:
        question_part = text.strip()

    option_lines = [f"{key}. {value}".strip() for key, value in options.items()]
    if option_lines:
        user_prompt = (
            f"{question_part}\n" + "\n".join(option_lines)
            if question_part
            else "\n".join(option_lines)
        )
    else:
        user_prompt = question_part

    return user_prompt.strip(), options, trailing_answer


def normalize_answer_label(label: str) -> str:
    cleaned = (label or "").strip().upper()
    if not cleaned:
        return ""
    return cleaned[0]


def build_table(
    samples: List[Dict[str, Any]],
    schema: pa.Schema,
    system_prompt: str,
    reward_style: str,
    split: str,
) -> pa.Table:
    prompt_type = schema.field("prompt").type
    reward_type = schema.field("reward_model").type
    extra_type = schema.field("extra_info").type

    ids: List[int] = []
    datasets: List[str] = []
    data_sources: List[str] = []
    abilities: List[str] = []
    prompts: List[List[Dict[str, str]]] = []
    rewards: List[Dict[str, str]] = []
    extras: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples):
        conversations = sample.get("conversations") or []
        if not conversations:
            raise ValueError(f"Sample {sample.get('id')} missing conversations.")
        question_text = (conversations[0].get("value") or "").strip()
        if not question_text:
            raise ValueError(f"Sample {sample.get('id')} missing question text.")

        user_prompt, option_map, trailing_answer = split_question_and_options(
            question_text
        )
        answer_label = normalize_answer_label(
            (sample.get("eval") or {}).get("answer") or trailing_answer
        )
        if not answer_label and trailing_answer:
            answer_label = normalize_answer_label(trailing_answer)
        if not answer_label:
            raise ValueError(f"Sample {sample.get('id')} missing answer label.")

        answer_text = option_map.get(answer_label, "").strip()
        ground_truth = (
            f"{answer_label}. {answer_text}" if answer_text else answer_label
        )

        prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({"content": system_prompt, "role": "system"})
        prompt_messages.append({"content": user_prompt, "role": "user"})

        try:
            sample_id = int(sample.get("id"))
        except (TypeError, ValueError):
            sample_id = index

        ids.append(sample_id)
        datasets.append("medmcqa")
        data_sources.append("medmcqa")
        abilities.append("medmcqa")
        prompts.append(prompt_messages)
        rewards.append({"ground_truth": ground_truth, "style": reward_style})
        extras.append(
            {
                "answer": {"answer": answer_text, "answer_idx": answer_label},
                "index": sample_id,
                "question": user_prompt,
                "split": split,
                "options": option_map,
                "subject": (sample.get("eval") or {}).get("type"),
            }
        )

    num_rows = len(samples)
    arrays = []
    for name in schema.names:
        field = schema.field(name)
        if name == "id":
            arrays.append(pa.array(ids, type=field.type))
        elif name == "instruction":
            arrays.append(pa.nulls(num_rows, type=field.type))
        elif name == "dataset":
            arrays.append(pa.array(datasets, type=field.type))
        elif name == "data_source":
            arrays.append(pa.array(data_sources, type=field.type))
        elif name == "prompt":
            arrays.append(pa.array(prompts, type=prompt_type))
        elif name == "ability":
            arrays.append(pa.array(abilities, type=field.type))
        elif name == "reward_model":
            arrays.append(pa.array(rewards, type=reward_type))
        elif name == "extra_info":
            arrays.append(pa.array(extras, type=extra_type))
        else:
            raise ValueError(f"Unexpected field '{name}' in schema.")

    return pa.Table.from_arrays(arrays, schema=schema)


def main() -> None:
    args = parse_args()
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference Parquet not found: {args.reference}")

    samples = read_source_json(args.input)
    schema, system_prompt, reward_style, split = read_reference_metadata(
        args.reference
    )
    table = build_table(samples, schema, system_prompt, reward_style, split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"Wrote {table.num_rows} rows to {args.output}")


if __name__ == "__main__":
    main()
