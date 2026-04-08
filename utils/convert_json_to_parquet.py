from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INPUT = Path(
    "/mnt/petrelfs/jiangshuyang/datasets/medical_train/medmcqa_llama31_filter.json"
)
DEFAULT_REFERENCE = Path("data/medqa/train.parquet")
DEFAULT_OUTPUT = Path("data/medmcqa/train.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedMCQA JSON data into the MedQA Parquet schema."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the MedMCQA JSON file.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference Parquet file whose schema will be reused.",
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
        raise ValueError(f"Expected a list of samples in {path}, got {type(payload)}")
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

    for idx, sample in enumerate(samples):
        conversations = sample.get("conversations") or []
        if len(conversations) < 2:
            raise ValueError(f"Sample {sample.get('id')} missing conversations field")
        question = conversations[0].get("value", "").strip()
        answer_text = conversations[1].get("value", "").strip()
        answer_idx = (sample.get("answer_idx") or "").strip()
        if not question:
            raise ValueError(f"Sample {sample.get('id')} missing question/answer text")
        if answer_text == "":
            answer_text = answer_idx
        prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({"content": system_prompt, "role": "system"})
        prompt_messages.append({"content": question, "role": "user"})

        ground_truth = f"{answer_idx}. {answer_text}" if answer_idx else answer_text

        ids.append(idx)
        datasets.append("medmcqa")
        data_sources.append("medmcqa")
        abilities.append("medmcqa")
        prompts.append(prompt_messages)
        rewards.append({"ground_truth": ground_truth, "style": reward_style})
        extras.append(
            {
                "answer": {"answer": answer_text, "answer_idx": answer_idx},
                "index": idx,
                "question": question,
                "split": split,
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
            raise ValueError(f"Unexpected field '{name}' in target schema")

    return pa.Table.from_arrays(arrays, schema=schema)


def main() -> None:
    args = parse_args()
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference Parquet file not found: {args.reference}")

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
