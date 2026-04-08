from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


TARGET_CATEGORIES = {"health", "biology"}
DATASET_NAME = "mmlupro_med"
DATA_SOURCE = "mmlupro"
ABILITY_NAME = "medical"
REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "medqa" / "test.parquet"
# OUTPUT_PATH = Path.home() / "datasets" / "medical_test" / "mmlu_pro_med.parquet"
OUTPUT_PATH = Path("data/mmlu_pro_med/test.parquet")


def read_reference_metadata(path: Path) -> Tuple[pa.Schema, str, str, str]:
    schema = pq.read_schema(path)
    parquet_file = pq.ParquetFile(path)
    sample_table = parquet_file.read_row_group(
        0, columns=["prompt", "reward_model", "extra_info"]
    )
    sample_row = sample_table.slice(0, 1).to_pylist()[0]

    system_prompt = ""
    for message in sample_row.get("prompt") or []:
        if isinstance(message, dict) and message.get("role") == "system":
            system_prompt = message.get("content", "")
            break

    reward_style = (sample_row.get("reward_model") or {}).get("style") or "rule"
    default_split = (sample_row.get("extra_info") or {}).get("split") or "test"
    return schema, system_prompt, reward_style, default_split


def normalize_label(label: Any, ordered_labels: Sequence[str]) -> str:
    raw = (str(label or "").strip()).upper()
    if not raw:
        return ""
    if raw.isdigit():
        idx = int(raw)
        for offset in (0, 1):
            adjusted = idx - offset
            if 0 <= adjusted < len(ordered_labels):
                return ordered_labels[adjusted]
    return raw[0]


def build_prompt_and_options(
    question: str, raw_options: Any
) -> Tuple[str, Dict[str, str], List[str]]:
    prompt_parts: List[str] = []
    clean_question = (question or "").strip()
    if clean_question:
        prompt_parts.append(clean_question)

    option_map: Dict[str, str] = {}
    ordered_labels: List[str] = []

    if isinstance(raw_options, Mapping):
        option_items: Iterable[Tuple[Any, Any]] = sorted(raw_options.items())
    else:
        option_items = list(enumerate(raw_options or []))

    for idx, option_value in option_items:
        text = (str(option_value or "").strip())
        if isinstance(raw_options, Mapping):
            label = (str(idx or "").strip() or "").upper()
            if not label or not label[0].isalpha():
                label = chr(ord("A") + len(option_map))
            else:
                label = label[0]
        else:
            label = chr(ord("A") + int(idx))
        ordered_labels.append(label)
        option_map[label] = text
        if text:
            prompt_parts.append(f"{label}. {text}")

    prompt_text = "\n".join(part for part in prompt_parts if part).strip()
    return prompt_text, option_map, ordered_labels


def build_table(
    records: Sequence[Dict[str, Any]],
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

    for idx, record in enumerate(records):
        prompt_text, option_map, ordered_labels = build_prompt_and_options(
            record.get("question", ""), record.get("options") or []
        )
        answer_label = normalize_label(record.get("answer"), ordered_labels)
        if not answer_label:
            raise ValueError(f"Sample {idx} missing answer label")
        if answer_label not in option_map:
            raise ValueError(f"Answer label '{answer_label}' missing in options for sample {idx}")

        answer_text = option_map.get(answer_label, "")
        prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({"content": system_prompt, "role": "system"})
        prompt_messages.append({"content": prompt_text, "role": "user"})

        ids.append(idx)
        datasets.append(DATASET_NAME)
        data_sources.append(DATA_SOURCE)
        abilities.append(ABILITY_NAME)
        prompts.append(prompt_messages)
        rewards.append(
            {
                "ground_truth": f"{answer_label}. {answer_text}" if answer_text else answer_label,
                "style": reward_style,
            }
        )
        extras.append(
            {
                "answer": {"answer": answer_text, "answer_idx": answer_label},
                "index": idx,
                "question": record.get("question"),
                "split": split,
                "category": record.get("category"),
            }
        )

    num_rows = len(records)
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
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")['test']
    filtered_records = []

    categories = dataset["category"]
    indices = [idx for idx, category in enumerate(categories) if category in TARGET_CATEGORIES]

    filtered_records.extend(dataset.select(indices).to_list())

    if not filtered_records:
        raise RuntimeError(f"No samples found for categories: {sorted(TARGET_CATEGORIES)}")

    schema, system_prompt, reward_style, split = read_reference_metadata(REFERENCE_PATH)
    table = build_table(filtered_records, schema, system_prompt, reward_style, split)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUTPUT_PATH)


if __name__ == "__main__":
    main()
