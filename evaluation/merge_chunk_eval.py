import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List
 
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Ensure the parent directory is also in sys.path if evaluation is a subdirectory
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.eval.score import score_task
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge chunked evaluation outputs and recompute metrics."
    )
    parser.add_argument(
        "result_dir", type=Path, help="Directory containing chunk_* sub-directories."
    )
    parser.add_argument(
        "--chunk_pattern",
        default="chunk_*",
        help="Glob pattern for chunk directories relative to result_dir.",
    )
    parser.add_argument(
        "--cache_file", default="cache.jsonl", help="Cache filename inside each chunk."
    )
    parser.add_argument(
        "--result_file", default="result.json", help="Name of the merged result file."
    )
    return parser.parse_args()


def chunk_sort_key(path: Path):
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else float("inf"), path.name)


def read_chunk_cache(cache_path: Path) -> List[Dict]:
    records = []
    with cache_path.open() as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def merge_records(chunk_dirs: List[Path], cache_name: str) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for chunk_dir in chunk_dirs:
        cache_path = chunk_dir / cache_name
        if not cache_path.is_file():
            print(f"[WARN] Missing {cache_path}, skip this chunk.", file=sys.stderr)
            continue
        for record in read_chunk_cache(cache_path):
            task_id = record.get("task", {}).get("id")
            if task_id is None or task_id in seen:
                continue
            seen.add(task_id)
            if "score" not in record:
                score_task(record)
            merged.append(record)
    return merged


def average_tokens(record: Dict) -> float:
    tokens = record.get("tokens") or []
    numeric = [tok for tok in tokens if isinstance(tok, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


def aggregate(records: List[Dict]) -> Dict:
    if not records:
        raise ValueError("No task records to aggregate.")

    totals: Dict[str, float] = {}
    for record in records:
        for metric, value in record["score"].items():
            totals[metric] = totals.get(metric, 0.0) + value
    for metric in totals:
        totals[metric] /= len(records)

    avg_time = sum(record.get("time", 0.0) for record in records) / len(records)
    total_time = sum(record.get("time", 0.0) for record in records)
    avg_tokens = sum(average_tokens(record) for record in records) / len(records)
    return {
        "score": totals,
        "count": len(records),
        "time": {"avg": avg_time, "total": total_time},
        "tokens": {"avg": avg_tokens},
    }


def load_chunk_args(chunk_dirs: List[Path], result_name: str):
    for chunk_dir in chunk_dirs:
        result_path = chunk_dir / result_name
        if not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text())
            if isinstance(data, dict) and "args" in data:
                return data["args"]
        except json.JSONDecodeError:
            continue
    return None


def main():
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"{result_dir} is not a directory.")

    chunk_dirs = sorted(
        [p for p in result_dir.glob(args.chunk_pattern) if p.is_dir()],
        key=chunk_sort_key,
    )
    if not chunk_dirs:
        raise SystemExit(f"No chunk directories found in {result_dir}.")

    records = merge_records(chunk_dirs, args.cache_file)
    if not records:
        raise SystemExit("No cache entries found across all chunks.")

    merged_cache_path = result_dir / args.cache_file
    with merged_cache_path.open("w") as fout:
        for record in records:
            fout.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ": ")) + "\n"
            )

    summary = aggregate(records)
    summary["args"] = load_chunk_args(chunk_dirs, args.result_file) or {
        "result_dir": str(result_dir),
        "chunk_pattern": args.chunk_pattern,
        "cache_file": args.cache_file,
        "result_file": args.result_file,
    }
    summary_path = result_dir / args.result_file
    summary_path.write_text(json.dumps(summary, indent=2, separators=(",", ": ")))
    print(f"Merged {len(records)} tasks into {merged_cache_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
