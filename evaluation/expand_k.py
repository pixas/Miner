
# 写一个脚本，可以接受一个路径，这个路径类似如下
# /mnt/petrelfs/jiangshuyang/repo/efficient_RL/results/aime2024/deepscaler_qwen34b_grpo_bapo_0.8_0.95_1.2_2.0_target_0.5_global_step_314_sc128
# 路径下，有result.json，里面存放了模型的路径名，用来加载模型用，以及cache.jsonl，是模型在数据集上的输出，每个问题有N个轨迹
# 现在，这个脚本需要扩大每个问题的输出数量， 比如原来每个问题有5个轨迹，现在需要扩展到N个轨迹，N通过命令行输入
# 仿照test.py中对应的逻辑，加载模型，对每个问题进行推理，直到每个问题有N个轨迹为止
# 当然，需要服用原本就有的K个轨迹，不然有点浪费
import os
import json
import argparse
from collections import defaultdict
import time
from typing import Dict, List, Any
import sys 
import math
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Ensure the parent directory is also in sys.path if evaluation is a subdirectory
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from evaluation.models import get_model
from evaluation.utils.prompt import get_prepare_func
from evaluation.eval.score import score_task
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Expand per-problem trajectories to target K using saved result.json + cache.jsonl")
    parser.add_argument("result_dir", type=str, help="Directory containing result.json and cache.jsonl")
    parser.add_argument("--target_k", "-k", type=int, required=True, help="Target number of trajectories per problem")
    parser.add_argument("--batch_size", type=int, default=32, help="Generation batch size (number of samples per rollout call)")
    parser.add_argument("--cache_file", type=str, default="cache.jsonl", help="Cache file name inside result_dir")
    parser.add_argument("--result_file", type=str, default="result.json", help="Result file name inside result_dir")
    parser.add_argument("--chunk_num", type=int, default=1, help="Total number of chunks to split pending tasks into")
    parser.add_argument("--chunk_idx", type=int, default=0, help="0-based chunk index to process")
    return parser.parse_args()


def load_result_args(result_path: str) -> Dict[str, Any]:
    with open(result_path, "r") as f:
        data = json.load(f)
    # Prefer nested args if present
    if isinstance(data, dict) and "args" in data and isinstance(data["args"], dict):
        return data["args"].copy()
    return data if isinstance(data, dict) else {}


def read_cache(cache_path: str) -> List[Dict[str, Any]]:
    logs = []
    if not os.path.exists(cache_path):
        return logs
    with open(cache_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def write_cache(cache_path: str, logs: List[Dict[str, Any]], overwrite=False):
    mode = 'w' if overwrite else 'a'
    with open(cache_path, mode=mode) as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False, separators=(",", ": ")) + "\n")


def recompute_result_json(result_path: str, logs: List[Dict[str, Any]], old_args: Dict[str, Any]):
    count = len(logs)
    if count == 0:
        summary = {"score": {}, "count": 0, "time": {"avg": 0, "total": 0}, "tokens": {"avg": 0}, "args": old_args}
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2, separators=(",", ": "))
        return

    # ensure scores exist and aggregate
    for log in tqdm(logs, total=len(logs), desc='Evaluating:'):
    # if "score" not in log:
        log["score"] = score_task(log)
    metrics = list(logs[0]["score"].keys()) if logs[0].get("score") else []
    score_sum = {m: 0.0 for m in metrics}
    for log in logs:
        for m in metrics:
            score_sum[m] += float(log["score"].get(m, 0.0))
    score_avg = {m: (score_sum[m] / count if count > 0 else 0.0) for m in metrics}

    avg_time = sum([log.get("time", 0) for log in logs]) / count if count > 0 else 0.0
    total_time = sum([log.get("time", 0) for log in logs])
    avg_tokens = 0.0
    if count > 0:
        per_log_avg = []
        for log in logs:
            toks = log.get("tokens", [])
            if toks:
                per_log_avg.append(sum(toks) / len(toks))
        avg_tokens = (sum(per_log_avg) / len(per_log_avg)) if per_log_avg else 0.0

    summary = {"score": score_avg, "count": count, "time": {"avg": avg_time, "total": total_time}, "tokens": {"avg": avg_tokens}, "args": old_args}
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, separators=(",", ": "))


def main():
    args = parse_args()
    if args.chunk_num < 1:
        raise ValueError("chunk_num must be >= 1")
    if args.chunk_idx < 0 or args.chunk_idx >= args.chunk_num:
        raise ValueError("chunk_idx must satisfy 0 <= chunk_idx < chunk_num")
    target_k = int(args.target_k)
    result_dir = args.result_dir
    cache_path = os.path.join(result_dir, args.cache_file)
    result_path = os.path.join(result_dir, args.result_file)

    # Load saved run args and cache
    saved_args = load_result_args(result_path) if os.path.exists(result_path) else {}
    sc_num = result_dir.split("_sc")[-1].split("_")[0]
    save_dir = result_dir.replace(f"sc{sc_num}", f"sc{target_k}")
    os.makedirs(save_dir, exist_ok=True)
    cache_filename = args.cache_file
    result_filename = args.result_file
    if args.chunk_num > 1:
        chunk_suffix = f"chunk{args.chunk_idx}of{args.chunk_num}"
        cache_root, cache_ext = os.path.splitext(args.cache_file)
        result_root, result_ext = os.path.splitext(args.result_file)
        cache_filename = f"{cache_root}_{chunk_suffix}{cache_ext}"
        result_filename = f"{result_root}_{chunk_suffix}{result_ext}"
    save_cache_path = os.path.join(save_dir, cache_filename)
    save_result_path = os.path.join(save_dir, result_filename)

    chunk_cache_exists = os.path.exists(save_cache_path)
    load_cache_path = os.path.join(save_dir, args.cache_file)
    # load_cache_path = save_cache_path if chunk_cache_exists else os.path.join(save_dir, args.cache_file)
    # load_cache_path = os.path.join(save_dir, cache_filename)
    original_logs = read_cache(cache_path)
    # previous_save_logs = read_cache(os.path.join(save_dir, args.cache_file))
    logs = read_cache(load_cache_path)

    if not logs:
        print(f"No cache logs found at: {load_cache_path}")
        logs = read_cache(cache_path)
    else:
        # pending logs should be the log term that ids in original logs and not in logs 
        pending_logs = []
        
        logs_current_ids = set([item['task']['id'] for item in logs])
        for log in original_logs:
            if log['task']['id'] in logs_current_ids:
                continue
            pending_logs.append(log)
        if not pending_logs:
            print("All tasks already have at least target_k trajectories. Nothing to do.")
            recompute_result_json(save_result_path, logs, saved_args)
            write_cache(save_cache_path, logs, overwrite=True)
            return
        logs = pending_logs
        
    # for item in original_logs:
    #     if item['task']['id'] in logs_current_ids:
    #         continue 
    #     else:
    #         logs.append(item)
    # todo: 按照item['task']['id']从小到大排序logs，id是一个xxx_0, xxx_1的格式
    def _task_id_key(entry: Dict[str, Any]):
        task = entry.get("task", {})
        tid = task.get("id")
        if isinstance(tid, str):
            head, sep, tail = tid.rpartition("_")
            if sep:
                try:
                    idx = int(tail)
                except ValueError:
                    idx = float("inf")
                return (head, idx, tid)
            return (tid, -1, tid)
        return ("", float("inf"), "")

    logs.sort(key=_task_id_key)
    
    if args.chunk_num > 1 and not chunk_cache_exists:
        pending_logs = []
        for log in logs:
            task = log.get("task", {})
            tid = task.get("id")
            if tid is None:
                continue
            traj = log.get("trajectory", []) or []
            pass_at_k = any(log.get("all_acc"))
            if len(traj) < target_k and (not pass_at_k):
                pending_logs.append(log)
        if not pending_logs:
            print("All tasks already satisfy target_k trajectories. Nothing to expand.")
            return
        chunk_size = math.ceil(len(pending_logs) / args.chunk_num)
        start_idx = chunk_size * args.chunk_idx
        if start_idx >= len(pending_logs):
            print(f"Chunk {args.chunk_idx}/{args.chunk_num} has no pending tasks.")
            return
        end_idx = min(len(pending_logs), start_idx + chunk_size)
        logs = pending_logs[start_idx:end_idx]
        print(f"Processing chunk {args.chunk_idx + 1}/{args.chunk_num}: {len(logs)} / {len(pending_logs)} tasks.")
    elif args.chunk_num > 1:
        pending_logs = []
        for log in logs:
            task = log.get("task", {})
            tid = task.get("id")
            if tid is None:
                continue
            traj = log.get("trajectory", []) or []
            if len(traj) < target_k:
                pending_logs.append(log)
        if not pending_logs:
            print("All tasks already satisfy target_k trajectories. Nothing to expand.")
            return
        chunk_size = math.ceil(len(pending_logs) / args.chunk_num)
        start_idx = chunk_size * args.chunk_idx
        if start_idx >= len(pending_logs):
            print(f"Chunk {args.chunk_idx}/{args.chunk_num} has no pending tasks.")
            return
        end_idx = min(len(pending_logs), start_idx + chunk_size)
        logs = pending_logs[start_idx:end_idx]
        already_processed_logs = read_cache(save_cache_path)
        already_saved_ids = set([item['task']['id'] for item in already_processed_logs])
        new_logs = []
        for log in logs:
            if log['task']['id'] in already_saved_ids:
                continue 
            else:
                new_logs.append(log)
        logs = new_logs
        print(f"Processing chunk {args.chunk_idx + 1}/{args.chunk_num}: {len(logs)} / {len(pending_logs)} tasks.")

    
    model_name = saved_args.get("model_name_or_path") or saved_args.get("model") or saved_args.get("model_path")
    if not model_name:
        raise ValueError("Cannot infer model path/name from result.json 'args'.")
    peft_path = saved_args.get("peft_path")
    use_vllm = bool(saved_args.get("use_vllm", True))
    prompt_type = saved_args.get("prompt_type", "base_default")
    temperature = float(saved_args.get("temperature", 1.0))
    max_new_tokens = int(saved_args.get("max_new_tokens", 2048))
    
    prepare_func = get_prepare_func(model_name, prompt_type, None)

    # Figure out how many more samples are needed per task
    need_per_id: Dict[str, int] = {}
    prompt_per_id: Dict[str, str] = {}
    for log in logs:
        task = log.get("task", {})
        tid = task.get("id")
        traj = log.get("trajectory", []) or []
        have = len(traj)
        pass_at_k = any(log.get("all_acc"))
        need = max(0, target_k - have)
        if tid is None or need == 0:
            continue
        if pass_at_k:
            
            write_cache(save_cache_path, [log])
            continue
        # Build prompt same as evaluation/test.py
        prepared = prepare_func(task).get("prompt")
        prompt_text = prepared if isinstance(prepared, str) else prepared[0]["content"]
        need_per_id[tid] = need
        prompt_per_id[tid] = prompt_text

    total_needed = sum(need_per_id.values())
    if total_needed == 0:
        print("All tasks already have at least target_k trajectories. Nothing to do.")
        recompute_result_json(save_result_path, logs, saved_args)
        write_cache(save_cache_path, logs, overwrite=True)
        return
    # Determine model + prompt settings
    
    

    # Build model and tokenizer
    Model = get_model(model_name, peft_path=peft_path, use_vllm=use_vllm, max_new_tokens=max_new_tokens, temperature=1.0, hf_model_path=saved_args.get("hf_model_path"))
    tokenizer = getattr(Model, "tokenizer", None)
    wrap_chat = ("instruct" in prompt_type)
    
    # Build expansion list and generate in batches
    expansions: List[str] = []
    expansion_owner: List[str] = []
    # todo: 163行打印的logs数量只有1/num_chunks * total，但是need_per_id会把所有的id都打印出来，这是为什么
    print(need_per_id)
    for tid, need in need_per_id.items():
        prompt_text = prompt_per_id[tid]
        for _ in range(need):
            expansions.append(prompt_text)
            expansion_owner.append(tid)

    batch_size = max(1, int(args.batch_size))
    start_time = time.time()
    progress = tqdm(total=len(expansions), desc="Expanding trajectories", ncols=80)
    idx = 0
    last_idx = 0
    id2log = None
    completed_tids = set()
    for log in logs:
        tid = log.get("task", {}).get("id")
        if tid is None:
            continue
        if len((log.get("trajectory") or [])) >= target_k:
            completed_tids.add(tid)
    # 如果某个question的target_k个轨迹已经生成完毕，及时save
    # 然后支持resume，允许从save_cache_path中读取已经生成的序列
    while idx < len(expansions):
        batch_prompts: List[str] = []
        batch_owners: List[str] = []
        while idx < len(expansions) and len(batch_prompts) < batch_size:
            owner = expansion_owner[idx]
            if owner in completed_tids:
                idx += 1
                continue
            batch_prompts.append(expansions[idx])
            batch_owners.append(owner)
            idx += 1

        progress.update(idx - last_idx)
        last_idx = idx

        if not batch_prompts:
            break

        batch_results = Model.rollout(
            batch_prompts,
            do_sample=True if len(batch_prompts) > 0 else False,
            wrap_chat=wrap_chat,
            n=1
        )
        texts = batch_results['response_text']
        tokens = batch_results['response_ids']
        # Append outputs back to corresponding logs
        if id2log is None:
            id2log = {log["task"]["id"]: log for log in logs if "task" in log and "id" in log["task"]}

        newly_completed = False
        for owner, text, token_id_list in zip(batch_owners, texts, tokens):
            log = id2log.get(owner)
            if log is None:
                continue
            log.get("trajectory", []).append(text)
            # tokens
            tok_len = len(token_id_list)
            # if tokenizer is not None:
            #     tok_len = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
            # else:
            #     tok_len = len(text.split()) if text else 0
            log.get("tokens", []).append(tok_len)
            # keep time as previous per-problem avg or set a tiny increment; we won't change here
            if owner not in completed_tids and len(log.get("trajectory", [])) >= target_k:
                completed_tids.add(owner)
                newly_completed = True
                write_cache(save_cache_path, [log])

        # progress.update(len(texts))
        # for log in logs:
        #     log["score"] = score_task(log)
        # if newly_completed:
        #     write_cache(save_cache_path, logs)
    progress.close()

    # Recompute score for updated logs and save
    # for log in logs:
    #     log["score"] = score_task(log)


    # write_cache(save_cache_path, logs)

    # Update result.json summary

    recompute_result_json(save_result_path, logs, saved_args)
    elapsed = time.time() - start_time
    print(f"Expanded {total_needed} new trajectories across {len(need_per_id)} tasks in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
