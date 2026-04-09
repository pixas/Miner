# MINER

Official code repository for our ACL 2026 Main paper: **MINER: Mining Intrinsic Mastery for Data-Efficient RL in Large
Reasoning Models**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](#environment)
[![Framework](https://img.shields.io/badge/RL-verl-orange)](#training)
[![Backend](https://img.shields.io/badge/Rollout-vLLM-green)](#training)

📄 Paper: [ACL 2026 Main Paper](https://arxiv.org/pdf/2601.04731)  
🤗 Hugging Face: [Miner-4B](https://huggingface.co/pixas/Miner-4B)  
🤗 Hugging Face: [Miner-8B](https://huggingface.co/pixas/Miner-8B)  
📝 Project Page: [Coming Soon](<project-page-link>)

MINER is built on top of `verl` and focuses on reinforcement learning for better exploration capability and higher training efficiency. This repository includes:

- our custom advantage estimator `miner`
- a `verl`-style PPO/GRPO training entry
- evaluation scripts for dataset-level metrics and cached generations

## Highlights

- Built on `verl` with `vLLM` rollout.
- Supports baseline estimators such as `grpo` and our method `miner`.
- Uses a simple shell entry for training: [`scripts/train.sh`](scripts/train.sh).
- Uses [`evaluation/test.py`](evaluation/test.py) for benchmark evaluation.

## Repository Layout

```text
.
|-- baselines/          # custom trainer, configs, and MINER implementation
|-- evaluation/         # evaluation entrypoints, scorers, prompts, model wrappers
|-- rollout/            # rollout backends
|-- scripts/train.sh    # main training launcher
|-- utils/              # data conversion and analysis utilities
|-- verl/               # local verl code used by this project
```

## Environment

Our experiments are run in the `miner` conda environment.

```bash
conda activate miner
python --version
```

The environment used for this repository is:

- Python `3.10.18`
- `verl==0.5.0.dev0`
- `vllm==0.8.5.post1`
- `ray==2.49.2`
- `transformers==4.56.2`
- `torch==2.6.0`
- `pandas==2.3.3`
- `pyarrow==21.0.0`
- `wandb==0.22.1`

If you need to recreate the environment from scratch, make sure your new environment matches the versions above and can run both `verl` and `vLLM` correctly. All commands below should be launched from the repository root after activating `miner`.

If you use Weights & Biases logging, log in first:

```bash
wandb login
```

If you do not want online logging, disable it before training:

```bash
export WANDB_MODE=disabled
```

## Data Format

Training uses file paths passed through environment variables:

- `train_files`: training parquet file
- `test_files`: validation parquet file

Evaluation and train files accept a parquet file with fields compatible with this repository's scorer, including `data_source`, `prompt`, and `reward_model.ground_truth`.
We use [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) as the train dataset and six evaluation datasets as main testbeds. We follow [DECS](https://github.com/pixas/DECS/tree/main/data) to construct data.

## Training

The main launcher is [`scripts/train.sh`](scripts/train.sh).

### Usage

```bash
bash scripts/train.sh \
  <data_name> \
  <adv_estimator> \
  <model_name_or_path> \
  <save_name> \
  [rollout_n=16] \
  [entry_name=baselines.baseline_main_ppo] \
  [additional hydra overrides ...]
```

### Required Setup

Before running the script, export the dataset paths:

```bash
export train_files=/path/to/train.parquet
export test_files=/path/to/val.parquet
```

We use `DeepScaleR` training set and `AIME 2024` as dev set.

### Important Arguments

- `data_name`: dataset alias used in the experiment name
- `adv_estimator`: RL estimator, e.g. `grpo` or `miner`
- `model_name_or_path`: base model path or Hugging Face model id
- `save_name`: suffix used in `trainer.experiment_name`
- `rollout_n`: number of rollout samples per prompt, default `16`
- `entry_name`: Python module entry, default `baselines.baseline_main_ppo`

### Example: Train MINER

```bash
export train_files=/path/to/deepscaler/train.parquet
export test_files=/path/to/aime24/test.parquet

bash scripts/train.sh \
  medqa \
  miner \
  /path/to/base_model \
  miner_main \
  16 \
  baselines.baseline_main_ppo "data.is_base=True trainer.total_epochs=2 trainer.val_before_train=False data.max_response_length=8192 data.prompt_type=base actor_rollout_ref.rollout.max_num_batched_tokens=18000 reward_model.reward_manager=naive actor_rollout_ref.actor.use_kl_loss=True data.train_batch_size=128 actor_rollout_ref.actor.ppo_mini_batch_size=128 algorithm.miner.incorrect_scale=0 algorithm.miner.clip_high=0 algorithm.miner.credit_method=focal algorithm.miner.scale_by_normal=0.0015"
```

### Example: Train a GRPO Baseline

```bash
export train_files=/path/to/deepscaler/train.parquet
export test_files=/path/to/aime24/test.parquet

bash scripts/train.sh \
  deepscaler \
  grpo \
  /path/to/base_model \
  grpo_baseline 
```

### Default Training Notes

The current shell launcher is configured with the following defaults:

- `NUM_GPUS=8`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
- `trainer.logger=['console','wandb']`
- `trainer.project_name='verl_math'`

If your machine setup is different, edit [`scripts/train.sh`](scripts/train.sh) or pass extra Hydra overrides at the end of the command.

## Evaluation

The main evaluation entry is [`evaluation/test.py`](evaluation/test.py).

### Basic Usage
First conver the fsdp checkpoints to huggingface formats:
```bash 
python -m evaluation.utils.model_merger $source_dir $target_dir
```

Then run evaluation locally:
```bash
python evaluation/test.py \
  --data_path /path/to/test.parquet \
  --output_path /path/to/eval_outputs \
  --model_name_or_path $target_dir \
  --prompt_type base_math \
  --use_vllm \
  --batch 8 \
  --temperature 0.7 \
  --max_new_tokens 8192 \
  --resume
```

### Common Arguments

- `--data_path`: input dataset in `.json` or `.parquet`
- `--output_path`: directory for evaluation artifacts
- `--model_name_or_path`: model path or model id. Should be converted from the saved fsdp checkpoints.
- `--prompt_type`: prompt template used for evaluation
- `--use_vllm`: enable vLLM inference backend
- `--batch`: evaluation batch size
- `--sample_num`: number of samples per example
- `--resume`: resume from an existing `cache.jsonl`
- `--hf_model_path`: optional Hugging Face checkpoint path
- `--peft_path`: optional PEFT adapter path

### Supported `prompt_type`

```text
base_math
instruct_default
```
where `base_math` is for evaluation of base models and `instruct_default` is for evaluation of instruct models

### Evaluation Outputs

For each run, the script writes:

- `cache.jsonl`: per-example generations, latency, token counts, and scores
- `result.json`: aggregated metrics, sample count, timing, token statistics, and full run arguments



## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{jiang2026miner,
  title={Miner: Mining Intrinsic Mastery for Data-Efficient RL in Large Reasoning Models},
  author={Jiang, Shuyang and Wang, Yuhao and Zhang, Ya and Wang, Yanfeng and Wang, Yu},
  journal={arXiv preprint arXiv:2601.04731},
  year={2026}
}
```
