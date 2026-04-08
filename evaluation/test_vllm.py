from rollout.vllm_rollout import VLLMRollout


model = VLLMRollout("r1distill-qwen-1.5b", response_length=8192)
