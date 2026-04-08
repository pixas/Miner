import torch
from tqdm import tqdm 
from evaluation.models.base_model import Local_Model
from utils.s3_client import client
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn.functional as F


def print_gpu_mem():
    alloc = torch.cuda.memory_allocated() / 1024**3
    res = torch.cuda.memory_reserved() / 1024**3
    print(f"[GPU] 已分配: {alloc:.3f} GB, 保留: {res:.3f} GB")
    
def encode_response( data_item, model: Local_Model):

    output = data_item['output']
    output_results = model.tokenizer(output, return_tensors="pt")
    output_ids = output_results['input_ids'].to(model.device)
    with torch.no_grad():
        output_emb = model.model(output_ids,output_hidden_states=True)
    results = output_emb.hidden_states[-1].mean(1)
    # output_emb.hidden_states = None 
    return results


def compute_similarity(rollout_data, prompt_id, model: Local_Model):
    data_item = rollout_data[prompt_id]
    # find all items that share the same problem 
    problem = data_item['input']
    same_problem_items = [item for item in rollout_data if item['input'] == problem]
    embeddings = []
    # print_gpu_mem()
    for item in tqdm(same_problem_items, total=len(same_problem_items)):
        emb = encode_response(item, model)
        embeddings.append(emb)
        # torch.cuda.empty_cache()
        # print_gpu_mem()
    embeddings = torch.cat(tensors=embeddings, dim=0).float()
    # sentence embeddings mean 
    x_normalized = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(x_normalized, x_normalized.t())
    # sim_matrix = (sim_matrix + 1) / 2  # Convert from [-1, 1] to [0, 1]
    return sim_matrix, same_problem_items
    


if __name__ == "__main__":
    data = client.read("/mnt/petrelfs/jiangshuyang/repo/efficient_RL/rollout_data/verl_math/deepscaler_qwen34b_grpo/41.jsonl")
    
    model = Local_Model("/mnt/phwfile/medai_p/LLMModels/LLMs/Qwen3-4B-Base")
    sim_matrix, same_problem_items = compute_similarity(data, 81, model)
    sim_matrix = sim_matrix.float().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: bar chart of scores
    scores = [item['score'] for item in same_problem_items]
    ax1.bar(range(len(scores)), scores)
    ax1.set_xlabel('Item Index')
    ax1.set_ylabel('Score')
    ax1.set_title('Scores of Same Problem Items')

    # Right subplot: heatmap of similarity matrix
    im = ax2.imshow(sim_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Item Index')
    ax2.set_ylabel('Item Index')
    ax2.set_title('Similarity Matrix Heatmap')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    # plt.show()
    plt.savefig("similarity_heatmap.png")
    correct_index=[i for i, n in enumerate(same_problem_items) if n['score']==1]
    incorrect_sample_with_correct_sim = {}
    for i in range(16):
        if i in correct_index:
            print(f"{i}-th correct sample's similarity with all correct samples")
            incorrect_sample_with_correct_sim[i] = sim_matrix[i,correct_index].mean()
            print(incorrect_sample_with_correct_sim[i])
    # 越高越好，越高基于越小的权重，说明这个错误的sample和正确的sample越相似，惩罚力度越轻
    # 计算归一化的相似度，这样可以区分哪个是最好的
    # min_all = min(list(incorrect_sample_with_correct_sim.values()))
    # max_all = max(list(incorrect_sample_with_correct_sim.values()))
    # incorrect_sample_with_correct_sim = {k: (v-min_all) / (max_all - min_all) for k, v in incorrect_sample_with_correct_sim.items()}
    # print(incorrect_sample_with_correct_sim)
    # # linear scale 
    # print("linear scale")
    # linear_scale = {k: 1 - v for k, v in incorrect_sample_with_correct_sim.items()}
    # print(linear_scale)
    # print("exponential scale")
    # exp_scale = {k: np.exp(-v) for k, v in incorrect_sample_with_correct_sim.items()}
    # print(exp_scale)