from transformers import AutoTokenizer 
import matplotlib.pyplot as plt 
import numpy as np 
import json
from scipy.optimize import curve_fit
tokenizer = AutoTokenizer.from_pretrained("/mnt/phwfile/medai_p/LLMModels/LLMs/Qwen3-4B-Base")

def show_length_with_pass1(response_data, tokenizer):
    """compute the max response length and corresponding pass@1 scores for samples which contains at least one correct sample

    Args:
        response_data (list[dict]): contains 'input' and 'response' keys, where 'input' may be the same across different samples
        tokenizer (_type_): _description_
    """
    same_problem_response = {}
    for item in response_data:
        problem = item['input']
        if problem not in same_problem_response:
            same_problem_response[problem] = []
        same_problem_response[problem].append(item)

    max_response_length = []
    pass_at_1 = []
    for problem, responses in same_problem_response.items():
        whether_correct = any([r['score'] == 1.0 for r in responses])
        if not whether_correct:
            continue
        all_responses_text = [r['output'] for r in responses]
        all_responses_length = [len(tokenizer(r)['input_ids']) for r in all_responses_text]
        max_length = min([l for score, l in zip([r['score'] for r in responses], all_responses_length) if score == 1.0])
        max_response_length.append(max_length)
        cur_pass_at_1 = sum([r['score'] for r in responses]) / len(responses)
        pass_at_1.append(cur_pass_at_1)
    return max_response_length, pass_at_1

def plot_different_length_info(iters, base_path):
    each_iter_info = {}
    for it in iters:
        with open(f"{base_path}/{it}.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        max_response_length, pass_at_1 = show_length_with_pass1(data, tokenizer)
        each_iter_info[it] = {
            "max_response_length": max_response_length,
            "pass_at_1": pass_at_1
        }
    # plot with differnt subplot
    fig, axs = plt.subplots(len(iters), 2, figsize=(12, 6 * len(iters)))
    for i, it in enumerate(iters):
        axs[i, 0].scatter(each_iter_info[it]['max_response_length'], each_iter_info[it]['pass_at_1'], alpha=0.6)
        axs[i, 0].set_xlabel('Min Response Length')
        axs[i, 0].set_ylabel('Pass@1 Score')
        axs[i, 0].set_title(f'Iteration {it}: Max Response Length vs Pass@1 Score')
        # 用负sigmoid函数y=-\alpha/(1+exp(-\beta x))+1作为拟合曲线进行拟合
        x = np.array(each_iter_info[it]['max_response_length'])
        y = np.array(each_iter_info[it]['pass_at_1'])
        def neg_sigmoid(x, alpha, beta):
            return -1 / (1 + np.exp(-beta * (x-alpha))) + 1
        
        try:
            popt, _ = curve_fit(neg_sigmoid, x, y, bounds=(0, [1000, 1.0]), maxfev=10000)
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = neg_sigmoid(x_fit, *popt)
            axs[i, 0].plot(x_fit, y_fit, color='red', label=f'Fit: -1/(1+exp(-{popt[1]:.2f}(x-{popt[0]:.2f}))) + 1')
            axs[i, 0].legend()
        except RuntimeError:
            print(f"Could not fit curve for iteration {it}")
        axs[i, 0].grid(True)

        # plot histogram of max_response_length
        axs[i, 1].hist(each_iter_info[it]['max_response_length'], bins=30, alpha=0.7, color='blue')
        axs[i, 1].set_xlabel('Min Response Length')
        axs[i, 1].set_ylabel('Frequency')
        axs[i, 1].set_title(f'Iteration {it}: Distribution of Max Response Length')
        axs[i, 1].grid(True)
    plt.tight_layout()
    plt.savefig(f"length_vs_passat1_min_iters_{','.join(iters)}.png")

if __name__ == "__main__":


    iters = ['1', '100', '150']
    # iters = ['1', '10']
    base_path = "/mnt/petrelfs/jiangshuyang/repo/efficient_RL/rollout_data/verl_math/deepscaler_qwen34b_grpo_neg"
    plot_different_length_info(iters, base_path)
        