
# from evaluation.models.base_model import LOCAL_MODEL_PATHS
from functools import partial
DEFAULT_PROMPT = "You are a helpful assistant."
INSTRUCT_MATH = "Think the math problem step by step and output the final answer within \\boxed{}"
# SYSTEM_PROMPT = """You are a helpful assistant at solving complex medical problems.
# You should first think about the reasoning process in the mind and then directly put the answer in the <answer></answer> tags. The reasoning process and answer should be enclosed within <think> </think> and
# <answer> </answer> tags, respectively."""

SYSTEM_PROMPT = """You are a helpful assistant at solving complex medical problems.
You should first think about the reasoning process in the mind and then summarize the answer by stating 'The answer is' in the end. The reasoning process should be enclosed within <think> </think> tages. After </think> tag, output your summarized answer."""

# SYSTEM_PROMPT = "Please think step by step and output the final answer as 'The answer is '."

INSTRUCT_MED = "Please think step by step and output the final answer as 'The answer is '."

THE_ANSWER_IS_PROMPT = """You are a helpful assistant at solving complex medical problems.
You should first think about the reasoning process in the mind and then summarize the answer by stating 'The answer is' in the end.
"""

BASE_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, ensuring that the final result in the answer is given as the answer index directly. The reasoning process and answer are enclosed within '<think>' '</think>' and '<answer>' '</answer>' tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}\nAssistant: """

BASE_INFER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and outputs the final answer as 'The answer is '. \nUser: {prompt}\nAssistant: """


BASE_GSM8K_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and output the final answer after \"####\". \nUser: {prompt}\nAssistant: """



BASE_MATH_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and output the final answer within \\boxed{{}}. \nUser: {prompt}\nAssistant: """

RAW_PROMPT = """{prompt}\n"""

THINK_BASE_INFER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within '<think>' '</think>'. After </think> tag, the assistant outputs the summarized answer. \nUser: {prompt}\nAssistant: """

ORZ_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e.,
<answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

prompt_mapping = {
    "base_default": BASE_INFER_PROMPT,
    "base_think": THINK_BASE_INFER_PROMPT,
    "base_zero": BASE_PROMPT,
    "base_orz": ORZ_PROMPT,
    "base_math": BASE_MATH_PROMPT,
    "instruct_default": DEFAULT_PROMPT,
    "instruct_think": SYSTEM_PROMPT,
    "instruct_answer": THE_ANSWER_IS_PROMPT,
    "instruct_math": INSTRUCT_MATH,
    "instruct_med": INSTRUCT_MED,
    "raw": RAW_PROMPT
}

def prepare_inputs_for_instruct(sample, prompt):
    input_ = {
            "prompt": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample["input"] if "context" not in sample else f"""{sample["context"]}\n\n{sample["input"]}""" },    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    if prompt == DEFAULT_PROMPT:
        input_['prompt'] = input_['prompt'][1:]
    return input_

def prepare_inputs_for_instruct_sft(sample, prompt):
    input_ = {
            "prompt": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample["input"]},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_


def prepare_inputs_for_gsm8k_sft(sample, prompt=None):
    input_ = {
            "prompt": [
                {"role": "user", "content": sample["input"] + " Let's think step by step and output the final answer after \"####\"."},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_

def prepare_inputs_for_math_sft(sample, prompt=None):
    input_ = {
            "prompt": [
                {"role": "user", "content": sample["input"] + " Let's think step by step and output the final answer within \\boxed{}."},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_

def prepare_inputs_for_medqa_sft(sample, prompt):
    input_ = {
            "prompt": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample["input"]},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_
    
    
def prepare_inputs_for_base(sample, prompt):
    sample_raw_input = sample["input"].split("Please reason step by step")[0].strip() if "Please reason step by step" in sample["input"] else sample["input"]
    input_ = {
            "prompt": prompt.format(prompt=sample_raw_input),
            "completion": sample["output"] if "output" in sample else "",
            "label": sample["eval"]
        }
    
    return input_




def get_prepare_func(model_name_or_path, prompt_type, data_source=None):
    prompt = prompt_mapping.get(prompt_type, None)
    if prompt is None:
        raise ValueError(f"Prompt type '{prompt_type}' not found.")

    # Handle specific data sources with appropriate prompts
    if data_source:
        if "gsm8k" in data_source.lower():
            # For GSM8K, use the specialized prompt or the default one
            if prompt_type == "base_default" or prompt_type == "base_think":
                return partial(prepare_inputs_for_base, prompt=BASE_GSM8K_PROMPT)
            else:
                return prepare_inputs_for_gsm8k_sft
        elif "math" in data_source.lower() or "aime" in data_source.lower() or 'amc' in data_source.lower():
            # For Math, use the specialized math prompt
            if prompt_type == 'base_orz':
                return partial(prepare_inputs_for_base, prompt=prompt)
            if "base" in prompt_type:
                return partial(prepare_inputs_for_base, prompt=BASE_MATH_PROMPT)
            if 'raw' in prompt_type:
                return partial(prepare_inputs_for_base, prompt=RAW_PROMPT)
            else:
                return prepare_inputs_for_math_sft
        elif data_source in ["medqa", "medmcqa"]:
            return partial(prepare_inputs_for_medqa_sft, prompt=prompt)

    
    # Default behavior (when data_source is None)
    if "base" in prompt_type:
        return partial(prepare_inputs_for_base, prompt=prompt)
    else:
        return partial(prepare_inputs_for_instruct, prompt=prompt)
