# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric, parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from math_verify.utils import timeout
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")



def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, consider_length=True) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    # use re to find the \\boxed{} pattern in model_output,
    # if cannot find such pattern, the model does not follow the instruction
    # its score should be zero no matter the answer is correct or not
    whether_model_output_boxed = re.search(r"\\boxed\{.*?\}", model_output)
    if not whether_model_output_boxed:
        return 0.0
    if len(ground_truth) > 1000 and consider_length:
        # this is very hard problem without calculator, the policy would never answer with correct answer
        # in such case, we directly use str compare 
        try:
            pred_answer = parse(model_output, (ExprExtractionConfig(), LatexExtractionConfig()))
            if pred_answer:
                pred_str_answer = pred_answer[1]
                return pred_str_answer.strip() == ground_truth.strip()
            else:
                return 0
        except TimeoutException:
            return timeout_score

    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score
