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
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import parse, verify 





# def format_reward(response: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     format_match = re.fullmatch(pattern, response)
#     return 1.0 if format_match else 0.0


# def accuracy_reward(response: str, ground_truth: str) -> float:
#     answer = extract_boxed_content(response)
#     return 1.0 if grade_answer(answer, ground_truth) else 0.0

def format_reward(answer: str) -> float:
    box_match = re.search(r'\\boxed\{.*?\}', answer)
    if not box_match:
        return 0.0
    return 1.0

pattern_boxed = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'

def accuracy_reward(answer, ground_truth: str) -> bool:
    boxed_matches_ans = re.findall(pattern_boxed, answer)
    if not boxed_matches_ans:
        return False
    boxed_ans = "\\boxed{" + boxed_matches_ans[-1] + "}"
    boxed_gt = "\\boxed{" + ground_truth + "}"
    if boxed_ans == boxed_gt:
        return 1.0
    try:
        return verify(parse(boxed_ans), parse(boxed_gt))
    except Exception as e:
        print("Parse/verify error:", e)
        return 0.0

def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
