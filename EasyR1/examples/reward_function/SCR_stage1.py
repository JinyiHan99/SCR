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
import pdb
from math_verify import parse, verify 


import json

REWARD_NAME = "math"
REWARD_TYPE = "batch"

BOXED_PATTERN = re.compile(r'\\boxed\{.*?\}', re.DOTALL)

def format_reward(text: str) -> float:
    text = text.strip()


    pattern_a = (
        r"^<answer>.*?</answer>\s*"
        r"<critic>.*?(T)\s*</critic>$"
    )

    pattern_b = (
        r"^<answer>.*?</answer>\s*"
        r"<critic>.*?(F)\s*</critic>\s*"
        r"<revised>.*?</revised>$"
    )

    is_a = re.match(pattern_a, text, re.DOTALL) is not None
    is_b = re.match(pattern_b, text, re.DOTALL) is not None

    if not (is_a or is_b):
        return 0.0

    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL).group(1)

    if BOXED_PATTERN.search(answer) is None:
        return 0.0

    if is_b:
        revised = re.search(r"<revised>(.*?)</revised>", text, re.DOTALL).group(1)
        if BOXED_PATTERN.search(revised) is None:
            return 0.0

    return 1.0


def extract_critic_label(text: str):
    m_block = re.search(r"<critic>(.*?)</critic>", text, flags=re.DOTALL)
    if not m_block:
        return None

    block = m_block.group(1)
    m_label = re.search(r"(T|F)\s*$", block)
    if not m_label:
        return None

    return m_label.group(1) == "T"

def critic_reward(response: str, accuracy_score: float) -> float:
    pred = extract_critic_label(response)
    if pred is None:
        return 0.0

    is_correct = (accuracy_score == 1.0)

    return 1.0 if pred == is_correct else 0.0


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
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"]) 
        format_score = format_reward(response)
        first_answer_content = response.split("<critic>")[0]
        accuracy_score = accuracy_reward(first_answer_content, reward_input["ground_truth"])  
        critic_score = critic_reward(response, accuracy_score)
        scores.append(
            {
                "format": format_score,
                "accuracy": accuracy_score,
                "critic": critic_score,
                "overall": 0.5*accuracy_score + format_weight * format_score + critic_score,
            }
        )

    return scores




