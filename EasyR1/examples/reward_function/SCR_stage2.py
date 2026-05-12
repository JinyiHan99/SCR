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

import re
from typing import Any, List, Dict, Optional

pattern_boxed = re.compile(
    r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
)

def normalize_response(response: str) -> str:
    """统一 tag 格式，避免空格问题"""
    return re.sub(r"\s*(<|>|/)\s*", r"\1", response)

def parse_response_blocks(text: str) -> Dict[str, Any]:
    """
    统一解析 response 的所有结构信息
    """
    text = text.strip()

    def extract(tag):
        return re.findall(
            rf"<{tag}>(.*?)</{tag}>",
            text,
            flags=re.DOTALL
        )

    answer_blocks = extract("answer")
    critic_blocks = extract("critic")
    revise_blocks = extract("revised")

    def last_boxed(block: Optional[str]) -> Optional[str]:
        if not block:
            return None
        matches = pattern_boxed.findall(block)
        return matches[-1] if matches else None

    first_answer = answer_blocks[0] if answer_blocks else None
    revised_answer = revise_blocks[0] if revise_blocks else None

    return {
        "answer_blocks": answer_blocks,
        "critic_blocks": critic_blocks,
        "revise_blocks": revise_blocks,
        "first_answer": first_answer,
        "revised_answer": revised_answer,
        "final_answer": revised_answer or first_answer,
        "first_answer_boxed": last_boxed(first_answer),
        "revised_answer_boxed": last_boxed(revised_answer),
    }


# def format_reward(parsed: Dict[str, Any]) -> float:
#     answer_blocks = parsed["answer_blocks"]
#     critic_blocks = parsed["critic_blocks"]
#     revise_blocks = parsed["revise_blocks"]

#     # 必须有 answer + critic
#     if len(answer_blocks) != 1 or len(critic_blocks) != 1:
#         return 0.0

#     # 只允许两种结构
#     if len(revise_blocks) not in (0, 1):
#         return 0.0

#     # answer / revise 必须有 boxed
#     if parsed["first_answer_boxed"] is None:
#         return 0.0

#     if revise_blocks and parsed["revised_answer_boxed"] is None:
#         return 0.0

#     # critic 结尾必须是 T / F
#     critic = critic_blocks[0].strip()
#     if not re.search(r"(T|F)\s*$", critic):
#         return 0.0

#     return 1.0


BOXED_PATTERN = re.compile(r'\\boxed\{.*?\}', re.DOTALL)

def format_reward(text: str) -> float:
    text = text.strip()

    # 模式 A: answer -> critic(T)
    pattern_a = (
        r"^<answer>.*?</answer>\s*"
        r"<critic>.*?(T)\s*</critic>$"
    )

    # 模式 B: answer -> critic(F) -> revised
    pattern_b = (
        r"^<answer>.*?</answer>\s*"
        r"<critic>.*?(F)\s*</critic>\s*"
        r"<revised>.*?</revised>$"
    )

    is_a = re.match(pattern_a, text, re.DOTALL) is not None
    is_b = re.match(pattern_b, text, re.DOTALL) is not None

    if not (is_a or is_b):
        return 0.0

    # 提取内容（结构已保证唯一）
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL).group(1)

    if BOXED_PATTERN.search(answer) is None:
        return 0.0

    if is_b:
        revised = re.search(r"<revised>(.*?)</revised>", text, re.DOTALL).group(1)
        if BOXED_PATTERN.search(revised) is None:
            return 0.0

    return 1.0


def extract_critic_label_from_block(block: str) -> Optional[bool]:
    m = re.search(r"(T|F)\s*$", block.strip())
    if not m:
        return None
    return m.group(1) == "T"


def critic_reward(parsed: Dict[str, Any], first_answer_correct: float) -> float:
    critic_blocks = parsed["critic_blocks"]
    if not critic_blocks:
        return 0.0

    pred = extract_critic_label_from_block(critic_blocks[0])
    if pred is None:
        return 0.0

    return 1.0 if pred == (first_answer_correct == 1.0) else 0.0


def accuracy_reward(
    boxed_answer: Optional[str],
    ground_truth: str
) -> float:
    if boxed_answer is None:
        return 0.0

    boxed_ans = f"\\boxed{{{boxed_answer}}}"
    boxed_gt = f"\\boxed{{{ground_truth}}}"

    if boxed_ans == boxed_gt:
        return 1.0

    try:
        return float(verify(parse(boxed_ans), parse(boxed_gt)))
    except Exception:
        return 0.0


from typing import Any, List, Dict


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
) -> List[Dict[str, float]]:

    scores = []

    # revise 行为奖励矩阵
    revise_reward_table = {
        (1.0, 1.0): -0.5,   # T -> T  0.5
        (1.0, 0.0): -0.5,   # T -> F  -0.5
        (0.0, 1.0): -0.1,   # F -> T  0.9
        (0.0, 0.0): -0.3,   # F -> F  -0.3
    }

    for reward_input in reward_inputs:
        response_raw = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        # ---------- 1. normalize ----------
        response = normalize_response(response_raw)

        # ---------- 2. parse once ----------
        parsed = parse_response_blocks(response)

        # ---------- 3. format reward ----------
        format_score = format_reward(response_raw)

        # ---------- 4. accuracy reward（最终答案） ----------
        final_boxed = (
            parsed["revised_answer_boxed"]
            if parsed["revised_answer_boxed"] is not None
            else parsed["first_answer_boxed"]
        )
        accuracy_score = accuracy_reward(final_boxed, ground_truth)

        # ---------- 5. first answer accuracy ----------
        first_answer_correct = accuracy_reward(
            parsed["first_answer_boxed"],
            ground_truth
        )

        # ---------- 6. critic reward ----------
        critic_score = critic_reward(
            parsed,
            first_answer_correct
        )

        # ---------- 7. revise behavior reward ----------
        correct_reward = 0.0
        if parsed["revised_answer_boxed"] is not None:
            revise_answer_correct = accuracy_reward(
                parsed["revised_answer_boxed"],
                ground_truth
            )
            correct_reward = revise_reward_table[
                (first_answer_correct, revise_answer_correct)
            ]

        # ---------- 8. overall ----------
        overall_score = (
            accuracy_score
            + format_weight * format_score
            + correct_reward
        )

        scores.append(
            {
                "format": format_score,
                "accuracy": accuracy_score,
                "critic": critic_score,
                "revise": correct_reward,
                "overall": overall_score,
            }
        )

    return scores



test = {"response":"<answer>\nTo find out how much profit Josh made, we need to follow these steps:\n\n1. Calculate the new value of the house after the repairs.\n2. Subtract the total cost (purchase price + repairs) from the new value to find the profit.\n\nStep 1: Calculate the new value of the house after the repairs.\nThe value of the house increased by 150% after the repairs. To find the new value, we need to calculate 150% of the total cost and add it to the total cost.\n\nTotal cost = Purchase price + Repairs\nTotal cost = $80,000 + $50,000\nTotal cost = $130,000\n\nNow, calculate 150% of the total cost:\n150% of $130,000 = 1.5 * $130,000 = $195,000\n\nNew value of the house = Total cost + 150% of total cost\nNew value of the house = $130,000 + $195,000\nNew value of the house = $325,000\n\nStep 2: Subtract the total cost from the new value to find the profit.\nProfit = New value of the house - Total cost\nProfit = $325,000 - $130,000\nProfit = $195,000\n\nSo, Josh made a profit of $195,000.\n\n\\boxed{195000}</answer>\n<critic>The answer is correct; all calculations and reasoning are accurate. T\n</critic>\n",
"ground_truth": "5000"
}
res = compute_score([test])
print(res)
# acc = accuracy_reward(test, "70000")
# format = format_reward(test)
# critic = critic_reward(test,acc)
# print(acc)
# print(format)
# print(critic)





# test = "thsths \\boxed{A}"

# acc = accuracy_reward(test, "B")
# print(acc)

# first_answer_content = test.split("<critic>")[0]

# print(first_answer_content)

