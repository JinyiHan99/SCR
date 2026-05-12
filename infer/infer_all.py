import os
import re
import json
import argparse
import time
import math
import gc
import torch
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

os.environ["VLLM_USE_V1"] = "1"
from math_verify import parse, verify  

normal_gen = "You are a helpful AI Assistant. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The final answer MUST BE put in \\boxed{}."

SCR_prompt = """You are a helpful AI assistant.
For each question, you must first solve the problem and put your complete solution inside <answer>...</answer>. The final result must be wrapped in \\boxed{}.
Then you must evaluate your own solution inside <critic>...</critic>. In this block, you must first give your evaluation and reasoning. At the end of the <critic>, you must give the final judgment using only one single symbol: T or F. T means the answer is correct. F means the answer is incorrect.
If the final judgment is F, you must give a corrected solution inside <revised>...</revised>, and the final result must also be wrapped in \\boxed{}. After that, you must give another <critic>...</critic> in exactly the same format as before.
If the final judgment is T, you must stop and give no further output.
"""

DEFAULT_PLAIN_SYSTEM_PROMPT = "{QUESTION}"
SAMPLING_PARAMS_DICT = dict(temperature=0.6, max_tokens=24000, repetition_penalty=1, skip_special_tokens=False, n=1, top_p=0.95)

test_data_dir = "/SCR/data/test_data"  
TASK_PATHS = {
    "math500":f"{test_data_dir}/math.jsonl",
    "gsm8k":f"{test_data_dir}/gsm.jsonl",
    "aime24":f"{test_data_dir}/aime24.jsonl",
    "olypaid":f"{test_data_dir}/olypaid.jsonl",
    "amc":f"{test_data_dir}/amc.jsonl",
    'minervamath':f"{test_data_dir}/minervamath.jsonl",
    "aime25":f"{test_data_dir}/aime25.jsonl",
    "arc":f"{test_data_dir}/arc.jsonl",
    "gpqa":f"{test_data_dir}/gpqa.jsonl",
    "mmlu_pro":f"{test_data_dir}/mmlu_pro.jsonl",
}

pattern_boxed = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
_PROMPT_PRINTED = False

def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    questions, ground_truths = [], []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                item = json.loads(line)
                questions.append(item["question"])
                ground_truths.append(item["std"])
        else:
            data = json.load(f)
            for item in data:
                questions.append(item["question"])
                ground_truths.append(item["std"])
    return questions, ground_truths

def reward_correct(ground_truth: str, answer: str) -> bool:
    boxed_matches_ans = re.findall(pattern_boxed, answer)
    if not boxed_matches_ans: return False
    boxed_ans = "\\boxed{" + boxed_matches_ans[-1] + "}"
    boxed_gt = "\\boxed{" + ground_truth + "}"
    if boxed_ans == boxed_gt: return True
    try:
        return verify(parse(boxed_ans), parse(boxed_gt))
    except Exception:
        return False

def _pass_at_k_from_counts(n: int, c: int, k: int) -> float:
    if k > n or n <= 0: return float("nan")
    if c == 0: return 0.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

class VLLMWorker:
    def __init__(self, model_path: str, use_chat_template: bool, chat_system_prompt_key: str, 
                 plain_system_prompt: str, dtype: str = "bfloat16", enforce_eager: bool = True):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.use_chat_template = use_chat_template
        self.model_path = model_path
        
        prompt_map = {
            "correct_prompt": correct_prompt,
            "normal_gen": normal_gen,
        }
        self.chat_system_prompt = prompt_map.get(chat_system_prompt_key, chat_system_prompt_key)
        self.plain_system_prompt = plain_system_prompt

        self.llm = LLM(
            model=model_path, 
            dtype=dtype, 
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.85, 
            tensor_parallel_size=1, 
            max_model_len=24000
        )

        if self.use_chat_template:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _build_inputs(self, qs: List[str]) -> List[str]:
        global _PROMPT_PRINTED
        if not self.use_chat_template:
            return [self.plain_system_prompt.format(QUESTION=q) for q in qs]
        
        tip_text = []
        for x in qs:
            s = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.chat_system_prompt},
                 {"role": "user", "content": x}],
                tokenize=False, add_generation_prompt=True,
            )
            tip_text.append(s)
        
        if not _PROMPT_PRINTED and tip_text:
            print(f"======PROMPT EXAMPLE=======:\n{tip_text[0]}")
            _PROMPT_PRINTED = True
        return tip_text

    def generate(self, qs: List[str], sampling_params_dict: Dict[str, Any]) -> List[Dict]:
        from vllm import SamplingParams
        
        tip_text = self._build_inputs(qs)
        v_sampling_params = SamplingParams(**sampling_params_dict)
        voutputs = self.llm.generate(tip_text, v_sampling_params, use_tqdm=False)
        
        results = []
        for v in voutputs:
            for z in v.outputs:
                results.append({"text": z.text})
        return results

def run_task(task_name: str, path: str, worker: VLLMWorker, task_sampling_params: Dict[str, Any], 
             batch_size: int = 50, repeat: int = 1, output_path: str = None):

    print(f"\n>>> Starting Task: {task_name} (Repeat: {repeat}, Temp: {task_sampling_params.get('temperature')})")
    questions_raw, gts_raw = load_dataset(path)
    n0 = len(questions_raw)
    
    questions = questions_raw * repeat
    gts = gts_raw * repeat
    total_items = len(questions)
    
    results: List[Dict[str, Any]] = []
    correct = total = 0
    
    fout = open(output_path, "a", encoding="utf-8") if output_path else None
    pbar = tqdm(total=total_items, desc=f"Processing {task_name}")

    for i in range(0, total_items, batch_size):
        sub_qs = questions[i:i + batch_size]
        sub_gts = gts[i:i + batch_size]
        
        answers = worker.generate(sub_qs, task_sampling_params)
        
        for j, item in enumerate(answers):
            ans_text = item["text"]
            gt = sub_gts[j]
            is_correct = reward_correct(gt, ans_text)
            
            rec = {
                "task": task_name,
                "question": sub_qs[j],
                "ground_truth": gt,
                "prediction": ans_text,
                "is_correct": is_correct
            }
            
            if fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            
            results.append(rec)
            correct += int(is_correct)
            total += 1
            pbar.update(1)

    pbar.close()
    if fout: fout.close()
    
    acc = correct / total if total else 0.0
    print(f"Task [{task_name}] Result: {correct}/{total} = {acc:.2%}")
    return results, correct, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=["math500"])
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--chat_system_prompt", type=str, default="normal_gen")
    parser.add_argument("--plain_system_prompt", type=str, default=DEFAULT_PLAIN_SYSTEM_PROMPT)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_file = args.output if args.output else f"./eval_results_{int(time.time())}.jsonl"

    worker = VLLMWorker(
        model_path=args.model_path,
        use_chat_template=args.use_chat_template,
        chat_system_prompt_key=args.chat_system_prompt,
        plain_system_prompt=args.plain_system_prompt,
        dtype=args.dtype
    )

    all_total_correct = 0
    all_total_count = 0

    for task in args.tasks:
        path = TASK_PATHS.get(task)
        if not path or not os.path.exists(path):
            print(f"[Skip] Task path not found: {task}")
            continue

        task_params = SAMPLING_PARAMS_DICT.copy()
        repeat = 1
        
        if any(k in task for k in ["aime", "amc"]):
            task_params["temperature"] = 1.0
            repeat = 10
        elif "train" in task:
            task_params["temperature"] = 1.0
            repeat = 8
        else:
            task_params["temperature"] = 0.0
            repeat = 1

        _, c, t = run_task(
            task_name=task,
            path=path,
            worker=worker,
            task_sampling_params=task_params,
            batch_size=args.batch_size,
            repeat=repeat,
            output_path=output_file
        )
        
        all_total_correct += c
        all_total_count += t

    if all_total_count > 0:
        print(f"\n{'='*30}\nFinal Overall Accuracy: {all_total_correct}/{all_total_count} = {all_total_correct/all_total_count:.2%}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")