import json
import csv
import os
import sys
import re
from typing import Optional
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from apply_bandaid import apply_bandaid

# params
MODEL_PATH = "Qwen/Qwen3-4B-Thinking-2507"
REPO_PATH = "anuragc14653/qwen_sft"
DATA_PATH = "../data/public.jsonl" 
OUTPUT_PATH = "../results/baseline_results_temp_0_few_shot.csv"
GPU_ID = "0"


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# check if public or private and then do evaluation
DO_EVAL = "public" in DATA_PATH.lower()

# prompt formatting
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a **single** \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
    "If the problem has multiple sub-answers, separate them by commas inside a **single** \\boxed{}, e.g. \\boxed{A,D}"
)

def build_messages(question: str, options: Optional[list]):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_content = f"{question}\n\nOptions:\n{opts_text}"
        system_content = SYSTEM_PROMPT_MCQ
    else:
        user_content = question
        system_content = SYSTEM_PROMPT_MATH
        
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

def extract_letter(text: str):
    """Extracts the MCQ answer letter for evaluation."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m: return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def main():
    print(f"Loading model and tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        #quantization="bitsandbytes", 
        load_format="auto",
        enable_prefix_caching=True,
        enforce_eager=False, 
        gpu_memory_utilization=0.75, 
        max_model_len=20480,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=64,
    )

    sampling_params = SamplingParams(
        max_tokens=16384,
        temperature=0.0,
        presence_penalty=0.0,
    )
    
    # Load Judger only if public.jsonl
    judger = None
    if DO_EVAL:
        print("Do eval! Loading Judger")
        sys.path.insert(0, ".")
        from old_judger import Judger
        judger = Judger(strict_extract=False)

    # load data and format it
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]

    prompts = []
    for item in data:
        messages = build_messages(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # run inference
    print(f"Generating responses for {len(prompts)} questions...")
    all_outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in all_outputs]

    # scoring and saving
    print("Scoring and compiling results...")
    results = []
    
    for item, response in zip(data, responses):
        is_mcq = bool(item.get("options"))
        
        # Regex sanitization for Kaggle
        if "</think>" in response:
            parts = response.split("</think>")
            thoughts = parts[0]
            final_answer = "</think>".join(parts[1:])
            safe_thoughts = thoughts.replace("\\boxed{", "boxed{")
            kaggle_safe_response = safe_thoughts + "</think>" + final_answer
        else:
            kaggle_safe_response = response

        # Bandaid: fix Issue B (multiple boxes) and Issue D (box in think)
        kaggle_safe_response = apply_bandaid(kaggle_safe_response)

        # Base record for the Kaggle CSV using the sanitized response
        record = {
            "id": item.get("id"),
            "response": kaggle_safe_response
        }
        
        if DO_EVAL and "answer" in item:
            gold = item["answer"]
            record["gold"] = gold
            record["is_mcq"] = is_mcq
            
            eval_text = kaggle_safe_response
            if "</think>" in kaggle_safe_response:
                eval_text = kaggle_safe_response.split("</think>")[-1]

            if is_mcq:
                record["correct"] = (extract_letter(eval_text) == str(gold).strip().upper())
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    record["correct"] = judger.auto_judge(pred=eval_text, gold=gold_list, options=[[]] * len(gold_list))
                except Exception:
                    record["correct"] = False
                    
        results.append(record)
    # metrics if public.jsonl
    if DO_EVAL:
        total = len(results)
        overall_correct = sum(1 for r in results if r["correct"])
        
        mcq_results = [r for r in results if r["is_mcq"]]
        mcq_correct = sum(1 for r in mcq_results if r["correct"])
        
        free_results = [r for r in results if not r["is_mcq"]]
        free_correct = sum(1 for r in free_results if r["correct"])
        
        print("\n")
        print("FINAL SFT BASELINE METRICS:")
        print(f"Overall Accuracy: {overall_correct / total:.2%} ({overall_correct}/{total})")
        if mcq_results:
            print(f"MCQ Accuracy: {mcq_correct / len(mcq_results):.2%} ({mcq_correct}/{len(mcq_results)})")
        if free_results:
            print(f"Free-Form Acc: {free_correct / len(free_results):.2%} ({free_correct}/{len(free_results)})")
        print("\n")
    
    # Ensure output directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(results)} records to {OUTPUT_PATH} as csv...")
    
    # Save as CSV
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
        # Strictly define only the two columns Kaggle wants
        fieldnames = ["id", "response"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                "id": r["id"],
                "response": r["response"]
            })
            
    print("Finished inference.")

if __name__ == "__main__":
    main()
