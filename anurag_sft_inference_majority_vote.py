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

# params
MODEL_PATH = "Qwen/Qwen3-4B-Thinking-2507"
REPO_PATH   = "anuragc14653/qwen_sft"
DATA_PATH   = "data/private.jsonl" 
OUTPUT_PATH = "results/sft_submission.csv"
GPU_ID      = "0"

# --- Majority voting ---
# Set MAJORITY_VOTE = True to generate N_VOTES responses per question and
# pick the most common answer.  Temperature is automatically raised to 0.6
# when voting is on (greedy decoding makes all votes identical, which is useless).
# Typical gains: +3-8% accuracy.  Runtime cost: N_VOTES x longer.
MAJORITY_VOTE = False
N_VOTES       = 4     # number of responses to generate per question


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

def build_messages(question: str, options: Optional[list]) -> list:
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

def extract_letter(text: str) -> str:
    """Extracts the MCQ answer letter for evaluation."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m: return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


# ---------------------------------------------------------------------------
# Majority voting helpers
# ---------------------------------------------------------------------------

def _extract_boxed_content(text: str) -> str:
    """
    Brace-aware extraction of the last \\boxed{} content from text.
    Handles nested braces correctly (e.g. \\boxed{\\frac{1}{2}}).
    Returns empty string if no box found.
    """
    last_content = ""
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":   depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start : i - 1].strip()
            if content:
                last_content = content
        start = max(brace_start, i)
    return last_content


def _normalise_answer(text: str) -> str:
    """
    Light normalisation for grouping votes:
    strip whitespace, collapse internal spaces, lowercase.
    Keeps LaTeX structure intact so \\frac{1}{2} != \\frac{1}{3}.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def majority_vote(responses: list[str], is_mcq: bool) -> str:
    """
    Given N raw model responses for one question, return the single
    best response — the one whose extracted answer matches the plurality.

    Strategy:
      1. Extract the post-think answer section from each response.
      2. Pull out the last \\boxed{} content (after applying bandaid).
      3. Normalise and count votes.
      4. Return the full response (including think section) that corresponds
         to the most-voted answer.  Ties are broken by whichever answer
         appeared first among the tied group.
    """
    from collections import Counter

    # (normalised_answer, original_response) pairs
    candidates = []
    for resp in responses:
        # Get post-think text
        post = resp.split("</think>")[-1] if "</think>" in resp else resp
        box  = _extract_boxed_content(post)
        norm = _normalise_answer(box) if box else "__no_answer__"
        candidates.append((norm, resp))

    if not candidates:
        return responses[0]

    # Count votes
    vote_counts = Counter(norm for norm, _ in candidates)
    winner_norm = vote_counts.most_common(1)[0][0]

    # Return the first response that matches the winning answer
    for norm, resp in candidates:
        if norm == winner_norm:
            return resp

    return responses[0]

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
        gpu_memory_utilization=0.85, 
        max_model_len=8192,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=64,
    )

    if MAJORITY_VOTE:
        print(f"Majority voting ON  — generating {N_VOTES} responses per question")
        sampling_params = SamplingParams(
            max_tokens=16384,
            temperature=0.6,   # must be > 0 for diverse votes; 0.2 makes all votes identical
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
            n=N_VOTES,         # vLLM generates N completions per prompt in one pass
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=16384,
            temperature=0.2,
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
        )
    
    # Load Judger only if public.jsonl
    judger = None
    if DO_EVAL:
        print("Do eval! Loading Judger")
        sys.path.insert(0, ".")
        from judger import Judger
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

    if MAJORITY_VOTE:
        # Each output has N_VOTES completions; pick the plurality answer
        is_mcq_flags = [bool(item.get("options")) for item in data]
        responses = [
            majority_vote(
                [out.text.strip() for out in output.outputs],
                is_mcq=is_mcq_flags[i],
            )
            for i, output in enumerate(all_outputs)
        ]
        print(f"  Voting complete — selected best of {N_VOTES} responses per question")
    else:
        responses = [out.outputs[0].text.strip() for out in all_outputs]

    # scoring and saving
    print("Scoring and compiling results...")
    results = []
    
    for item, response in zip(data, responses):
        is_mcq = bool(item.get("options"))
        
        # --- REGEX SANITIZATION FOR KAGGLE ---
        if "</think>" in response:
            parts = response.split("</think>")
            thoughts = parts[0]
            final_answer = "</think>".join(parts[1:])
            
            # Remove the backslash to break the regex trap in the thoughts
            safe_thoughts = thoughts.replace("\\boxed{", "boxed{")
            kaggle_safe_response = safe_thoughts + "</think>" + final_answer
        else:
            kaggle_safe_response = response
        # -------------------------------------

        # Base record for the Kaggle CSV using the SANITIZED response
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
        
        print("\n" + "="*30)
        print(" FINAL SFT BASELINE METRICS ")
        print("="*30)
        print(f"Overall Accuracy:  {overall_correct / total:.2%} ({overall_correct}/{total})")
        if mcq_results:
            print(f"MCQ Accuracy:      {mcq_correct / len(mcq_results):.2%} ({mcq_correct}/{len(mcq_results)})")
        if free_results:
            print(f"Free-Form Acc:     {free_correct / len(free_results):.2%} ({free_correct}/{len(free_results)})")
        print("="*30 + "\n")
    
    # Ensure output directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(results)} records to {OUTPUT_PATH} as csv...")
    
    # Save as a strict, Kaggle-compliant CSV
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
            
    print("Finsihed inference.")

if __name__ == "__main__":
    main()