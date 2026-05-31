import json
import csv
import re
import sys

# Load the local Judger
try:
    from judger import Judger
    judger = Judger(strict_extract=False)
except ImportError:
    print("Error: judger.py not found in the current directory.")
    sys.exit(1)

CSV_PATH = "results/sft_results_bandaid.csv"
JSONL_PATH = "data/public.jsonl"

def extract_final_answer(response: str):
    """Isolates the text after </think> and extracts the LAST \boxed{}"""
    # 1. Ignore the thought process
    if "</think>" in response:
        eval_text = response.split("</think>")[-1]
    else:
        eval_text = response
        
    # 2. Find all boxed answers in the final text
    matches = re.findall(r"\\boxed\{([^}]*)\}", eval_text)
    
    # 3. Return the very last one, or None if missing
    if matches:
        return matches[-1].strip()
    return None

def main():
    print("Loading Ground Truth from JSONL...")
    ground_truth = {}
    with open(JSONL_PATH, "r") as f:
        for line in f:
            data = json.loads(line)
            ground_truth[str(data["id"])] = {
                "answer": data["answer"],
                "is_mcq": bool(data.get("options"))
            }

    print("Loading Predictions from CSV...")
    predictions = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row)

    print("Evaluating True Accuracy...")
    results = {"mcq_correct": 0, "mcq_total": 0, "free_correct": 0, "free_total": 0, "missing_box": 0}

    for pred in predictions:
        item_id = str(pred["id"])
        if item_id not in ground_truth:
            continue
            
        gt_info = ground_truth[item_id]
        gold = gt_info["answer"]
        is_mcq = gt_info["is_mcq"]
        
        extracted_ans = extract_final_answer(pred["response"])
        
        # Track formatting failures
        if extracted_ans is None:
            results["missing_box"] += 1
            correct = False
        else:
            # Score it
            if is_mcq:
                # Just compare the first letter for MCQ (e.g., "A")
                extracted_letter = re.sub(r"[^A-Za-z]", "", extracted_ans)
                correct = (extracted_letter.upper() == str(gold).strip().upper())
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    # Pass the extracted text to the judger
                    correct = judger.auto_judge(pred=extracted_ans, gold=gold_list, options=[[]] * len(gold_list))
                except Exception:
                    correct = False

        if is_mcq:
            results["mcq_total"] += 1
            if correct: results["mcq_correct"] += 1
        else:
            results["free_total"] += 1
            if correct: results["free_correct"] += 1

    # Print Final Metrics
    overall_correct = results["mcq_correct"] + results["free_correct"]
    overall_total = results["mcq_total"] + results["free_total"]
    
    print("\n" + "="*40)
    print(" CORRECTED SFT BASELINE METRICS ")
    print("="*40)
    print(f"Overall Accuracy:  {overall_correct / overall_total:.2%} ({overall_correct}/{overall_total})")
    if results["mcq_total"] > 0:
        print(f"MCQ Accuracy:      {results['mcq_correct'] / results['mcq_total']:.2%} ({results['mcq_correct']}/{results['mcq_total']})")
    if results["free_total"] > 0:
        print(f"Free-Form Acc:     {results['free_correct'] / results['free_total']:.2%} ({results['free_correct']}/{results['free_total']})")
    print("-" * 40)
    print(f"Formatting Failures (Missing \\boxed): {results['missing_box']} / {overall_total}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()