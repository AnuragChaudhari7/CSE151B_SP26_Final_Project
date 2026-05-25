import json
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from judger import Judger

# INFERENCE SECTION

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices carefully. "
    "Think through the problem step by step, then select the single best answer. "
    "At the very end of your response, output the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question

    prompts = []
for item in experiment_data:
    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt_text)

# Generate
print(f"Generating responses for {len(prompts)} questions...")
# outputs = llm.generate(prompts, sampling_params=sampling_params)
# Added manual chunking because above line was causing EngineDead error
################################################################################
BATCH_SIZE = 8   # start small

all_outputs = []

for i in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[i:i + BATCH_SIZE]

    # print(f"Processing batch {i//BATCH_SIZE + 1}")

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    # 3. Generate the output
    # Note: It will automatically use the GenerationConfig you saved earlier, 
    # but you can override parameters here if needed.
    outputs = llm.generate(
        **inputs,
        max_new_tokens=16384,       # Limit the length of the generated response
        pad_token_id=tokenizer.eos_token_id
    )

    # 4. Decode and print the result
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    all_outputs.extend(response)
################################################################################

responses_50 = [out.outputs[0].text.strip() for out in all_outputs]

all_round_metrics = {}

def compute_metrics(results, label=""):
    y_true = [1] * len(results)                        # every question has a correct answer
    y_pred = [1 if r["correct"] else 0 for r in results]  # 1 if model got it right

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def subset_scores(subset):
        yt = [1] * len(subset)
        yp = [1 if r["correct"] else 0 for r in subset]
        return (
            accuracy_score(yt, yp),
            precision_score(yt, yp, zero_division=0),
            recall_score(yt, yp, zero_division=0),
            f1_score(yt, yp, zero_division=0),
            sum(yp),
            len(yp)
        )

    print(f"\n{'='*50}")
    print(f"METRICS — {label}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Correct   : {sum(y_pred)} / {len(y_pred)}")

    for subset, name in [(mcq_res, "MCQ"), (free_res, "Free-form")]:
        if not subset:
            continue
        a, p, r, f, correct, total = subset_scores(subset)
        print(f"\n  [{name}]")
        print(f"    Accuracy  : {a:.4f}  ({a*100:.2f}%)")
        print(f"    Precision : {p:.4f}")
        print(f"    Recall    : {r:.4f}")
        print(f"    F1        : {f:.4f}")
        print(f"    Correct   : {correct} / {total}")

    print(f"{'='*50}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def record_round(round_name, results):
    metrics = compute_metrics(results, label=round_name)
    all_round_metrics[round_name] = metrics
    return metrics

def print_comparison():
    print(f"\n{'Round':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for name, m in all_round_metrics.items():
        print(f"{name:<20} {m['accuracy']:>9.4f}  {m['precision']:>9.4f}  {m['recall']:>9.4f}  {m['f1']:>9.4f}")

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


# Load Judger for free-form scoring
sys.path.insert(0, ".")

judger = Judger(strict_extract=False)

results = []
# for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
for item, response in tqdm(zip(experiment_data, responses_50), total=len(data), desc="Scoring"): # CHANGED data TO experiment_data

    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": response,
        "correct":  correct,
    })

print(f"Scoring complete. {len(results)} results.")


mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)