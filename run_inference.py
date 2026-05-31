"""
run_inference.py
================
Inference script using the baseline Qwen model, with a layered
post-processing pipeline and detailed error analysis.

Post-processing pipeline (applied in order)
--------------------------------------------
  1. apply_bandaid()       — Issue B: multiple \\boxed{} after </think> → merged
                           — Issue D: \\boxed{} only inside <think> → moved after
  2. fix_thousands_sep()   — "105,950" → "105950" inside \\boxed{}
                             The judger's split_by_comma() treats a thousands-
                             separator comma the same as a multi-answer comma,
                             so "105,950" becomes ['105','950'] → count mismatch.
  3. fix_decimal_precision()
                           — Decimal with < 6 decimal places → try to extend from
                             the response text, or convert to an exact LaTeX fraction.
                             Judger uses round(gold, 6) then 1e-8 relative tolerance,
                             so "0.33" fails where "0.333333" or "\\frac{1}{3}" passes.
  4. Kaggle safety         — \\boxed{} inside <think> is sanitised to boxed{}

Error categories tracked per response
--------------------------------------
  no_box           — no \\boxed{} anywhere in the response
  box_in_think     — \\boxed{} found only inside <think> (Issue D, pre-bandaid)
  multi_box        — multiple \\boxed{} after </think> (Issue B, pre-bandaid)
  thousands_sep    — comma-formatted number in boxed (e.g. "105,950")
  low_precision    — decimal answer with fewer than 6 decimal places
  pm_expansion     — \\pm in boxed (judger expands to two values; count must match)
  missing_think    — model produced no </think> tag at all
  count_mismatch   — #sub-answers in prediction ≠ #sub-answers in gold (post-fix)
  pct_likely       — possible percentage/decimal confusion (0–1 value, % in question)
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.insert(0, ".")
from apply_bandaid import apply_bandaid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Baseline: raw Qwen3-4B-Thinking model (no fine-tuning).
# Swap to "anuragc14653/qwen_sft_fixed" to run the merged SFT model.
MODEL_PATH  = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/run_inference_results.csv"
GPU_ID      = "0"


# ---------------------------------------------------------------------------
# Improved system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Solve the problem step-by-step inside <think> </think> tags. "
    "After </think>, output ONLY a single \\boxed{} containing your final answer — "
    "no explanation, no extra text. "
    "Do NOT use multiple \\boxed{} blocks. "
    "If there are multiple sub-answers, separate them by commas inside ONE \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "Never place \\boxed{} inside <think> </think>. "
    "For decimal answers, give at least 6 decimal places or use an exact fraction."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and answer choices, then reason inside <think> </think> tags. "
    "After </think>, output ONLY a single \\boxed{} containing the letter(s) of your answer — "
    "no explanation, no extra text. "
    "Do NOT use multiple \\boxed{} blocks. "
    "If multiple options are correct, separate them by commas inside ONE \\boxed{}, "
    "e.g. \\boxed{A, D}. "
    "Never place \\boxed{} inside <think> </think>."
)

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOTS_MATH = [
    {
        # Demonstrates exact fraction over decimal approximation.
        # The judger uses round(gold, 6) then 1e-8 relative tolerance, so
        # writing 0.333 or 0.3333 will FAIL. \frac{1}{3} is parsed by sympy
        # and compared exactly — always prefer exact form over truncated decimal.
        "question": "Three friends share a pizza equally. What fraction of the pizza does each person receive?",
        "answer": (
            "<think>\n"
            "Each person gets 1 out of 3 equal parts, which is 1 ÷ 3.\n"
            "As a decimal this is 0.3333..., which is infinite and cannot be written exactly.\n"
            "The exact answer is \\frac{1}{3}.\n"
            "Never write 0.333 or 0.3333 — always use the exact fraction \\frac{1}{3}.\n"
            "</think>\n"
            "\\boxed{\\frac{1}{3}}"
        ),
    },
    {
        "question": "Solve the system of equations:\nx + y = 10\nx - y = 4",
        "answer": (
            "<think>\n"
            "Adding: 2x = 14 → x = 7.\n"
            "Substituting: 7 + y = 10 → y = 3.\n"
            "</think>\n"
            "\\boxed{7, 3}"
        ),
    },
    {
        "question": "Find all values of x such that x² - x - 6 = 0.",
        "answer": (
            "<think>\n"
            "Factoring: (x - 3)(x + 2) = 0 → x = 3 or x = -2.\n"
            "</think>\n"
            "\\boxed{3, -2}"
        ),
    },
]

FEW_SHOTS_MCQ = [
    {
        "question": "Which of the following is a prime number?\nA) 15  B) 17  C) 21  D) 25",
        "answer": (
            "<think>\n"
            "15 = 3×5, 21 = 3×7, 25 = 5×5. Only 17 has no divisors other than 1 and itself.\n"
            "</think>\n"
            "\\boxed{B}"
        ),
    },
    {
        "question": (
            "Which of the following are factors of 12? Select all that apply.\n"
            "A) 2  B) 5  C) 3  D) 7  E) 6"
        ),
        "answer": (
            "<think>\n"
            "12 ÷ 2 = 6 ✓, 12 ÷ 5 = 2.4 ✗, 12 ÷ 3 = 4 ✓, 12 ÷ 7 ≈ 1.71 ✗, 12 ÷ 6 = 2 ✓.\n"
            "</think>\n"
            "\\boxed{A, C, E}"
        ),
    },
    {
        "question": (
            "Which equations have x = 2 as a solution? Select all that apply.\n"
            "A) x + 3 = 5  B) 2x = 8  C) x² = 4  D) x - 1 = 2"
        ),
        "answer": (
            "<think>\n"
            "A) 2+3=5 ✓  B) 2(2)=4≠8 ✗  C) 2²=4 ✓  D) 2-1=1≠2 ✗.\n"
            "</think>\n"
            "\\boxed{A, C}"
        ),
    },
]

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_messages(question: str, options: Optional[list]) -> list:
    if options:
        labels   = [chr(65 + i) for i in range(len(options))]
        opts_txt = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_content   = f"{question}\n\nOptions:\n{opts_txt}"
        system_content = SYSTEM_PROMPT_MCQ
        shots          = FEW_SHOTS_MCQ
    else:
        user_content   = question
        system_content = SYSTEM_PROMPT_MATH
        shots          = FEW_SHOTS_MATH

    messages = [{"role": "system", "content": system_content}]
    for shot in shots:
        messages.append({"role": "user",      "content": shot["question"]})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

_THINK_END = "</think>"


def _extract_boxed_contents(text: str) -> list[tuple[int, int, str]]:
    """
    Return list of (start_idx, end_idx, content) for every \\boxed{...} in text.
    Handles nested braces correctly.
    """
    results = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        depth = 1
        i = idx + len("\\boxed{")
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append((idx, i, text[idx + len("\\boxed{"):i - 1]))
        start = i
    return results


def _is_thousands_number(s: str) -> bool:
    """
    Return True if s looks like a thousands-formatted integer, e.g. '105,950' or '1,234,567'.
    """
    return bool(re.match(r"^-?\d{1,3}(,\d{3})+$", s.strip()))


def fix_thousands_sep(boxed_content: str) -> str:
    """
    Remove thousands separators from a boxed answer string.

    The judger's split_by_comma() treats '105,950' as two separate answers
    ['105', '950'], causing an immediate count-mismatch failure.
    We remove the separator only when the comma is unambiguously a thousands
    separator (digit groups of exactly 3 after the first group).

    Works on comma-separated multi-answer strings too: each sub-answer is
    checked independently so "3, 1,000" → "3, 1000".
    """
    # Split on top-level commas (not inside brackets)
    parts = re.split(r",(?!\d{3}(?:[,\d]|$))", boxed_content)
    # Simpler: just strip thousands seps from each comma-separated chunk
    # by checking the pattern after splitting naively, then reassembling.

    # Split by comma, check each part for thousands pattern
    raw_parts = boxed_content.split(",")
    # Rebuild: merge adjacent parts that form a thousands-formatted number
    merged = []
    i = 0
    while i < len(raw_parts):
        chunk = raw_parts[i].strip()
        # Try to extend with subsequent parts to form a thousands number
        candidate = chunk
        j = i + 1
        while j < len(raw_parts):
            next_part = raw_parts[j].strip()
            if re.match(r"^\d{3}$", next_part):
                # Could be thousands group
                test = candidate + next_part
                candidate = candidate + "," + next_part
                # Keep extending if it keeps looking like thousands
                j += 1
            else:
                break
        # Check if the whole candidate (with commas) is a thousands number
        if _is_thousands_number(candidate):
            merged.append(candidate.replace(",", ""))
            i = j
        else:
            merged.append(chunk)
            i += 1
    return ", ".join(merged)


def fix_decimal_precision(boxed_content: str, full_response: str) -> str:
    """
    If boxed_content is a decimal number with < 6 decimal places, attempt to:
      1. Find a more precise version of the same number in the response text.
      2. Convert to an exact LaTeX fraction using Python's Fraction.

    The judger rounds gold to 6 decimal places and uses 1e-8 relative
    tolerance, so "0.33" fails where "0.333333" or "\\frac{1}{3}" passes.

    Only acts on single-number boxed answers (not multi-answer strings).
    """
    content = boxed_content.strip()

    # Only handle simple decimal numbers (optionally negative)
    m = re.fullmatch(r"(-?\d+)\.(\d+)", content)
    if not m:
        return boxed_content

    dp = len(m.group(2))
    if dp >= 6:
        return boxed_content  # Already precise enough

    # ── Strategy 1: find a longer version of the same decimal in the response ──
    escaped = re.escape(content)
    # Look for the same prefix followed by more digits
    pattern = re.compile(rf"(?<!\d){escaped}(\d+)(?!\d)")
    matches = pattern.findall(full_response)
    if matches:
        # Pick the extension with the most additional digits (cap at 10 dp total)
        best = max(matches, key=len)
        extended = content + best
        # Verify it still parses as a float
        try:
            float(extended)
            return extended
        except ValueError:
            pass

    # ── Strategy 2: convert to exact LaTeX fraction ──────────────────────────
    try:
        frac = Fraction(content).limit_denominator(100_000)
        # Only substitute if the fraction is a close match
        if abs(float(frac) - float(content)) < 5e-7:
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"
    except Exception:
        pass

    return boxed_content


def apply_precision_to_response(response: str, full_response: str) -> str:
    """
    Apply fix_decimal_precision to every \\boxed{} in the answer section
    (after </think>).
    Returns the full modified response string.
    """
    think_idx = response.rfind(_THINK_END)
    if think_idx >= 0:
        prefix = response[: think_idx + len(_THINK_END)]
        answer_section = response[think_idx + len(_THINK_END):]
    else:
        prefix = ""
        answer_section = response

    boxes = _extract_boxed_contents(answer_section)
    if not boxes:
        return response

    # Build replacement from right to left to preserve indices
    for start, end, content in reversed(boxes):
        fixed_content = fix_decimal_precision(
            fix_thousands_sep(content), full_response
        )
        answer_section = (
            answer_section[:start]
            + f"\\boxed{{{fixed_content}}}"
            + answer_section[end:]
        )

    return prefix + answer_section


# ---------------------------------------------------------------------------
# Kaggle safety (sanitise \\boxed inside <think>)
# ---------------------------------------------------------------------------

def kaggle_safe(response: str) -> str:
    """
    Replace \\boxed{ with boxed{ ONLY inside <think>...</think> so that
    Kaggle's scoring regex doesn't pick up in-thinking boxes.
    The final answer \\boxed{} (after </think>) is left intact.
    """
    if _THINK_END not in response:
        return response
    think_end_idx = response.rfind(_THINK_END)
    inside  = response[:think_end_idx].replace("\\boxed{", "boxed{")
    outside = response[think_end_idx:]
    return inside + outside


# ---------------------------------------------------------------------------
# Error diagnosis (pre-bandaid, on raw model output)
# ---------------------------------------------------------------------------

def diagnose(raw: str, question: str) -> dict:
    """
    Return a dict of error flags for a raw model response.
    All checks are on the original text BEFORE any post-processing.
    """
    flags = defaultdict(bool)

    has_think_end = _THINK_END in raw
    flags["missing_think"] = not has_think_end

    boxes_full   = _extract_boxed_contents(raw)
    all_contents = [c for _, _, c in boxes_full]

    if not all_contents:
        flags["no_box"] = True
        return dict(flags)

    # Position of </think>
    think_end_idx = raw.rfind(_THINK_END) if has_think_end else -1

    boxes_after_think = [
        (s, e, c) for s, e, c in boxes_full
        if think_end_idx < 0 or s > think_end_idx
    ]
    boxes_before_think = [
        (s, e, c) for s, e, c in boxes_full
        if think_end_idx >= 0 and s < think_end_idx
    ]

    if not boxes_after_think and boxes_before_think:
        flags["box_in_think"] = True  # Issue D

    if len(boxes_after_think) > 1:
        flags["multi_box"] = True     # Issue B

    for _, _, content in boxes_after_think:
        # Check thousands separator
        for part in content.split(","):
            if _is_thousands_number(part.strip()):
                flags["thousands_sep"] = True

        # Check low precision decimal
        for part in content.split(","):
            part = part.strip()
            dm = re.fullmatch(r"-?\d+\.(\d+)", part)
            if dm and len(dm.group(1)) < 6:
                flags["low_precision"] = True

        # Check \pm expansion risk
        if "\\pm" in content or "\\mp" in content:
            flags["pm_expansion"] = True

    # Percentage confusion heuristic
    if "%" in question or "percent" in question.lower():
        for _, _, content in boxes_after_think:
            for part in content.split(","):
                part = part.strip()
                try:
                    val = float(part)
                    if 0.0 < val <= 1.0:
                        flags["pct_likely"] = True
                except ValueError:
                    pass

    return dict(flags)


# ---------------------------------------------------------------------------
# MCQ letter extractor (fallback for eval)
# ---------------------------------------------------------------------------

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str = MODEL_PATH,
    data_path: str = DATA_PATH,
    output_path: str = OUTPUT_PATH,
    gpu_id: str = GPU_ID,
):
    """
    Full end-to-end inference pipeline.

    Loads the model (fine-tuned checkpoint or base), runs inference on the
    dataset at data_path, applies the full post-processing pipeline, and
    writes the final submission CSV to output_path.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    do_eval = "public" in data_path.lower()

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading vLLM engine...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
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

    # Load judger only for public dataset (has ground-truth answers)
    judger = None
    if do_eval:
        print("Public dataset detected — loading Judger for evaluation.")
        from judger import Judger
        judger = Judger(strict_extract=False)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    # ── Build prompts ────────────────────────────────────────────────────────
    prompts = []
    for item in data:
        msgs = build_messages(item["question"], item.get("options"))
        prompts.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f"Generating {len(prompts)} responses...")
    outputs  = llm.generate(prompts, sampling_params=sampling_params)
    raw_responses = [out.outputs[0].text.strip() for out in outputs]

    # ── Post-process & evaluate ──────────────────────────────────────────────
    print("Post-processing and scoring...")
    results     = []
    error_tally = defaultdict(int)

    for item, raw in tqdm(zip(data, raw_responses), total=len(data)):
        is_mcq   = bool(item.get("options"))
        question = item["question"]

        # ── Diagnose raw response (before any fixes) ─────────────────────────
        flags = diagnose(raw, question)
        for flag, val in flags.items():
            if val:
                error_tally[flag] += 1

        # ── Post-processing pipeline ──────────────────────────────────────────

        # Step 1: bandaid (Issue B + D)
        response = apply_bandaid(raw)

        # Step 2: precision fixes (thousands separator + decimal precision)
        response = apply_precision_to_response(response, raw)

        # Step 3: Kaggle safety — sanitise \\boxed inside <think>
        if _THINK_END in response:
            parts        = response.split(_THINK_END)
            think_part   = parts[0].replace("\\boxed{", "boxed{")
            answer_part  = _THINK_END.join(parts[1:])
            kaggle_response = think_part + _THINK_END + answer_part
        else:
            kaggle_response = response

        # ── Build CSV record ─────────────────────────────────────────────────
        record = {"id": item.get("id"), "response": kaggle_response}

        if do_eval and "answer" in item:
            gold   = item["answer"]
            gold_list = gold if isinstance(gold, list) else [gold]

            # Text to evaluate: everything after </think>, or full response
            think_idx = kaggle_response.rfind(_THINK_END)
            eval_text = (
                kaggle_response[think_idx + len(_THINK_END):]
                if think_idx >= 0
                else kaggle_response
            )

            if is_mcq:
                correct = extract_letter(eval_text) == str(gold).strip().upper()
            else:
                try:
                    correct = judger.auto_judge(
                        pred=eval_text,
                        gold=gold_list,
                        options=[[]] * len(gold_list),
                    )
                except Exception:
                    correct = False

            record.update({"gold": gold, "is_mcq": is_mcq, "correct": correct})

            # Count mismatch check (post-fix)
            if not correct and not is_mcq:
                boxes = _extract_boxed_contents(eval_text)
                if boxes:
                    # Count sub-answers in boxed vs gold
                    boxed_content = ", ".join(c for _, _, c in boxes)
                    pred_count    = len(boxed_content.split(","))
                    gold_count    = len(gold_list)
                    if pred_count != gold_count:
                        error_tally["count_mismatch"] += 1
                        record["error_count_mismatch"] = True

        results.append(record)

    # ── Print metrics ─────────────────────────────────────────────────────────
    if do_eval:
        total   = len(results)
        correct_all  = sum(1 for r in results if r.get("correct"))
        mcq_res = [r for r in results if r.get("is_mcq")]
        ff_res  = [r for r in results if not r.get("is_mcq") and "correct" in r]

        print("\n" + "=" * 40)
        print("  INFERENCE RESULTS")
        print("=" * 40)
        print(f"  Overall accuracy  : {correct_all / total:.2%} ({correct_all}/{total})")
        if mcq_res:
            mcq_c = sum(1 for r in mcq_res if r["correct"])
            print(f"  MCQ accuracy      : {mcq_c / len(mcq_res):.2%} ({mcq_c}/{len(mcq_res)})")
        if ff_res:
            ff_c = sum(1 for r in ff_res if r["correct"])
            print(f"  Free-form accuracy: {ff_c / len(ff_res):.2%} ({ff_c}/{len(ff_res)})")

        print("\n  Error analysis (pre-fix, on raw responses):")
        print("  " + "-" * 36)
        error_labels = {
            "no_box":          "No \\boxed{} anywhere",
            "box_in_think":    "\\boxed{} only inside <think> (Issue D)",
            "multi_box":       "Multiple \\boxed{} after </think> (Issue B)",
            "thousands_sep":   "Thousands separator in boxed (e.g. 105,950)",
            "low_precision":   "Decimal with < 6 decimal places",
            "pm_expansion":    "\\pm in boxed (judger expands to 2 values)",
            "missing_think":   "No </think> tag produced",
            "pct_likely":      "Likely percentage/decimal confusion",
            "count_mismatch":  "Sub-answer count mismatch (post-fix)",
        }
        for key, label in error_labels.items():
            count = error_tally.get(key, 0)
            if count:
                print(f"  {label:<45} {count:>4} ({count / total:.1%})")
        print("=" * 40 + "\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(results)} records to {output_path}...")
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in results:
            writer.writerow({"id": r["id"], "response": r["response"]})

    print("Done.")


if __name__ == "__main__":
    run_inference()