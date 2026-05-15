"""
01_prepare_data.py

Reads training_data.json, handles all answer/question types,
formats to the Qwen <think> schema, and writes a 70/15/15 train/val/test split.

Question types handled:
  - Free-form / fill-in-the-blank  (answer is a list of values)
  - Multiple choice                (answer is a letter; options list is included in prompt)
  - Single numerical answer        (answer is a string or single-element list)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from pathlib import Path

import config


# ── Question type detection ───────────────────────────────────────────────────

MULTIPLE_CHOICE_LETTERS = set("ABCDEFGHIJ")

def is_multiple_choice(item: dict) -> bool:
    """True if the answer is a single letter and options are present."""
    answer = item.get("answer", "")
    if isinstance(answer, list):
        return False
    return str(answer).strip().upper() in MULTIPLE_CHOICE_LETTERS and bool(item.get("options"))


def is_multi_answer(item: dict) -> bool:
    return isinstance(item.get("answer"), list) and len(item["answer"]) > 1


# ── Answer normalisation ──────────────────────────────────────────────────────

def format_answer_string(answer) -> str:
    """Return a clean string representation of the ground truth answer."""
    if isinstance(answer, list):
        if len(answer) == 1:
            return str(answer[0]).strip()
        return ", ".join(str(a).strip() for a in answer)
    return str(answer).strip()


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    """
    Build the user-facing prompt.
    For multiple-choice questions the options are appended so the model
    can see them at both training and inference time.
    """
    question = item["question"].strip()

    if is_multiple_choice(item):
        options_block = "\n".join(
            f"{chr(65 + i)}. {opt}"
            for i, opt in enumerate(item["options"])
        )
        return f"{question}\n\nOptions:\n{options_block}"

    return question


# ── Reasoning trace construction ─────────────────────────────────────────────

def build_think_block(item: dict, answer_str: str) -> str:
    """
    Construct a minimal <think> block.
    We don't have ground-truth reasoning traces, so we keep this short.
    The synthetic data (if used later) provides the heavy reasoning scaffolding.
    """
    question = item["question"].strip()

    if is_multiple_choice(item):
        chosen_idx = MULTIPLE_CHOICE_LETTERS.intersection(answer_str.upper())
        letter = answer_str.strip().upper()
        try:
            idx = ord(letter) - ord('A')
            chosen_text = item["options"][idx] if idx < len(item["options"]) else ""
            return (
                f"The problem presents multiple options. "
                f"Evaluating each option against the question:\n\n"
                f"{question}\n\n"
                f"The correct choice is option {letter}: {chosen_text}."
            )
        except Exception:
            return f"Evaluating the options, the correct answer is {letter}."

    if is_multi_answer(item):
        answers = [str(a).strip() for a in item["answer"]]
        placeholders = question.count("[ANS]")
        parts = [f"For part {i+1}: the answer is {a}." for i, a in enumerate(answers)]
        return (
            f"This problem has {placeholders} parts requiring {len(answers)} answers.\n\n"
            + "\n".join(parts)
        )

    return (
        f"Working through the problem step by step:\n\n"
        f"{question}\n\n"
        f"The final answer is {answer_str}."
    )


# ── Qwen format ───────────────────────────────────────────────────────────────

def format_qwen(item: dict) -> dict:
    prompt     = build_prompt(item)
    answer_str = format_answer_string(item["answer"])
    reasoning  = build_think_block(item, answer_str)

    text = (
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        f"{reasoning}\n"
        "</think>\n"
        f"{answer_str}\n"
        "<|im_end|>"
    )
    return {
        "text":        text,
        "id":          item.get("id"),
        "answer":      answer_str,
        "is_mc":       is_multiple_choice(item),
        "is_multi":    is_multi_answer(item),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Reading {config.INPUT_JSON}…")
    with open(config.INPUT_JSON) as f:
        raw = json.load(f)

    # Handle dict wrappers e.g. {"data": [...]}
    if isinstance(raw, dict):
        for key in ("data", "examples", "samples", "train"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    print(f"Loaded {len(raw):,} raw examples")

    # Format all examples
    records = []
    skipped = 0
    type_counts = {"multiple_choice": 0, "multi_answer": 0, "single_answer": 0}

    for item in raw:
        if not item.get("question") or item.get("answer") is None:
            skipped += 1
            continue

        record = format_qwen(item)
        records.append(record)

        if record["is_mc"]:
            type_counts["multiple_choice"] += 1
        elif record["is_multi"]:
            type_counts["multi_answer"] += 1
        else:
            type_counts["single_answer"] += 1

    print(f"Formatted {len(records):,} examples  (skipped {skipped} with missing fields)")
    print(f"  Multiple choice:  {type_counts['multiple_choice']:,}")
    print(f"  Multi-part:       {type_counts['multi_answer']:,}")
    print(f"  Single answer:    {type_counts['single_answer']:,}")

    # Shuffle then split 70 / 15 / 15
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(records)

    n       = len(records)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    # test gets the remainder to avoid rounding loss
    n_test  = n - n_train - n_val

    train = records[:n_train]
    val   = records[n_train : n_train + n_val]
    test  = records[n_train + n_val :]

    def write_jsonl(path, items):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    write_jsonl(config.TRAIN_FILE, train)
    write_jsonl(config.VAL_FILE,   val)
    write_jsonl(config.TEST_FILE,  test)

    print(f"\nSplit complete:")
    print(f"  Train: {len(train):,} → {config.TRAIN_FILE}")
    print(f"  Val:   {len(val):,}   → {config.VAL_FILE}")
    print(f"  Test:  {len(test):,}  → {config.TEST_FILE}")
    print("\nNOTE: val and test sets are held-out — never train on them.")


if __name__ == "__main__":
    main()
