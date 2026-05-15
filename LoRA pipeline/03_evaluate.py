"""
03_evaluate.py

Evaluates the best checkpoint on either the val or test split.
Handles all three answer types: multiple-choice, multi-part, single.

Usage:
    python scripts/03_evaluate.py                    # runs on test split by default
    python scripts/03_evaluate.py --split val        # run on val split
    python scripts/03_evaluate.py --checkpoint path/to/checkpoint
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# ── Answer extraction ─────────────────────────────────────────────────────────

MULTIPLE_CHOICE_LETTERS = set("ABCDEFGHIJ")

def extract_predicted_answers(generation: str) -> list[str]:
    """
    Extract all answers from the model's generation.
    Searches only in the text after </think>.
    Returns a list (to handle multi-part answers uniformly).
    """
    # Search after </think> if present
    think_end = generation.rfind("</think>")
    text = generation[think_end + len("</think>"):] if think_end != -1 else generation

    # Strip special tokens
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL).strip()

    if not text:
        return []

    # Try to extract a list of comma-separated answers (for multi-part)
    # Split on ", " or newline boundaries
    parts = [p.strip() for p in re.split(r",\s*|\n", text) if p.strip()]

    return parts if parts else [text.strip()]


def normalise(s: str) -> str:
    """Normalise an answer string for comparison."""
    s = s.strip().lower()
    s = s.replace(",", "").replace(" ", "")
    # Try numeric normalisation (so "8.0" == "8")
    try:
        return str(float(s))
    except ValueError:
        return s


def single_match(pred: str, gold: str) -> bool:
    return normalise(pred) == normalise(gold)


def score_prediction(predicted: list[str], gold_str: str, is_mc: bool, is_multi: bool) -> dict:
    """
    Score the prediction against the ground truth.
    Returns a dict with 'correct' (bool) and 'match_detail'.

    For multi-part: all sub-answers must match (strict).
    For single/MC:  first extracted answer must match.
    """
    gold_parts = [g.strip() for g in gold_str.split(",")]

    if is_multi:
        if len(predicted) < len(gold_parts):
            return {"correct": False, "match_detail": f"expected {len(gold_parts)} answers, got {len(predicted)}"}
        # Match each gold part against corresponding predicted part
        matches = [single_match(predicted[i], gold_parts[i]) for i in range(len(gold_parts))]
        all_correct = all(matches)
        return {
            "correct": all_correct,
            "match_detail": f"{sum(matches)}/{len(matches)} sub-answers correct",
        }
    else:
        pred_first = predicted[0] if predicted else ""
        correct = single_match(pred_first, gold_str)
        return {"correct": correct, "match_detail": "exact match" if correct else "mismatch"}


# ── Load split ────────────────────────────────────────────────────────────────

def load_split(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # Reconstruct the prompt (everything up to <|im_start|>assistant)
            text = item["text"]
            split_token = "<|im_start|>assistant"
            prompt_end = text.find(split_token)
            if prompt_end == -1:
                continue

            records.append({
                "prompt":    text[:prompt_end + len(split_token)] + "\n",
                "gold":      item["answer"],
                "is_mc":     item.get("is_mc", False),
                "is_multi":  item.get("is_multi", False),
                "id":        item.get("id"),
                "full_text": text,
            })
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint. Defaults to best_checkpoint.txt.")
    args = parser.parse_args()

    # Resolve checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        best_ckpt_file = Path(config.OUTPUT_DIR) / "best_checkpoint.txt"
        if best_ckpt_file.exists():
            checkpoint_path = best_ckpt_file.read_text().strip()
        else:
            raise FileNotFoundError("No checkpoint found. Run 02_train.py first.")
    print(f"Checkpoint:  {checkpoint_path}")

    split_file = config.VAL_FILE if args.split == "val" else config.TEST_FILE
    print(f"Evaluating:  {args.split} split  ({split_file})")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.BF16 else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    examples = load_split(split_file)
    print(f"Examples:    {len(examples):,}\n")

    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    results      = []
    correct      = 0
    type_correct = {"multiple_choice": [0, 0], "multi_answer": [0, 0], "single_answer": [0, 0]}

    for i, ex in enumerate(examples):
        inputs = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generation  = tokenizer.decode(new_tokens, skip_special_tokens=False)

        predicted   = extract_predicted_answers(generation)
        score       = score_prediction(predicted, ex["gold"], ex["is_mc"], ex["is_multi"])

        correct += int(score["correct"])

        # Track by type
        if ex["is_mc"]:
            t = "multiple_choice"
        elif ex["is_multi"]:
            t = "multi_answer"
        else:
            t = "single_answer"
        type_correct[t][0] += int(score["correct"])
        type_correct[t][1] += 1

        results.append({
            "id":           ex["id"],
            "gold":         ex["gold"],
            "predicted":    predicted,
            "correct":      score["correct"],
            "match_detail": score["match_detail"],
            "generation":   generation,
            "is_mc":        ex["is_mc"],
            "is_multi":     ex["is_multi"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"  [{i+1:>4}/{len(examples)}]  running accuracy: {correct/(i+1)*100:.1f}%")

    # ── Report ────────────────────────────────────────────────────────────────
    accuracy = correct / len(examples) if examples else 0.0
    print(f"\n{'='*55}")
    print(f"Overall accuracy:  {correct}/{len(examples)} = {accuracy*100:.2f}%")
    print(f"{'='*55}")
    for t, (c, n) in type_correct.items():
        if n > 0:
            print(f"  {t:<20}  {c}/{n} = {c/n*100:.1f}%")
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    results_path = Path(config.RESULTS_DIR) / f"eval_{args.split}.json"
    with results_path.open("w") as f:
        json.dump({
            "checkpoint":    checkpoint_path,
            "split":         args.split,
            "n_examples":    len(examples),
            "n_correct":     correct,
            "accuracy":      accuracy,
            "by_type":       {t: {"correct": c, "total": n, "accuracy": c/n if n else 0}
                              for t, (c, n) in type_correct.items()},
            "results":       results,
        }, f, indent=2)
    print(f"Full results → {results_path}")

    # ── Sample failures ───────────────────────────────────────────────────────
    failures = [r for r in results if not r["correct"]][:3]
    if failures:
        print("\nSample failures:")
        for r in failures:
            print(f"  ID:        {r['id']}")
            print(f"  Gold:      {r['gold']}")
            print(f"  Predicted: {r['predicted']}")
            print(f"  Detail:    {r['match_detail']}")
            print()


if __name__ == "__main__":
    main()
