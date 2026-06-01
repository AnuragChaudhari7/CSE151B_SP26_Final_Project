"""
Reads the raw synthetic SFT dataset (prompt-completion format) and produces
a corrected version where every completion has exactly ONE \\boxed{} after </think>.

Fixes applied to completions:
    - Issue B: multiple \\boxed{} in post-think section --> deduplicated and merged into \\boxed{v1, v2, ...}
    - Issue D: \\boxed{} only inside <think>, nothing after </think> --> last \\boxed{} from think section rescued into post-think

System prompts are also updated to the revised wording (with **single**
\\boxed{} instruction) that matches the inference script.

Usage
-----
    python fix_sft_dataset.py <input.jsonl> <output.jsonl>

    Where:
    input  = 8k_max_openr1_math_7_5k_stratified.jsonl
    output = data/sft_train_fixed.jsonl

Validation
----------
After writing, the script runs a full pass over the output and checks:
    - Every completion has exactly one \\boxed{} in the post-think section
    - No \\boxed{} content is empty or has unbalanced braces
    - The judger can successfully extract an answer from every completion
    - Summary statistics are printed
"""

import json
import re
import sys
from pathlib import Path

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a **single** \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "If the problem has multiple sub-answers, separate them by commas inside a **single** \\boxed{}, "
    "e.g. \\boxed{A,D}."
)

def extract_boxed(text: str):
    """
    Return (start, end, content) for every properly-closed \\boxed{} in text.
    Brace-depth matching handles nested braces like \\boxed{\\frac{1}{2}}.
    """
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start : i - 1].strip()
            if content:
                entries.append((idx, i, content))
        start = max(brace_start, i)
    return entries


def remove_all_boxed(text: str):
    """Strip every \\boxed{...} from text (brace-aware)."""
    parts, i = [], 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx < 0:
            parts.append(text[i:])
            break
        parts.append(text[i:idx])
        brace_start = idx + len("\\boxed{")
        depth, j = 1, brace_start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        i = j
    return "".join(parts)

def fix_completion(text: str):
    """
    Fix a completion so it has exactly one \\boxed{} after </think>.
    """
    if "</think>" in text:
        split_idx  = text.rfind("</think>")
        think_part = text[:split_idx]
        post_think = text[split_idx + len("</think>"):]

        entries = extract_boxed(post_think)

        # Already correct
        if len(entries) == 1:
            return text, "ok"

        # Issue D: no box after </think> — rescue from think section
        if len(entries) == 0:
            think_entries = extract_boxed(think_part)
            if think_entries:
                unique = list(dict.fromkeys(e[2] for e in think_entries))
                merged = ", ".join(unique)
                fixed  = (think_part + "</think>"
                          + post_think.rstrip()
                          + f"\n\n\\boxed{{{merged}}}")
                return fixed, "issue_d"
            return text, "truncated"

        # Issue B: multiple boxes — deduplicate and merge
        unique = list(dict.fromkeys(e[2] for e in entries))
        merged = ", ".join(unique)
        clean_post = remove_all_boxed(post_think)
        fixed = (think_part + "</think>"
                 + clean_post.rstrip()
                 + f"\n\n\\boxed{{{merged}}}")
        return fixed, "issue_b"

    else:
        # No think tags
        entries = extract_boxed(text)
        if len(entries) == 1:
            return text, "ok"
        if len(entries) == 0:
            return text, "truncated"
        unique = list(dict.fromkeys(e[2] for e in entries))
        merged = ", ".join(unique)
        clean  = remove_all_boxed(text)
        fixed  = clean.rstrip() + f"\n\n\\boxed{{{merged}}}"
        return fixed, "issue_b"


def update_system_prompt(messages: list[dict]):
    """Replace system message with updated wording."""
    updated = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if "single best answer" in content or "letter of your chosen" in content:
                updated.append({"role": "system", "content": SYSTEM_PROMPT_MCQ})
            else:
                updated.append({"role": "system", "content": SYSTEM_PROMPT_MATH})
        else:
            updated.append(msg)
    return updated


# Validation

def validate_output(output_path):
    """
    Read the fixed output file and verify judger-compliance:
    - Exactly one \\boxed{} in post-think section.
    - Box content is non-empty and has balanced braces.
    
    Summary stats printed at the end.
    """
    print("\nRunning validation on output file...")

    exactly_one = 0
    zero_box = 0
    multi_box = 0
    empty_content = 0
    unbalanced = 0
    total = 0

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            completion = record.get("completion", [])
            comp_text  = (completion[0].get("content", "")
                          if isinstance(completion, list) and completion
                          else str(completion))

            split_idx  = comp_text.rfind("</think>")
            post_think = (comp_text[split_idx + len("</think>"):]
                          if split_idx >= 0
                          else comp_text)

            entries = extract_boxed(post_think)

            if len(entries) == 0:
                zero_box += 1
            elif len(entries) > 1:
                multi_box += 1
            else:
                exactly_one += 1
                content = entries[0][2]
                if not content.strip():
                    empty_content += 1
                # Check brace balance in content
                depth = 0
                for ch in content:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                if depth != 0:
                    unbalanced += 1

    print()
    print("VALIDATION RESULTS: ")
    print(f"Total records          : {total}")
    print(f"Exactly one \\boxed{{}} : {exactly_one}  ({exactly_one/total:.1%})")
    print(f"Zero \\boxed{{}}        : {zero_box}   (truncated, unfixable)")
    print(f"Multiple \\boxed{{}}    : {multi_box}   (should be 0)")
    print(f"Empty box content    : {empty_content}")
    print(f"Unbalanced braces    : {unbalanced}")
    judger_ready = exactly_one - empty_content - unbalanced
    print()
    print(f"Judger-ready records : {judger_ready}  ({judger_ready/total:.1%})")

    if multi_box > 0:
        print(f"\n{multi_box} records still have multiple boxes, check fix_completion logic.")
    if zero_box > 0:
        print(f"\n {zero_box} truncated records could not be fixed, consider filtering these from the SFT training set.")


def main():
    input_path  = (Path(sys.argv[1]) if len(sys.argv) > 1
                   else Path("data/8k_max_openr1_math_7_5k_stratified.jsonl"))
    output_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                   else Path("data/sft_train_fixed.jsonl"))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "issue_b": 0, "issue_d": 0, "truncated": 0}
    parse_errors = 0
    written = 0

    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_num, raw_line in enumerate(f_in, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                print(f"[WARN] line {line_num}: JSON error — {exc}", file=sys.stderr)
                continue

            prompt     = record.get("prompt", [])
            completion = record.get("completion", [])

            if not prompt or not completion:
                continue

            # Get completion text
            if isinstance(completion, list) and completion:
                comp_text = completion[0].get("content", "")
            else:
                comp_text = str(completion)

            # Fix the completion
            fixed_text, change_type = fix_completion(comp_text)
            counts[change_type] += 1

            # Skip truncated completions
            if change_type == "truncated":
                continue

            # Update system prompt
            fixed_prompt = update_system_prompt(prompt)

            # Rebuild completion with fixed text
            if isinstance(completion, list) and completion:
                fixed_completion = [{"role": completion[0].get("role", "assistant"),
                                     "content": fixed_text}]
            else:
                fixed_completion = [{"role": "assistant", "content": fixed_text}]

            out = {
                "prompt":     fixed_prompt,
                "completion": fixed_completion,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    total_in = sum(counts.values()) + parse_errors
    print()
    print(f"Input records          : {total_in}")
    print(f"Parse errors           : {parse_errors}")
    print(f"Unchanged (ok)         : {counts['ok']}")
    print(f"Fixed Issue B (multi)  : {counts['issue_b']}")
    print(f"Fixed Issue D (rescue) : {counts['issue_d']}")
    print(f"Skipped (truncated)    : {counts['truncated']}")
    print(f"Written to output      : {written}")
    print(f"Output                 : {output_path}")

    # Run validation
    validate_output(output_path)


if __name__ == "__main__":
    main()
