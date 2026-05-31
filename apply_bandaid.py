"""
apply_bandaid.py
================
Applies Issue-B and Issue-D formatting fixes to an existing sft_results.csv,
producing a new CSV ready for submission or re-evaluation.

  Issue B: multiple \\boxed{} in the post-think section
           → merged into one \\boxed{v1, v2, ...}
  Issue D: \\boxed{} only inside <think>, nothing after </think>
           → answer rescued into post-think section

Usage
-----
    python apply_bandaid.py                          # uses defaults below
    python apply_bandaid.py <input.csv> <output.csv>
"""

import csv
import sys
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Bandaid logic (self-contained copy — no external import needed)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> list:
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


def _remove_all_boxed(text: str) -> str:
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


def apply_bandaid(response: str) -> str:
    """
    Fix Issue B and Issue D in a single response string.
    Operates on kaggle_safe_response format:
      - think section uses 'boxed{' (no backslash)
      - post-think section uses '\\boxed{' (normal LaTeX)
    """
    if "</think>" in response:
        split_idx   = response.rfind("</think>")
        think_part  = response[:split_idx]
        post_think  = response[split_idx + len("</think>"):]

        entries = _extract_boxed(post_think)

        # Issue D — no box after </think>
        if not entries:
            think_restored = think_part.replace("boxed{", "\\boxed{")
            think_entries  = _extract_boxed(think_restored)
            if think_entries:
                unique = list(dict.fromkeys(e[2] for e in think_entries))
                merged = ", ".join(unique)
                return (think_part + "</think>"
                        + post_think.rstrip()
                        + f"\n\n\\boxed{{{merged}}}")
            return response                       # truncated — nothing to rescue

        if len(entries) == 1:
            return response                       # already correct

        # Issue B — multiple boxes: deduplicate then merge
        unique = list(dict.fromkeys(e[2] for e in entries))
        merged = ", ".join(unique)
        clean_post = _remove_all_boxed(post_think)
        return (think_part + "</think>"
                + clean_post.rstrip()
                + f"\n\n\\boxed{{{merged}}}")

    else:
        entries = _extract_boxed(response)
        if not entries or len(entries) == 1:
            return response
        unique = list(dict.fromkeys(e[2] for e in entries))
        merged = ", ".join(unique)
        clean  = _remove_all_boxed(response)
        return clean.rstrip() + f"\n\n\\boxed{{{merged}}}"


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def main() -> None:
    input_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sft_submission_8k_max_token.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/sft_submission_bandaid_8k_max_token.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    changed = unchanged = issue_b = issue_d = 0

    with open(input_path,  encoding="utf-8", newline="") as f_in, \
         open(output_path, encoding="utf-8", newline="", mode="w") as f_out:

        reader  = csv.DictReader(f_in)
        writer  = csv.DictWriter(f_out, fieldnames=["id", "response"],
                                 quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for row in reader:
            original = row["response"]
            fixed    = apply_bandaid(original)

            # Track what type of fix was applied
            if fixed != original:
                changed += 1
                # Was it an Issue D rescue?
                if "</think>" in original:
                    post = original[original.rfind("</think>") + len("</think>"):]
                    if "\\boxed{" not in post:
                        issue_d += 1
                    else:
                        issue_b += 1
            else:
                unchanged += 1

            writer.writerow({"id": row["id"], "response": fixed})

    total = changed + unchanged
    print("\n" + "=" * 42)
    print("  apply_bandaid.py  —  complete")
    print("=" * 42)
    print(f"  Total rows processed : {total}")
    print(f"  Unchanged            : {unchanged}")
    print(f"  Fixed (total)        : {changed}")
    print(f"    Issue B (multi-box): {issue_b}")
    print(f"    Issue D (box rescue): {issue_d}")
    print(f"  Output               : {output_path}")
    print("=" * 42)


if __name__ == "__main__":
    main()