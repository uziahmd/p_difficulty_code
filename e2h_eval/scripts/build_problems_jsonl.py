#!/usr/bin/env python3
import argparse, json, os, sys
from typing import List, Tuple

TEST_HEADER = """# --- Auto-generated HumanEval-style IO test harness ---
def _run_script_once(src: str, stdin_text: str) -> str:
    import sys, io
    # Create globals that mimic the __main__ module environment
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    old_stdin, old_stdout = sys.stdin, sys.stdout
    
    # Use StringIO for both input patterns: line-by-line input() and bulk stdin.read()
    stdin_io = io.StringIO(stdin_text)
    stdout_io = io.StringIO()
    sys.stdin, sys.stdout = stdin_io, stdout_io
    
    try:
        exec(src, g)
        return sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout

def _norm(s: str) -> str:
    return "\\n".join(line.rstrip() for line in s.strip().splitlines())

def check(candidate_src: str):
    for (inp, expected) in CASES:
        out = _run_script_once(candidate_src, inp)
        assert _norm(out) == _norm(expected), f"Expected {repr(_norm(expected))}, got {repr(_norm(out))}"
"""

def norm_line_ending(s: str) -> str:
    # Normalize to single '\n' and ensure exactly one trailing '\n'
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if not s.endswith("\n"):
        s += "\n"
    return s

def cases_from_obj(obj) -> List[Tuple[str, str]]:
    # Prefer bundled input_output if present
    inputs, outputs = None, None
    if isinstance(obj.get("input_output"), dict):
        inputs = obj["input_output"].get("inputs")
        outputs = obj["input_output"].get("outputs")
    if not inputs or not outputs:
        # fallbacks
        inputs = inputs or obj.get("inputs") or obj.get("sample_inputs")
        outputs = outputs or obj.get("answers") or obj.get("sample_outputs")
    if not inputs or not outputs or len(inputs) != len(outputs):
        return []
    pairs = []
    for inp, out in zip(inputs, outputs):
        pairs.append((norm_line_ending(inp), norm_line_ending(out)))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2h-json", required=True, help="Path to E2H-Codeforces.json")
    ap.add_argument("--out", required=True, help="Output problems jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Only first N problems (debug)")
    args = ap.parse_args()

    with open(args.e2h_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = len(data)
    if args.limit and args.limit < n:
        n = args.limit

    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(n):
            obj = data[i]
            contest_id = obj.get("contest_id")
            idx = obj.get("problem_index")
            task_id = f"E2H_CF{contest_id}{idx}"
            cases = cases_from_obj(obj)
            if not cases:
                # skip if no cases
                continue
            # Build Python literal for CASES
            cases_py = repr(cases)
            test_code = f"CASES = {cases_py}\n\n{TEST_HEADER}"
            prompt = "# E2H Codeforces I/O task. Candidate is a script using input()/print().\n"
            entry_point = "check"
            row = {
                "task_id": task_id,
                "prompt": prompt,
                "entry_point": entry_point,
                "test": test_code,
                # pass-through metadata (optional):
                "name": obj.get("problem_name"),
                "tag": obj.get("tag"),
                "detailed_tag": obj.get("detailed_tag"),
                "rating": obj.get("unnorm_rating") or obj.get("rating"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote problems to {args.out}")

if __name__ == "__main__":
    main()
