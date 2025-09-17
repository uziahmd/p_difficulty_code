#!/usr/bin/env python3
import argparse, json, os, re, sys

FILENAME_RE = re.compile(r"^(\d+)_none_none\.json$")

def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True, help="Path to logs folder for ONE model run")
    ap.add_argument("--e2h-json", required=True, help="Path to E2H-Codeforces.json")
    ap.add_argument("--out", required=True, help="Output samples jsonl")
    args = ap.parse_args()

    with open(args.e2h_json, "r", encoding="utf-8") as f:
        problems = json.load(f)

    files = []
    for name in os.listdir(args.logs_dir):
        m = FILENAME_RE.match(name)
        if m:
            files.append((int(m.group(1)), name))
    files.sort()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for idx, fname in files:
            path = os.path.join(args.logs_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    rec = json.load(f)
                except Exception as e:
                    print(f"Skip {fname}: invalid JSON ({e})")
                    continue
            code = rec.get("extracted_answer") or rec.get("response")
            if not isinstance(code, str):
                print(f"Skip {fname}: no string 'response' or 'extracted_answer'")
                continue
            code = strip_fences(code)

            # Map 1-based index to E2H problem to get task_id
            if idx < 1 or idx > len(problems):
                print(f"Skip {fname}: index {idx} out of range")
                continue
            obj = problems[idx - 1]
            task_id = f"E2H_CF{obj.get('contest_id')}{obj.get('problem_index')}"

            row = {"task_id": task_id, "completion": code}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote samples to {args.out}")

if __name__ == "__main__":
    main()
