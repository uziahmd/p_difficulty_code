#!/usr/bin/env python3
import argparse, json, os, sys, time

# Ensure we can import the sibling 'engine' package when running this script directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # .../e2h_eval
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.harness import run_one

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def problems_by_task_id(path):
    d = {}
    for row in load_jsonl(path):
        d[row["task_id"]] = row
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_file", required=True)
    ap.add_argument("--samples_file", required=True)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--results_out", default="")
    ap.add_argument("--max", type=int, default=0, help="limit samples for quick run")
    args = ap.parse_args()

    probs = problems_by_task_id(args.problem_file)
    samples = load_jsonl(args.samples_file)
    if args.max and args.max < len(samples):
        samples = samples[:args.max]

    if not args.results_out:
        base = os.path.splitext(os.path.basename(args.samples_file))[0]
        args.results_out = os.path.join("results", f"{base}_results.jsonl")
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)

    passed = failed = timed = 0
    t0 = time.time()

    with open(args.results_out, "w", encoding="utf-8") as outf:
        for i, s in enumerate(samples, 1):
            task_id = s["task_id"]
            prob = probs.get(task_id)
            if not prob:
                print(f"[{i}/{len(samples)}] MISSING PROBLEM for task_id={task_id} -- skip")
                continue
            completion = s["completion"]
            r = run_one(prob, completion, timeout_s=args.timeout)
            status = r["status"]
            if status == "passed": passed += 1
            elif status == "timed out": timed += 1
            else: failed += 1

            row = {
                "task_id": task_id,
                "status": status,
                "elapsed_ms": r.get("ms"),
                "error": r.get("err"),
            }
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(samples)}] {task_id}: {status} ({r.get('ms')} ms)")

    dt = time.time() - t0
    total = passed + failed + timed
    pass_at_1 = (passed / total) if total else 0.0
    print(f"\nDone in {dt:.1f}s. total={total} passed={passed} failed={failed} timed_out={timed}")
    print(f"pass@1 = {pass_at_1:.3f}")
    print(f"Results -> {args.results_out}")
    print(f"\nNote: This is pass@1 because we have 1 sample per problem.")
    print(f"For pass@k (k>1), you need multiple samples per problem or use calculate_pass_at_k.py")

if __name__ == "__main__":
    main()
