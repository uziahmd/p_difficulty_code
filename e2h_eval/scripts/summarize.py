#!/usr/bin/env python3
import os, json, csv, statistics as stats
from typing import Dict, List

PROBLEMS_JSONL = os.path.join("problems", "e2h_problems.jsonl")
RESULTS_DIR = "results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_runs.csv")

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def infer_run_id_from_filename(filename: str) -> str:
    # expects "<base>_results.jsonl" -> return "<base>"
    base = os.path.basename(filename)
    if base.endswith("_results.jsonl"):
        return base[:-len("_results.jsonl")]
    return base

def write_csv(path: str, rows: List[dict], fieldnames: List[str]):
    """Write list of dicts to CSV with specified fieldnames."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    print("📊 Generating basic summary statistics...")
    
    # Find all results files
    results_files = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith("_results.jsonl"):
            results_files.append(os.path.join(RESULTS_DIR, fname))
    
    if not results_files:
        print("No results files found in results/ directory")
        return
    
    # Process each results file
    summary_rows = []
    
    for results_file in sorted(results_files):
        run_id = infer_run_id_from_filename(results_file)
        print(f"Processing: {run_id}")
        
        # Load results
        results = load_jsonl(results_file)
        if not results:
            continue
        
        # Basic statistics
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "passed")
        failed = sum(1 for r in results if r["status"] == "failed")
        timed_out = sum(1 for r in results if r["status"] == "timed out")
        pass_at_1 = passed / total if total > 0 else 0.0
        
        # Timing statistics (only for passed tests)
        passed_times = [r["elapsed_ms"] for r in results if r["status"] == "passed" and r.get("elapsed_ms") is not None]
        avg_ms = stats.mean(passed_times) if passed_times else 0.0
        median_ms = stats.median(passed_times) if passed_times else 0.0
        
        summary_rows.append({
            "run_id": run_id,
            "total": total,
            "passed": passed,
            "failed": failed,
            "timed_out": timed_out,
            "pass_at_1": f"{pass_at_1:.3f}",
            "avg_ms": f"{avg_ms:.1f}",
            "median_ms": f"{median_ms:.1f}"
        })
    
    # Write summary CSV
    if summary_rows:
        write_csv(SUMMARY_CSV, summary_rows, [
            "run_id", "total", "passed", "failed", "timed_out", 
            "pass_at_1", "avg_ms", "median_ms"
        ])
        print(f" Wrote: {SUMMARY_CSV}")
        print(f"Processed {len(summary_rows)} evaluation runs")
    else:
        print("No valid results found")
    
    print("\n💡 For comprehensive pass@k analysis and visualizations, run:")
    print("   python3 scripts/pass_k_analysis.py")

if __name__ == "__main__":
    main()
