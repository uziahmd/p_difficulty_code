#!/usr/bin/env python3
"""
Run complete evaluation across all models, years, and problems.
"""
import os
import subprocess
import sys
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"
SAMPLES_DIR = BASE_DIR / "samples"
RESULTS_DIR = BASE_DIR / "results"

def find_model_dirs():
    """Find all model/year combinations."""
    model_dirs = []
    for year_dir in sorted(DATA_DIR.glob("logs_*")):
        for model_dir in sorted(year_dir.glob("*_E2H-Codeforces")):
            year = year_dir.name.replace("logs_", "")
            model = model_dir.name.replace("_E2H-Codeforces", "")
            model_dirs.append({
                "year": year,
                "model": model,
                "logs_dir": model_dir,
                "sample_id": f"{model}_{year}"
            })
    return model_dirs

def extract_samples(model_info):
    """Extract samples for one model/year."""
    sample_file = SAMPLES_DIR / f"{model_info['sample_id']}.jsonl"
    if sample_file.exists():
        print(f"✓ Samples already exist: {sample_file}")
        return sample_file
    
    print(f"Extracting samples: {model_info['sample_id']}")
    cmd = [
        "python3", str(SCRIPTS_DIR / "extract_samples_from_logs.py"),
        "--logs-dir", str(model_info["logs_dir"]),
        "--e2h-json", str(DATA_DIR / "E2H-Codeforces.json"),
        "--out", str(sample_file)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f" Failed to extract {model_info['sample_id']}: {result.stderr}")
        return None
    print(f"✓ Extracted: {sample_file}")
    return sample_file

def run_evaluation(sample_file, model_id):
    """Run evaluation for one sample file."""
    results_file = RESULTS_DIR / f"{model_id}_results.jsonl"
    if results_file.exists():
        print(f"✓ Results already exist: {results_file}")
        return results_file
    
    print(f"Evaluating: {model_id}")
    cmd = [
        "python3", str(SCRIPTS_DIR / "run_eval.py"),
        "--problem_file", str(BASE_DIR / "problems" / "e2h_problems.jsonl"),
        "--samples_file", str(sample_file),
        "--timeout", "5.0",
        "--results_out", str(results_file)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to evaluate {model_id}: {result.stderr}")
        return None
    print(f"✓ Evaluated: {results_file}")
    return results_file

def main():
    print("Running complete E2H-Codeforces evaluation")
    print("=" * 50)
    
    # Ensure problems file exists
    problems_file = BASE_DIR / "problems" / "e2h_problems.jsonl"
    if not problems_file.exists():
        print(f"Building problems file...")
        cmd = [
            "python3", str(SCRIPTS_DIR / "build_problems_jsonl.py"),
            "--e2h-json", str(DATA_DIR / "E2H-Codeforces.json"),
            "--out", str(problems_file)
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ Built problems: {problems_file}")
    
    # Find all model combinations
    model_dirs = find_model_dirs()
    print(f"\n Found {len(model_dirs)} model/year combinations:")
    for info in model_dirs:
        print(f"  - {info['sample_id']}")
    
    # Extract samples for all models
    print(f"\n Extracting samples...")
    sample_files = []
    for info in model_dirs:
        sample_file = extract_samples(info)
        if sample_file:
            sample_files.append((sample_file, info['sample_id']))
    
    # Run evaluations
    print(f"\n Running evaluations...")
    results_files = []
    for sample_file, model_id in sample_files:
        results_file = run_evaluation(sample_file, model_id)
        if results_file:
            results_files.append(results_file)
    
    # Generate summary report
    print(f"\n Generating summary statistics...")
    cmd = ["python3", str(SCRIPTS_DIR / "summarize.py")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Summary CSV: {RESULTS_DIR / 'summary_runs.csv'}")
        print(" For comprehensive analysis, run: python3 scripts/pass_k_analysis.py")
    else:
        print(f" Failed to generate summary: {result.stderr}")
    
    print(f"\n Complete! Evaluated {len(results_files)} model/year combinations")
    print(f" Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()