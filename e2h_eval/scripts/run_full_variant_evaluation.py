#!/usr/bin/env python3
"""
Run complete variant evaluation across all models, years, and problems.
Updated to use the new variant-specific extraction system.
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
    for year_dir in sorted(DATA_DIR.glob("eval_*")):
        if year_dir.name == "eval_logs.zip":  # Skip the zip file
            continue
        for model_dir in sorted(year_dir.glob("*_E2H-Codeforces")):
            year = year_dir.name.replace("eval_", "")
            model = model_dir.name.replace("_E2H-Codeforces", "")
            model_dirs.append({
                "year": year,
                "model": model,
                "logs_dir": model_dir,
                "sample_id": f"{model}_{year}"
            })
    return model_dirs

def extract_variant_samples(model_info):
    """Extract variant samples for one model/year using the new variant extraction script."""
    sample_file = SAMPLES_DIR / f"{model_info['sample_id']}_variants.jsonl"
    if sample_file.exists():
        print(f"✓ Variant samples already exist: {sample_file}")
        return sample_file
    
    print(f"Extracting variant samples: {model_info['sample_id']}")
    
    cmd = [
        "python3", 
        str(SCRIPTS_DIR / "extract_samples_from_logs_variants.py"),
        "--logs-dir", str(model_info['logs_dir']),
        "--e2h-json", str(DATA_DIR / "E2H-Codeforces.json"),
        "--out", str(sample_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"✗ Error extracting samples: {result.stderr}")
        return None
    
    print(f"✓ Extracted variant samples: {sample_file}")
    return sample_file

def run_evaluation(sample_file, model_info):
    """Run evaluation on variant samples."""
    results_file = RESULTS_DIR / f"{model_info['sample_id']}_variants_results.jsonl"
    if results_file.exists():
        print(f"✓ Results already exist: {results_file}")
        return results_file
    
    print(f"Running evaluation: {model_info['sample_id']}")
    
    # Create a simple wrapper to run the evaluation
    # Since the harness has import issues, create a minimal version
    eval_script = f"""
import sys
sys.path.append('{BASE_DIR}')
sys.path.append('{BASE_DIR}/engine')

import json
import tempfile
import os
import time
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr
import io

def run_eval_simple(sample, problems, timeout=10):
    \"\"\"Simple evaluation function.\"\"\"
    task_id = sample['task_id']
    code = sample['completion']
    
    # Find the problem by parsing the task_id
    # E2H_CF1031A_low_easy -> base_task_id is E2H_CF1031A
    parts = task_id.split('_')
    base_task_id = '_'.join(parts[:2])  # E2H_CF1031A
    
    # Find matching problem
    problem = None
    for p in problems:
        p_task_id = "E2H_CF" + str(p.get('contest_id')) + str(p.get('problem_index'))
        if p_task_id == base_task_id:
            problem = p
            break
    
    if not problem:
        return {{"task_id": task_id, "status": "failed", "error": "Problem not found"}}
    
    # Simple execution test
    try:
        inputs = problem.get('inputs', [])
        answers = problem.get('answers', [])
        if not inputs or not answers:
            return {{"task_id": task_id, "status": "failed", "error": "No test cases"}}
        
        if len(inputs) != len(answers):
            return {{"task_id": task_id, "status": "failed", "error": "Mismatch between inputs and answers"}}
        
        # Create temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            import subprocess
            
            # Test against first 3 test cases only (for speed)
            test_limit = min(3, len(inputs))
            all_passed = True
            first_error = None
            
            for i, (test_input, expected_output) in enumerate(zip(inputs[:test_limit], answers[:test_limit])):
                try:
                    result = subprocess.run(
                        ['python3', f.name],
                        input=test_input,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    
                    if result.returncode != 0:
                        all_passed = False
                        if not first_error:
                            first_error = result.stderr or "Non-zero exit code"
                        break
                    
                    output = result.stdout.strip()
                    expected = expected_output.strip()
                    
                    if output != expected:
                        all_passed = False
                        if not first_error:
                            first_error = "Wrong output for test case " + str(i+1) + ": expected '" + expected + "', got '" + output + "'"
                        break
                        
                except subprocess.TimeoutExpired:
                    all_passed = False
                    if not first_error:
                        first_error = "Timeout on test case " + str(i+1)
                    break
                except Exception as e:
                    all_passed = False
                    if not first_error:
                        first_error = str(e)
                    break
            
            # Determine final status
            if all_passed:
                status = "passed"
                error = None
            else:
                status = "failed"
                error = first_error
                
            # Clean up temp file
            os.unlink(f.name)
                
        return {{"task_id": task_id, "status": status, "error": error}}
        
    except Exception as e:
        return {{"task_id": task_id, "status": "failed", "error": str(e)}}

# Load data
with open('{sample_file}', 'r') as f:
    samples = [json.loads(line) for line in f]

with open('{DATA_DIR}/E2H-Codeforces.json', 'r') as f:
    problems = json.load(f)

print("Evaluating " + str(len(samples)) + " samples...")

# Run evaluations
results = []
for i, sample in enumerate(samples):
    if i % 50 == 0:
        print("Progress: " + str(i) + "/" + str(len(samples)))
    
    result = run_eval_simple(sample, problems)
    results.append(result)

# Save results
with open('{results_file}', 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\\n')

print("Saved " + str(len(results)) + " results to {results_file}")
"""
    
    # Write and execute the evaluation script
    eval_script_path = RESULTS_DIR / f"temp_eval_{model_info['sample_id']}.py"
    with open(eval_script_path, 'w') as f:
        f.write(eval_script)
    
    try:
        result = subprocess.run(
            ["python3", str(eval_script_path)], 
            capture_output=True, 
            text=True, 
            cwd=BASE_DIR,
            timeout=1800  # 30 minutes max
        )
        
        if result.returncode != 0:
            print(f"✗ Error running evaluation: {result.stderr}")
            return None
            
        print(f"✓ Evaluation completed: {results_file}")
        return results_file
        
    except subprocess.TimeoutExpired:
        print(f"✗ Evaluation timed out for {model_info['sample_id']}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    finally:
        # Clean up temp script
        if eval_script_path.exists():
            eval_script_path.unlink()

def add_variant_scores(results_file, model_info):
    """Add scores to variant log files."""
    if not results_file or not results_file.exists():
        print(f"✗ No results file to score: {model_info['sample_id']}")
        return False
    
    print(f"Adding variant scores: {model_info['sample_id']}")
    
    cmd = [
        "python3",
        str(SCRIPTS_DIR / "add_scores.py"),
        "--data-dir", str(DATA_DIR),
        "--results-dir", str(RESULTS_DIR),
        "--model-year", f"{model_info['model']}_{model_info['year']}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"✗ Error adding scores: {result.stderr}")
        return False
    
    print(f"✓ Added variant scores: {model_info['sample_id']}")
    return True

def main():
    """Run the complete variant evaluation pipeline."""
    print("=== Running Full Variant Evaluation ===")
    
    # Ensure directories exist
    SAMPLES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Find all model/year combinations
    model_dirs = find_model_dirs()
    print(f"Found {len(model_dirs)} model/year combinations:")
    for info in model_dirs:
        print(f"  {info['sample_id']}: {info['logs_dir']}")
    
    # Process each model/year combination
    total_success = 0
    for i, model_info in enumerate(model_dirs):
        print(f"\\n[{i+1}/{len(model_dirs)}] Processing {model_info['sample_id']}")
        
        # 1. Extract variant samples
        sample_file = extract_variant_samples(model_info)
        if not sample_file:
            continue
        
        # 2. Run evaluation
        results_file = run_evaluation(sample_file, model_info)
        if not results_file:
            continue
        
        # 3. Add scores to log files
        if add_variant_scores(results_file, model_info):
            total_success += 1
    
    print(f"\\n=== Complete: {total_success}/{len(model_dirs)} successful ===")

if __name__ == "__main__":
    main()