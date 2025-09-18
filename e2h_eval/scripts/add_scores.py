#!/usr/bin/env python3
"""
Update 'score' field in original log files based on evaluation results:
- score = -1: compilation error (failed status with compilation-related error)
- score = 0: runs but incorrect (failed status with runtime/logic error)
- score = 1: correct (passed status)
"""
import argparse
import json
import os
import sys
from pathlib import Path
import glob

def load_jsonl(path):
    """Load JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def is_compilation_error(error_msg):
    """Check if error message indicates compilation/syntax error."""
    if not error_msg:
        return False
    
    error_msg = error_msg.lower()
    compilation_indicators = [
        "syntaxerror",
        "indentationerror", 
        "tabserror",
        "nameerror",
        "importerror",
        "modulenotfounderror",
        "invalid syntax",
        "unexpected indent",
        "unindent does not match",
        "inconsistent use of tabs and spaces"
    ]
    
    return any(indicator in error_msg for indicator in compilation_indicators)

def load_results_mapping(results_dir):
    """Load all result files and create a mapping from task_id to evaluation results."""
    results_mapping = {}
    
    result_files = list(Path(results_dir).glob("*_results.jsonl"))
    
    for result_file in result_files:
        # Extract model and year from filename
        filename = result_file.stem  # removes .jsonl
        model_year = filename.replace("_results", "")
        
        # Load results
        results = load_jsonl(str(result_file))
        
        # Create mapping for this model/year combination
        if model_year not in results_mapping:
            results_mapping[model_year] = {}
            
        for result in results:
            task_id = result.get("task_id", "")
            status = result.get("status", "failed")
            error = result.get("error", "")
            
            # Determine score based on evaluation
            if status == "passed":
                score = 1
            elif status == "timed out":
                score = 0  # Runtime error (could run but took too long)
            elif status == "failed":
                if is_compilation_error(error):
                    score = -1
                else:
                    score = 0
            else:
                score = 0  # Default to runtime error
                
            results_mapping[model_year][task_id] = score
    
    return results_mapping

def update_log_file(log_file_path, task_id, new_score):
    """Update the score field in a log file."""
    try:
        # Read the log file
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # Update the score
        log_data['score'] = new_score
        
        # Write back to file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
            
        return True
    except Exception as e:
        print(f"Error updating {log_file_path}: {e}")
        return False

def create_task_to_problem_mapping():
    """Create mapping from task_id to problem number (1-20)."""
    mapping = {
        "E2H_CF1031A": 1, "E2H_CF151A": 2, "E2H_CF404A": 3, "E2H_CF339B": 4, 
        "E2H_CF492B": 5, "E2H_CF88A": 6, "E2H_CF173A": 7, "E2H_CF633B": 8,
        "E2H_CF1141D": 9, "E2H_CF1767D": 10, "E2H_CF822C": 11, "E2H_CF498A": 12,
        "E2H_CF1846E2": 13, "E2H_CF1092C": 14, "E2H_CF270E": 15, "E2H_CF1146D": 16,
        "E2H_CF808E": 17, "E2H_CF980E": 18, "E2H_CF409I": 19, "E2H_CF1709F": 20
    }
    return mapping

def process_model_logs(data_dir, model_year, task_scores):
    """Process all log files for a specific model/year combination."""
    
    # Parse model_year to extract model and year
    if model_year.endswith(('_2025', '_2026', '_2027', '_2028')):
        year = model_year[-4:]
        model = model_year[:-5]
    else:
        print(f"Warning: Could not parse year from {model_year}")
        return 0
    
    # Find the logs directory for this model/year
    logs_dir = Path(data_dir) / f"logs_{year}" / f"{model}_E2H-Codeforces"
    
    if not logs_dir.exists():
        print(f"Warning: Logs directory does not exist: {logs_dir}")
        return 0
    
    # Create task_id to problem number mapping
    task_mapping = create_task_to_problem_mapping()
    
    updated_count = 0
    
    # Process each task
    for task_id, score in task_scores.items():
        # Get problem number from task_id using mapping
        if task_id not in task_mapping:
            print(f"Warning: Unknown task_id: {task_id}")
            continue
            
        problem_num = task_mapping[task_id]
        
        # Find all log files for this problem (different difficulty variants)
        log_files = list(logs_dir.glob(f"{problem_num}_*.json"))
        
        if not log_files:
            print(f"Warning: No log files found for problem {problem_num} in {logs_dir}")
            continue
        
        # Update all variants of this problem
        for log_file in log_files:
            if update_log_file(log_file, task_id, score):
                updated_count += 1
            
    return updated_count

def main():
    parser = argparse.ArgumentParser(description="Update score field in original log files based on evaluation results")
    parser.add_argument("--data-dir", default="data", help="Data directory containing logs_202* folders")
    parser.add_argument("--results-dir", default="results", help="Results directory containing evaluation results")
    parser.add_argument("--model-year", help="Specific model/year to process (e.g., 'gpt-5-mini-2025-08-07_2025')")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
        
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    print("Loading evaluation results...")
    results_mapping = load_results_mapping(results_dir)
    
    if not results_mapping:
        print("No evaluation results found!")
        sys.exit(1)
    
    print(f"Found evaluation results for {len(results_mapping)} model/year combinations")
    
    total_updated = 0
    
    if args.model_year:
        # Process specific model/year
        if args.model_year in results_mapping:
            task_scores = results_mapping[args.model_year]
            updated = process_model_logs(data_dir, args.model_year, task_scores)
            total_updated += updated
            print(f"Updated {updated} log files for {args.model_year}")
        else:
            print(f"No evaluation results found for {args.model_year}")
            sys.exit(1)
    else:
        # Process all model/year combinations
        for model_year, task_scores in results_mapping.items():
            print(f"\nProcessing {model_year}...")
            updated = process_model_logs(data_dir, model_year, task_scores)
            total_updated += updated
            print(f"  Updated {updated} log files")
    
    print(f"\nTotal updated: {total_updated} log files")
    print("Score update completed!")

if __name__ == "__main__":
    main()