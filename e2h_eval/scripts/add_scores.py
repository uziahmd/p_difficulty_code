#!/usr/bin/env python3
"""
Update 'score' field in original log files based on evaluation results:
- score = -1: no output generated (compilation errors, runtime exceptions, or empty output)
- score = 0: output generated but incorrect (failed status with wrong output)
- score = 1: correct output (passed status)
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

def is_no_output_error(error_msg):
    """Check if error indicates no output was generated.
    
    Score -1: No output generated (compilation errors, runtime exceptions, or empty output)
    Score 0: Output generated but wrong
    Score 1: Correct output
    """
    if not error_msg:
        return False
    
    error_msg_lower = error_msg.lower()
    
    # Check for empty output: "got ''" or "got \"\""
    if "got ''" in error_msg or 'got ""' in error_msg:
        return True
    
    # Traditional compilation/syntax errors (no output generated)
    syntax_indicators = [
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
    
    # Runtime exceptions that prevent output generation
    runtime_exception_indicators = [
        "traceback",
        "typeerror",
        "indexerror", 
        "keyerror",
        "attributeerror",
        "valueerror",
        "zerodivisionerror",
        "recursionerror",
        "unboundlocalerror",
        "assertionerror",
        "stopiteration"
    ]
    
    return any(indicator in error_msg_lower for indicator in syntax_indicators + runtime_exception_indicators)

def load_results_mapping(results_dir):
    """Load all result files and create a mapping from task_id to evaluation results."""
    results_mapping = {}
    
    # Look for both regular and variant result files
    result_files = list(Path(results_dir).glob("*_results.jsonl"))
    variant_result_files = list(Path(results_dir).glob("*_variants_results.jsonl"))
    all_result_files = result_files + variant_result_files
    
    for result_file in all_result_files:
        # Extract model and year from filename
        filename = result_file.stem  # removes .jsonl
        if "_variants_results" in filename:
            model_year = filename.replace("_variants_results", "")
        else:
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
                score = 0  # Runtime error (could run but took too long, output may have been generated)
            elif status == "failed":
                if is_no_output_error(error):
                    score = -1  # No output generated (compilation errors, exceptions, or empty output)
                else:
                    score = 0   # Output generated but wrong
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
    """Create mapping from base task_id to problem number (1-60)."""
    mapping = {
        "E2H_CF1031A": 1, "E2H_CF404A": 2, "E2H_CF492B": 3, "E2H_CF173A": 4,
        "E2H_CF1141D": 5, "E2H_CF822C": 6, "E2H_CF1846E2": 7, "E2H_CF270E": 8,
        "E2H_CF808E": 9, "E2H_CF409I": 10, "E2H_CF151A": 11, "E2H_CF339B": 12,
        "E2H_CF88A": 13, "E2H_CF633B": 14, "E2H_CF1767D": 15, "E2H_CF498A": 16,
        "E2H_CF1092C": 17, "E2H_CF1146D": 18, "E2H_CF980E": 19, "E2H_CF1709F": 20,
        "E2H_CF1152A": 21, "E2H_CF152A": 22, "E2H_CF1191B": 23, "E2H_CF915B": 24,
        "E2H_CF646B": 25, "E2H_CF1355C": 26, "E2H_CF934C": 27, "E2H_CF336D": 28,
        "E2H_CF464C": 29, "E2H_CF31E": 30, "E2H_CF401A": 31, "E2H_CF1095B": 32,
        "E2H_CF169B": 33, "E2H_CF808B": 34, "E2H_CF353C": 35, "E2H_CF1452D": 36,
        "E2H_CF892D": 37, "E2H_CF1763E": 38, "E2H_CF1333E": 39, "E2H_CF1542E2": 40,
        "E2H_CF1162A": 41, "E2H_CF1769B1": 42, "E2H_CF960A": 43, "E2H_CF1006B": 44,
        "E2H_CF736A": 45, "E2H_CF447C": 46, "E2H_CF1152C": 47, "E2H_CF61C": 48,
        "E2H_CF60C": 49, "E2H_CF1762E": 50, "E2H_CF141A": 51, "E2H_CF379A": 52,
        "E2H_CF23A": 53, "E2H_CF820B": 54, "E2H_CF1009B": 55, "E2H_CF958E1": 56,
        "E2H_CF222D": 57, "E2H_CF557D": 58, "E2H_CF1866H": 59, "E2H_CF1129C": 60
    }
    return mapping

def parse_variant_task_id(task_id):
    """Parse variant task_id like 'E2H_CF1031A_low_easy' into components."""
    if '_' not in task_id:
        return None, None, None
    
    parts = task_id.split('_')
    if len(parts) < 3:  # Need at least E2H_CF1031A
        return task_id, None, None
    
    # For variant task_ids: E2H_CF{contest_id}{problem_index}_{difficulty}_{complexity}
    # Base is E2H_CF{contest_id}{problem_index}
    if len(parts) >= 4:
        base_task_id = '_'.join(parts[:2])  # E2H_CF1031A  
        difficulty = parts[2]  # low, medium, none
        complexity = '_'.join(parts[3:]) if len(parts) > 3 else None  # easy, very_easy, etc.
        return base_task_id, difficulty, complexity
    
    # If no variant info, return just the base
    return '_'.join(parts[:2]), None, None

def process_model_logs(data_dir, model_year, task_scores):
    """Process all log files for a specific model/year combination."""
    
    # Parse model_year to extract model and year
    if model_year.endswith(('_2025', '_2026', '_2027', '_2028')):
        year = model_year[-4:]
        model = model_year[:-5]
    else:
        print(f"Warning: Could not parse year from {model_year}")
        return 0
    
    # Find the logs directory for this model/year - updated to use eval_ prefix
    logs_dir = Path(data_dir) / f"eval_{year}" / f"{model}_E2H-Codeforces"
    
    if not logs_dir.exists():
        print(f"Warning: Logs directory does not exist: {logs_dir}")
        return 0
    
    # Create task_id to problem number mapping
    task_mapping = create_task_to_problem_mapping()
    
    updated_count = 0
    
    # Process each task
    for task_id, score in task_scores.items():
        # Parse variant task_id
        base_task_id, difficulty, complexity = parse_variant_task_id(task_id)
        
        # Get problem number from base task_id
        if base_task_id not in task_mapping:
            print(f"Warning: Unknown base task_id: {base_task_id} from {task_id}")
            continue
            
        problem_num = task_mapping[base_task_id]
        
        # Find the specific log file for this variant
        if difficulty and complexity:
            log_file_name = f"{problem_num}_{difficulty}_{complexity}.json"
            log_file = logs_dir / log_file_name
            
            if log_file.exists():
                if update_log_file(log_file, task_id, score):
                    updated_count += 1
            else:
                print(f"Warning: Variant log file not found: {log_file}")
        else:
            print(f"Warning: Could not parse variant from task_id: {task_id}")
            
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