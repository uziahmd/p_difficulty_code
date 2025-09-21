#!/usr/bin/env python3
"""
Regenerate all result files with proper score extraction from log files.
This script reads sample files and extracts scores from corresponding log files.
"""

import json
import os
from pathlib import Path

def get_problem_mapping():
    """Create mapping from task_id to problem number"""
    mapping = {}
    with open('problems/e2h_problems.jsonl') as f:
        for i, line in enumerate(f):
            problem = json.loads(line)
            task_id = problem['task_id']
            mapping[task_id] = i + 1
    return mapping

def process_sample_file(sample_file, result_file, year):
    """Process a single sample file and generate results"""
    print(f"Processing {sample_file} -> {result_file}")
    
    # Get model name from file path
    model_name = os.path.basename(sample_file).split('_')[0]
    
    # Get problem mapping
    problem_mapping = get_problem_mapping()
    
    with open(sample_file) as f:
        samples = [json.loads(line) for line in f]
    
    results = []
    scores_found = 0
    
    for sample in samples:
        task_id = sample['task_id']
        
        # Parse task_id: E2H_CF1031A_low_very_easy -> CF1031A, low, very_easy
        parts = task_id.split('_')
        if len(parts) >= 3:
            cf_id = parts[1]  # CF1031A
            difficulty = parts[2]  # low
            # Join all remaining parts for complexity (handles very_easy, very_hard)
            complexity = '_'.join(parts[3:]) if len(parts) > 3 else 'easy'
            
            # Get problem number
            base_task_id = f'E2H_{cf_id}'
            problem_num = problem_mapping.get(base_task_id)
            
            if problem_num:
                # Find log file
                log_file = f'data/eval_{year}/{model_name}_E2H-Codeforces/{problem_num}_{difficulty}_{complexity}.json'
                
                result = {'task_id': task_id}
                
                if os.path.exists(log_file):
                    try:
                        with open(log_file) as lf:
                            log_data = json.load(lf)
                        
                        # Extract score and status
                        score = log_data.get('score')
                        if score is not None:
                            result['score'] = score
                            scores_found += 1
                            
                            if score == 1:
                                result['status'] = 'passed'
                                result['error'] = None
                            elif score == 0:
                                result['status'] = 'failed' 
                                result['error'] = 'Wrong output'
                            else:  # score == -1
                                result['status'] = 'failed'
                                result['error'] = 'No output/compilation error'
                        else:
                            result['status'] = 'unknown'
                            result['score'] = None
                            result['error'] = 'No score in log file'
                    except Exception as e:
                        result['status'] = 'error'
                        result['score'] = None
                        result['error'] = f'Error reading log: {str(e)}'
                else:
                    result['status'] = 'unknown'
                    result['score'] = None
                    result['error'] = f'Log file not found: {log_file}'
            else:
                result = {
                    'task_id': task_id,
                    'status': 'error',
                    'score': None,
                    'error': f'Problem mapping not found for {base_task_id}'
                }
        else:
            result = {
                'task_id': task_id,
                'status': 'error', 
                'score': None,
                'error': f'Invalid task_id format: {task_id}'
            }
        
        results.append(result)
    
    print(f"Found scores for {scores_found}/{len(results)} results")
    
    # Save results
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Saved to {result_file}")
    return scores_found, len(results)

def main():
    """Regenerate all result files"""
    print("Regenerating all result files with proper score extraction...")
    
    # Models and years
    models = ['qwen3-14b', 'qwen3-8b', 'llama-8b', 'deepseek-r1-0528-qwen3-8b', 'deepseek-r1-llama-8b', 'gemini-2.5-flash', 'gpt-5-mini-2025-08-07']
    years = ['2025', '2026', '2027', '2028']
    
    total_scores = 0
    total_samples = 0
    
    for model in models:
        for year in years:
            # Process variants file
            sample_file = f'samples/{model}_{year}_variants.jsonl'
            result_file = f'results/{model}_{year}_variants_results.jsonl'
            
            if os.path.exists(sample_file):
                scores_found, sample_count = process_sample_file(sample_file, result_file, year)
                total_scores += scores_found
                total_samples += sample_count
            else:
                print(f"Sample file not found: {sample_file}")
    
    print(f"\nSummary:")
    print(f"Total samples processed: {total_samples}")
    print(f"Total scores found: {total_scores}")
    print(f"Score coverage: {total_scores/total_samples*100:.1f}%")

if __name__ == "__main__":
    main()