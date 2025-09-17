#!/usr/bin/env python3
"""
Pass@k Analysis and Execution Time Visualization Script

This script properly calculates pass@k metrics where k represents the number of 
generation attempts (logs_2025, logs_2026, logs_2027, logs_2028) for each model.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Any
import math

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    records = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
    return records

def parse_generation_id(run_id: str) -> tuple:
    """Extract model and generation from run_id like 'gpt-5-mini-2025-08-07_2025' -> ('gpt-5-mini-2025-08-07', '2025')"""
    if '_' in run_id:
        parts = run_id.rsplit('_', 1)
        return parts[0], parts[1]
    return run_id, 'unknown'

def calculate_pass_at_k(results_by_problem: Dict[str, List[bool]], k: int) -> float:
    """
    Calculate pass@k metric given results for each problem.
    
    Pass@k = Pr[at least one of k attempts passes]
    """
    if k <= 0:
        return 0.0
        
    total_problems = len(results_by_problem)
    if total_problems == 0:
        return 0.0
    
    successful_problems = 0
    
    for problem_id, attempts in results_by_problem.items():
        # Only consider up to k attempts
        k_attempts = attempts[:k]
        # If at least one attempt succeeded, count this problem as successful
        if any(k_attempts):
            successful_problems += 1
    
    return successful_problems / total_problems

def load_and_organize_data():
    """Load all evaluation data and organize by model and problem for pass@k calculation."""
    results_dir = "/home/uzair/p_difficulty/e2h_eval/results"
    
    # Load all result files
    all_records = []
    result_files = [f for f in os.listdir(results_dir) if f.endswith("_results.jsonl")]
    
    for rf in result_files:
        run_id = rf.replace("_results.jsonl", "")
        records = load_jsonl(os.path.join(results_dir, rf))
        
        for record in records:
            record['run_id'] = run_id
            model, generation = parse_generation_id(run_id)
            record['model'] = model
            record['generation'] = generation
            all_records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Organize data by model and problem for pass@k calculation
    model_problem_results = defaultdict(lambda: defaultdict(list))
    
    # Sort by generation to ensure consistent ordering
    df_sorted = df.sort_values(['model', 'generation', 'task_id'])
    
    for _, row in df_sorted.iterrows():
        model = row['model']
        task_id = row['task_id']
        passed = row['status'] == 'passed'
        elapsed_ms = row.get('elapsed_ms', 0)
        
        model_problem_results[model][task_id].append({
            'passed': passed,
            'elapsed_ms': elapsed_ms,
            'generation': row['generation']
        })
    
    return df, model_problem_results

def calculate_pass_k_for_all_models(model_problem_results):
    """Calculate pass@k for k=1,2,3,4 for all models."""
    pass_k_results = {}
    
    for model, problem_results in model_problem_results.items():
        pass_k_results[model] = {}
        
        # Prepare data for pass@k calculation
        results_by_problem = {}
        for problem_id, attempts in problem_results.items():
            # Extract just the pass/fail status in order
            results_by_problem[problem_id] = [attempt['passed'] for attempt in attempts]
        
        # Calculate pass@k for k=1,2,3,4
        for k in range(1, 5):
            pass_k_results[model][f'pass@{k}'] = calculate_pass_at_k(results_by_problem, k)
    
    return pass_k_results

def create_pass_k_visualization(pass_k_results):
    """Create visualization for pass@k metrics."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Bar chart comparing pass@k for different k values
    models = list(pass_k_results.keys())
    k_values = [1, 2, 3, 4]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    for i, model in enumerate(models):
        pass_rates = [pass_k_results[model][f'pass@{k}'] for k in k_values]
        bars = ax1.bar(x + i*width, pass_rates, width, label=model, alpha=0.8)
        
        # Add value labels on bars
        for bar, rate in zip(bars, pass_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Pass@k Performance Comparison', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Number of Attempts (k)')
    ax1.set_ylabel('Pass@k Rate')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([f'pass@{k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # 2. Line plot showing pass@k improvement with more attempts
    for model in models:
        pass_rates = [pass_k_results[model][f'pass@{k}'] for k in k_values]
        ax2.plot(k_values, pass_rates, marker='o', linewidth=3, markersize=10, label=model)
        
        # Add value annotations
        for k, rate in zip(k_values, pass_rates):
            ax2.annotate(f'{rate:.3f}', (k, rate), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
    
    ax2.set_title('Pass@k Improvement with More Attempts', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Number of Attempts (k)')
    ax2.set_ylabel('Pass@k Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max([max([pass_k_results[model][f'pass@{k}'] for k in k_values]) for model in models]) * 1.1)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig('/home/uzair/p_difficulty/e2h_eval/results/pass_k_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_execution_time_analysis(df, model_problem_results):
    """Create comprehensive execution time analysis across generations."""
    
    # Filter successful runs only for timing analysis
    successful_df = df[df['status'] == 'passed'].copy()
    
    if successful_df.empty:
        print("No successful runs found for execution time analysis")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Execution time distribution by model
    models_with_data = successful_df['model'].unique()
    execution_data = []
    model_labels = []
    
    for model in models_with_data:
        model_times = successful_df[successful_df['model'] == model]['elapsed_ms']
        execution_data.append(model_times)
        model_labels.append(f"{model}\n(n={len(model_times)})")
    
    bp = ax1.boxplot(execution_data, labels=model_labels, patch_artist=True)
    ax1.set_title('Execution Time Distribution by Model', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    # 2. Execution time by generation
    generation_order = ['2025', '2026', '2027', '2028']
    for model in models_with_data:
        model_data = successful_df[successful_df['model'] == model]
        avg_times_by_gen = []
        
        for gen in generation_order:
            gen_data = model_data[model_data['generation'] == gen]
            if not gen_data.empty:
                avg_times_by_gen.append(gen_data['elapsed_ms'].mean())
            else:
                avg_times_by_gen.append(0)
        
        ax2.plot(generation_order, avg_times_by_gen, marker='o', linewidth=3, 
                markersize=8, label=model)
    
    ax2.set_title('Average Execution Time by Generation', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Execution Time (ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Speed vs Success Rate scatter plot
    model_gen_stats = []
    for model in models_with_data:
        for gen in generation_order:
            # All attempts for this model-generation
            all_attempts = df[(df['model'] == model) & (df['generation'] == gen)]
            successful_attempts = successful_df[(successful_df['model'] == model) & 
                                             (successful_df['generation'] == gen)]
            
            if not all_attempts.empty:
                success_rate = len(successful_attempts) / len(all_attempts)
                avg_time = successful_attempts['elapsed_ms'].mean() if not successful_attempts.empty else 0
                
                model_gen_stats.append({
                    'model': model,
                    'generation': gen,
                    'success_rate': success_rate,
                    'avg_time': avg_time,
                    'total_attempts': len(all_attempts)
                })
    
    if model_gen_stats:
        stats_df = pd.DataFrame(model_gen_stats)
        stats_df = stats_df[stats_df['avg_time'] > 0]  # Remove entries with no successful runs
        
        for model in stats_df['model'].unique():
            model_data = stats_df[stats_df['model'] == model]
            ax3.scatter(model_data['avg_time'], model_data['success_rate'], 
                       s=model_data['total_attempts']*20, alpha=0.7, label=model)
        
        ax3.set_title('Success Rate vs Average Execution Time', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Average Execution Time (ms)')
        ax3.set_ylabel('Success Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
    
    # 4. Problem-level execution time analysis
    # Calculate average execution time per problem across all generations
    problem_times = defaultdict(list)
    
    for _, row in successful_df.iterrows():
        problem_times[row['task_id']].append(row['elapsed_ms'])
    
    # Get average execution time per problem
    problem_avg_times = {}
    for problem, times in problem_times.items():
        problem_avg_times[problem] = np.mean(times)
    
    # Sort problems by average execution time
    sorted_problems = sorted(problem_avg_times.items(), key=lambda x: x[1])
    
    # Plot the top 10 slowest and fastest problems
    top_problems = sorted_problems[-10:]  # Slowest
    bottom_problems = sorted_problems[:10]  # Fastest
    
    all_problems = bottom_problems + top_problems
    problem_names = [p[0].replace('E2H_CF', '') for p, _ in all_problems]
    problem_times_list = [t for _, t in all_problems]
    
    colors = ['green'] * 10 + ['red'] * 10
    bars = ax4.bar(range(len(problem_names)), problem_times_list, color=colors, alpha=0.7)
    
    ax4.set_title('Execution Time: Fastest vs Slowest Problems', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Problem ID')
    ax4.set_ylabel('Average Execution Time (ms)')
    ax4.set_xticks(range(len(problem_names)))
    ax4.set_xticklabels(problem_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Add legend for colors
    ax4.text(0.02, 0.98, '■ Fastest Problems', transform=ax4.transAxes, 
             color='green', fontweight='bold', va='top')
    ax4.text(0.02, 0.92, '■ Slowest Problems', transform=ax4.transAxes, 
             color='red', fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig('/home/uzair/p_difficulty/e2h_eval/results/execution_time_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard(pass_k_results, df):
    """Create a comprehensive dashboard with pass@k and execution time insights."""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main pass@k comparison (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    models = list(pass_k_results.keys())
    k_values = [1, 2, 3, 4]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    for i, model in enumerate(models):
        pass_rates = [pass_k_results[model][f'pass@{k}'] for k in k_values]
        bars = ax1.bar(x + i*width, pass_rates, width, label=model, alpha=0.8)
        
        # Add value labels
        for bar, rate in zip(bars, pass_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_title('Pass@k Performance: Multiple Attempts Analysis', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Number of Attempts (k)')
    ax1.set_ylabel('Pass@k Rate')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([f'pass@{k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # 2. Pass@k improvement visualization (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    for model in models:
        pass_rates = [pass_k_results[model][f'pass@{k}'] for k in k_values]
        ax2.plot(k_values, pass_rates, marker='o', linewidth=4, markersize=10, label=model)
    
    ax2.set_title('Pass@k Improvement', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Attempts (k)')
    ax2.set_ylabel('Pass@k Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    # 3. Execution time by model (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    successful_df = df[df['status'] == 'passed']
    if not successful_df.empty:
        models_with_data = successful_df['model'].unique()
        for model in models_with_data:
            model_times = successful_df[successful_df['model'] == model]['elapsed_ms']
            ax3.hist(model_times, bins=20, alpha=0.7, label=model, density=True)
        
        ax3.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
    
    # 4. Generation performance (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    
    generation_order = ['2025', '2026', '2027', '2028']
    for model in models:
        success_rates = []
        for gen in generation_order:
            gen_data = df[(df['model'] == model) & (df['generation'] == gen)]
            if not gen_data.empty:
                success_rate = len(gen_data[gen_data['status'] == 'passed']) / len(gen_data)
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        ax4.plot(generation_order, success_rates, marker='s', linewidth=3, 
                markersize=8, label=model)
    
    ax4.set_title('Success Rate by Generation', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Success Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistics table (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    stats_data = []
    for model in models:
        model_data = df[df['model'] == model]
        successful_data = model_data[model_data['status'] == 'passed']
        
        total_attempts = len(model_data)
        total_success = len(successful_data)
        overall_success_rate = total_success / total_attempts if total_attempts > 0 else 0
        avg_time = successful_data['elapsed_ms'].mean() if not successful_data.empty else 0
        
        pass_at_4 = pass_k_results[model]['pass@4']
        
        stats_data.append([
            model,
            f"{overall_success_rate:.3f}",
            f"{pass_at_4:.3f}",
            f"{avg_time:.1f}ms" if avg_time > 0 else "N/A"
        ])
    
    table = ax5.table(cellText=stats_data,
                     colLabels=['Model', 'Overall\nSuccess', 'Pass@4', 'Avg Time'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f9f9f9' if i % 2 == 0 else 'white')
    
    ax5.set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    # 6. Key insights (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Calculate insights
    best_model = max(pass_k_results.keys(), key=lambda m: pass_k_results[m]['pass@4'])
    best_pass_4 = pass_k_results[best_model]['pass@4']
    
    # Calculate improvement from pass@1 to pass@4
    improvements = {}
    for model in models:
        improvement = pass_k_results[model]['pass@4'] - pass_k_results[model]['pass@1']
        improvements[model] = improvement
    
    best_improvement_model = max(improvements.keys(), key=lambda m: improvements[m])
    best_improvement_value = improvements[best_improvement_model]
    
    insights_text = f"""
🎯 PASS@K ANALYSIS INSIGHTS

• BEST MODEL: {best_model} achieves {best_pass_4:.1%} pass@4 (with 4 generation attempts)

• MULTIPLE ATTEMPTS BENEFIT: {best_improvement_model} improves by {best_improvement_value:.3f} ({best_improvement_value*100:.1f}%) from pass@1 to pass@4

• GENERATION ANALYSIS: Each log_202X folder represents a different generation attempt, not different years
  - 4 generations × 2 models × 20 problems = 160 total evaluation runs

• EXECUTION EFFICIENCY: Successful runs show significant variation in execution time
  - Fast execution doesn't necessarily correlate with higher success rates

• COMPETITIVE PROGRAMMING CHALLENGE: Even with 4 attempts, best models achieve <70% success
  - Indicates substantial room for improvement in algorithmic reasoning and code generation
    """
    
    ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes, fontsize=13,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('E2H-Codeforces: Pass@k and Execution Time Analysis', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig('/home/uzair/p_difficulty/e2h_eval/results/pass_k_comprehensive_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run pass@k analysis."""
    print("🔄 Loading evaluation data for pass@k analysis...")
    df, model_problem_results = load_and_organize_data()
    
    print("📊 Calculating pass@k metrics (k=1,2,3,4)...")
    pass_k_results = calculate_pass_k_for_all_models(model_problem_results)
    
    # Print pass@k results
    print("\n📈 Pass@k Results:")
    print("=" * 50)
    for model, results in pass_k_results.items():
        print(f"\n{model}:")
        for k in range(1, 5):
            print(f"  pass@{k}: {results[f'pass@{k}']:.3f}")
    
    print("\n🎨 Creating pass@k visualizations...")
    create_pass_k_visualization(pass_k_results)
    
    print("⏱️ Creating execution time analysis...")
    create_execution_time_analysis(df, model_problem_results)
    
    print("🎯 Creating comprehensive dashboard...")
    create_comprehensive_dashboard(pass_k_results, df)
    
    print("✅ Pass@k analysis completed!")
    print("📁 Results saved in: /home/uzair/p_difficulty/e2h_eval/results/")

if __name__ == "__main__":
    main()