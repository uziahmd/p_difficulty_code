#!/usr/bin/env python3
"""
Comprehensive variant analysis for E2H evaluation.
Analyzes performance of each difficulty/complexity combination per model.
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from collections import defaultdict, Counter
import itertools

# ---------------------------- Visual Style Helper (visuals only) ----------------------------
def _setup_plot_style():
    """Global, visual-only styling. Does not alter any logic or computations."""
    sns.set_theme(
        context="talk",      # slightly larger readable defaults
        style="whitegrid",   # subtle grid
        font_scale=0.9,      # compact labels
    )
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#fbfbfc",
        "axes.edgecolor":   "#d9d9e3",
        "axes.titleweight": "bold",
        "axes.titlesize":   16,
        "axes.labelsize":   12,
        "axes.grid":        True,
        "grid.color":       "#eaeaf2",
        "grid.linewidth":   0.8,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.frameon":   False,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })

# --------------------------------------------------------------------------------------------

def load_jsonl(path):
    """Load JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def parse_variant_task_id(task_id):
    """Parse variant task_id like 'E2H_CF1031A_low_easy' into components."""
    if '_' not in task_id:
        return None, None, None
    
    parts = task_id.split('_')
    if len(parts) < 3:  # Need at least E2H_CF1031A
        return task_id, None, None
    
    # For variant task_ids: E2H_CF{contest_id}{problem_index}_{difficulty}_{complexity}
    if len(parts) >= 4:
        base_task_id = '_'.join(parts[:2])  # E2H_CF1031A  
        difficulty = parts[2]  # low, medium, none
        complexity = '_'.join(parts[3:]) if len(parts) > 3 else None  # easy, very_easy, etc.
        return base_task_id, difficulty, complexity
    
    # If no variant info, return just the base
    return '_'.join(parts[:2]), None, None

def load_evaluation_results(results_dir):
    """Load all evaluation results from log files and organize by model, variant, and year."""
    data = defaultdict(lambda: defaultdict(list))
    
    # Get data directory (assuming it's parallel to results)
    data_dir = Path(results_dir).parent / "data"
    
    # Load from the actual log files which have scores
    for year in ['2025', '2026', '2027', '2028']:
        eval_dir = data_dir / f"eval_{year}"
        if not eval_dir.exists():
            continue
            
        for model_dir in eval_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name.replace("_E2H-Codeforces", "")
            
            for log_file in model_dir.glob("*.json"):
                # Parse filename to get variant info
                filename = log_file.stem  # e.g., "10_none_moderate"
                parts = filename.split('_')
                if len(parts) < 3:
                    continue
                    
                problem_id = parts[0]
                difficulty = parts[1] 
                complexity = '_'.join(parts[2:])
                variant = f"{difficulty}_{complexity}"
                
                # Load the log file
                try:
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                    
                    # Get score from log file
                    score = log_data.get('score', 0)
                    
                    data[model_name][variant].append({
                        'year': year,
                        'problem': problem_id,
                        'score': score,
                        'log_file': str(log_file)
                    })
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not process {log_file}: {e}")
                    continue
    
    return data

def is_compilation_error(error_msg):
    """Check if error message indicates compilation/syntax error."""
    if not error_msg:
        return False
    
    error_msg = error_msg.lower()
    
    # Traditional compilation/syntax errors
    syntax_indicators = [
        "syntaxerror", "indentationerror", "tabserror", "nameerror",
        "importerror", "modulenotfounderror", "invalid syntax",
        "unexpected indent", "unindent does not match",
        "inconsistent use of tabs and spaces"
    ]
    
    # Runtime exceptions that indicate programming errors (should be treated like compilation errors)
    programming_error_indicators = [
        "typeerror", "indexerror", "keyerror", "attributeerror",
        "valueerror", "zerodivisionerror", "recursionerror",
        "unboundlocalerror", "assertionerror", "stopiteration",
        "traceback (most recent call last)"
    ]
    
    return any(indicator in error_msg for indicator in syntax_indicators + programming_error_indicators)

def calculate_metrics(results):
    """Calculate success rate and pass@k metrics for results."""
    if not results:
        return {"success_rate": 0.0, "pass_at_1": 0.0, "pass_at_4": 0.0, "total_attempts": 0, "total_problems": 0}
    
    scores = [r['score'] for r in results]
    total = len(scores)
    successes = sum(1 for s in scores if s == 1)
    
    success_rate = successes / total if total > 0 else 0.0
    
    # For pass@k, group by problem (each problem has 4 attempts across years 2025-2028)
    problem_groups = defaultdict(list)
    for result in results:
        key = result['problem']  # Group by problem ID
        problem_groups[key].append((result['year'], result['score']))
    
    # Calculate pass@1 and pass@4
    pass_1_successes = 0
    pass_4_successes = 0
    total_problems = len(problem_groups)
    
    for problem_attempts in problem_groups.values():
        # Sort by year to ensure consistent ordering (2025, 2026, 2027, 2028)
        problem_attempts.sort(key=lambda x: x[0])
        scores_only = [score for year, score in problem_attempts]
        
        # pass@1: first attempt (2025) succeeds
        if scores_only and scores_only[0] == 1:
            pass_1_successes += 1
        
        # pass@4: at least one of the 4 attempts (years) succeeds
        if any(score == 1 for score in scores_only):
            pass_4_successes += 1
    
    pass_at_1 = pass_1_successes / total_problems if total_problems > 0 else 0.0
    pass_at_4 = pass_4_successes / total_problems if total_problems > 0 else 0.0
    
    return {
        "success_rate": success_rate,
        "pass_at_1": pass_at_1,
        "pass_at_4": pass_at_4,
        "total_attempts": total,
        "total_problems": total_problems
    }

def analyze_failure_modes(results):
    """Analyze failure modes for a set of results."""
    if not results:
        return {"correct": 0, "incorrect": 0, "compilation_error": 0}
    
    scores = [r['score'] for r in results]
    return {
        "correct": sum(1 for s in scores if s == 1),
        "incorrect": sum(1 for s in scores if s == 0),
        "compilation_error": sum(1 for s in scores if s == -1)
    }

def create_variant_performance_table(data):
    """Create table showing performance of each variant per model."""
    models = list(data.keys())
    
    # Get all unique variants
    all_variants = set()
    for model_data in data.values():
        all_variants.update(model_data.keys())
    all_variants = sorted(all_variants)
    
    # Create performance tables
    success_rate_table = []
    pass_4_table = []
    
    for model in models:
        success_row = {'Model': model}
        pass4_row = {'Model': model}
        
        for variant in all_variants:
            if variant in data[model]:
                metrics = calculate_metrics(data[model][variant])
                success_row[variant] = f"{metrics['success_rate']:.3f}"
                pass4_row[variant] = f"{metrics['pass_at_4']:.3f}"
            else:
                success_row[variant] = "0.000"
                pass4_row[variant] = "0.000"
        
        success_rate_table.append(success_row)
        pass_4_table.append(pass4_row)
    
    return pd.DataFrame(success_rate_table), pd.DataFrame(pass_4_table)

def find_best_variants_per_model(data):
    """Find the best performing variant for each model."""
    best_variants = {}
    
    for model, model_data in data.items():
        best_variant = None
        best_pass4 = -1
        best_success_rate = -1
        
        for variant, results in model_data.items():
            metrics = calculate_metrics(results)
            pass4 = metrics['pass_at_4']
            success_rate = metrics['success_rate']
            
            # Primary sort by pass@4, secondary by success rate
            if (pass4 > best_pass4) or (pass4 == best_pass4 and success_rate > best_success_rate):
                best_pass4 = pass4
                best_success_rate = success_rate
                best_variant = variant
        
        best_variants[model] = {
            'variant': best_variant,
            'pass_at_4': best_pass4,
            'success_rate': best_success_rate
        }
    
    return best_variants

def create_failure_mode_analysis(data):
    """Analyze failure modes across all variants."""
    failure_analysis = {}
    
    # Get all unique variants
    all_variants = set()
    for model_data in data.values():
        all_variants.update(model_data.keys())
    
    for variant in sorted(all_variants):
        # Aggregate results across all models for this variant
        all_results = []
        for model_data in data.values():
            if variant in model_data:
                all_results.extend(model_data[variant])
        
        failure_modes = analyze_failure_modes(all_results)
        total = sum(failure_modes.values())
        
        if total > 0:
            failure_analysis[variant] = {
                'correct_pct': failure_modes['correct'] / total * 100,
                'incorrect_pct': failure_modes['incorrect'] / total * 100,
                'compilation_error_pct': failure_modes['compilation_error'] / total * 100,
                'total_attempts': total
            }
    
    return failure_analysis

# ------------------------------------- Visuals --------------------------------------

def plot_variant_heatmap(data, metric='pass_at_4', output_dir='.'):
    """Create heatmap showing variant performance across models."""
    _setup_plot_style()  # visual-only
    models = list(data.keys())
    
    # Get all unique variants and organize by difficulty/complexity
    all_variants = set()
    for model_data in data.values():
        all_variants.update(model_data.keys())
    
    # Organize variants into matrix format
    difficulties = ['low', 'medium', 'none']
    complexities = ['very_easy', 'easy', 'moderate', 'hard', 'very_hard', 'none']
    
    # Create matrix for each model
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(f'Variant Performance Heatmap ({metric})', fontsize=20, fontweight='bold')
    
    for idx, model in enumerate(models[:6]):  # Show first 6 models
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create matrix
        matrix = np.zeros((len(difficulties), len(complexities)))
        
        for i, diff in enumerate(difficulties):
            for j, comp in enumerate(complexities):
                variant = f"{diff}_{comp}"
                if variant in data[model]:
                    metrics = calculate_metrics(data[model][variant])
                    matrix[i, j] = metrics[metric]
        
        # Plot heatmap (visual changes only)
        hm = sns.heatmap(
            matrix,
            annot=True,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
            fmt='.3f',
            xticklabels=complexities,
            yticklabels=difficulties,
            ax=ax,
            cmap='YlGnBu',      # colorblind-friendly sequential
            vmin=0, vmax=1,
            cbar=True,
            square=True,
            linewidths=0.6,
            linecolor="#ffffff"
        )
        # Colorbar label
        cbar = hm.collections[0].colorbar
        cbar.set_label(metric.replace('_', ' '), rotation=90, labelpad=12)
        
        ax.set_title(model, fontsize=14, pad=8)
        ax.set_xlabel('MANIPULATION →', fontsize=12)
        ax.set_ylabel('EFFORT →', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#e4e6ef")
    
    # Hide unused subplots
    for idx in range(len(models), 6):
        row = idx // 3
        col = idx % 3
        axes[row, col].set_visible(False)
    
    # Save high-quality outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    png_path = output_dir / f"variant_heatmap_{metric}.png"
    svg_path = output_dir / f"variant_heatmap_{metric}.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved variant heatmap: {png_path}")
    print(f"✓ Saved vector heatmap: {svg_path}")

def plot_failure_modes(failure_analysis, output_dir='.'):
    """Plot failure mode distribution across variants."""
    _setup_plot_style()  # visual-only
    variants = list(failure_analysis.keys())
    correct_pcts = [failure_analysis[v]['correct_pct'] for v in variants]
    incorrect_pcts = [failure_analysis[v]['incorrect_pct'] for v in variants]
    compile_pcts = [failure_analysis[v]['compilation_error_pct'] for v in variants]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    
    x = np.arange(len(variants))
    width = 0.8
    
    # Colorblind-friendly, high contrast colors
    col_correct = "#4E937A"   # greenish
    col_incorrect = "#F0A202" # orange
    col_compile = "#D1495B"   # deep red
    
    p1 = ax.bar(x, correct_pcts, width, label='Correct (score=1)', color=col_correct, alpha=0.85, edgecolor="white")
    p2 = ax.bar(x, incorrect_pcts, width, bottom=correct_pcts, label='Incorrect (score=0)', color=col_incorrect, alpha=0.9, edgecolor="white")
    p3 = ax.bar(x, compile_pcts, width, bottom=np.array(correct_pcts) + np.array(incorrect_pcts), 
                label='Compilation Error (score=-1)', color=col_compile, alpha=0.9, edgecolor="white")
    
    ax.set_xlabel('Variant (EFFORT_MANIPULATION)', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Failure Mode Distribution by Variant (Across All Models)', fontsize=18, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=35, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.05), loc='lower center', frameon=False)
    ax.grid(True, alpha=0.35, axis='y')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#e4e6ef")
    
    # Optional: light value labels on top segments (visual only)
    # Show labels for the top of each stack (sum to ~100%)
    totals = np.array(correct_pcts) + np.array(incorrect_pcts) + np.array(compile_pcts)
    for xi, total in zip(x, totals):
        ax.text(xi, total + 1.0, f"{total:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Save high-quality outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    png_path = output_dir / "failure_modes_by_variant.png"
    svg_path = output_dir / "failure_modes_by_variant.svg"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved failure modes plot: {png_path}")
    print(f"✓ Saved vector failure modes plot: {svg_path}")

# ------------------------------------- Pipeline --------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze variant performance in E2H evaluation")
    parser.add_argument("--results-dir", default="results", help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="results", help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return 1
    
    output_dir.mkdir(exist_ok=True)
    
    _setup_plot_style()  # set once at program start (visuals only)
    
    print("🔄 Loading evaluation results...")
    data = load_evaluation_results(results_dir)
    
    if not data:
        print("No evaluation results found!")
        return 1
    
    print(f"📊 Found data for {len(data)} models")
    
    # Create variant performance tables
    print("📈 Creating variant performance tables...")
    success_table, pass4_table = create_variant_performance_table(data)
    
    # Save tables
    success_table.to_csv(output_dir / "variant_success_rates.csv", index=False)
    pass4_table.to_csv(output_dir / "variant_pass_at_4.csv", index=False)
    
    print("✓ Saved variant performance tables")
    
    # Find best variants per model
    print("🎯 Finding best variants per model...")
    best_variants = find_best_variants_per_model(data)
    
    # Save best variants
    with open(output_dir / "best_variants_per_model.json", 'w') as f:
        json.dump(best_variants, f, indent=2)
    
    # Print best variants summary
    print("\n🏆 Best Variants per Model (by pass@4):")
    print("=" * 60)
    for model, info in best_variants.items():
        if info['variant']:
            print(f"{model:30} | {info['variant']:15} | pass@4: {info['pass_at_4']:.3f} | success: {info['success_rate']:.3f}")
    
    # Failure mode analysis
    print("\n🔍 Analyzing failure modes...")
    failure_analysis = create_failure_mode_analysis(data)
    
    # Save failure analysis
    with open(output_dir / "failure_mode_analysis.json", 'w') as f:
        json.dump(failure_analysis, f, indent=2)
    
    # Print failure mode summary
    print("\n💥 Failure Mode Analysis (Across All Models):")
    print("=" * 80)
    print(f"{'Variant':15} | {'Correct %':>10} | {'Incorrect %':>10} | {'Compile %':>10} | {'Total':>8}")
    print("-" * 80)
    
    for variant, analysis in sorted(failure_analysis.items()):
        print(f"{variant:15} | {analysis['correct_pct']:>9.1f}% | {analysis['incorrect_pct']:>9.1f}% | "
              f"{analysis['compilation_error_pct']:>9.1f}% | {analysis['total_attempts']:>8}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    plot_variant_heatmap(data, 'pass_at_4', output_dir)
    plot_variant_heatmap(data, 'success_rate', output_dir)
    plot_failure_modes(failure_analysis, output_dir)
    
    print(f"\n✅ Variant analysis completed!")
    print(f"📁 Results saved in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
