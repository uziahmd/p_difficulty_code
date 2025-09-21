#!/usr/bin/env python3
"""
Improved failure mode analysis for E2H evaluation.
Shows most com    fig.suptitle('Dominant                if variant in analysis[model]:
                    data = analysis[model][variant]
                    failure_mode = data['most_common_failure_mode']
                    matrix[i, j] = mode_colors[failure_mode]
                    
                    # Create annotation with failure percentages (among all attempts)
                    if failure_mode == 'no_failures':
                        annotations[i, j] = f"NF\n{data['correct_pct']:.0f}%"
                    elif failure_mode == 'incorrect':
                        annotations[i, j] = f"I\n{data['incorrect_pct']:.0f}%"
                    elif failure_mode == 'compilation_error':
                        annotations[i, j] = f"C\n{data['compilation_error_pct']:.0f}%"by Model and Variant\n(Among Failed Attempts: 20 Problems × 4 Years = 80 Attempts)', fontsize=20, fontweight='bold')on failure mode for each mode                      elif mode == 'incorrect':
                        annotations[i, j] = f\"I\\n{data['incorrect_pct']:.0f}%\"                 elif mode == 'incorrect':
                        annotations[i, j] = f"I\n{data['incorrect_pct']:.0f}%"                 elif mode == 'incorrect':
                        annotations[i, j] = f"I\n{data['incorrect_pct']:.0f}%"variant combination across all problems and years.
Only shows "correct" when there are no failures.
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Import from variant_analysis
sys.path.append('.')
from variant_analysis import load_evaluation_results

def analyze_failure_modes_aggregated(data):
    """Analyze failure modes for each model-variant combination, aggregated across all problems and years."""
    analysis = {}
    
    for model, model_data in data.items():
        analysis[model] = {}
        
        for variant, results in model_data.items():
            # Count all failure modes across all problems and years
            failure_counts = {
                'correct': 0,
                'incorrect': 0, 
                'compilation_error': 0
            }
            
            for result in results:
                score = result['score']
                if score == 1:
                    failure_counts['correct'] += 1
                elif score == 0:
                    failure_counts['incorrect'] += 1
                elif score == -1:
                    failure_counts['compilation_error'] += 1
            
            total = sum(failure_counts.values())
            
            if total > 0:
                # Calculate percentages
                percentages = {k: v/total * 100 for k, v in failure_counts.items()}
                
                # Focus on failure mode analysis - what type of failure is most common among failures
                total_failures = failure_counts['incorrect'] + failure_counts['compilation_error']
                
                if total_failures == 0:
                    # No failures - all correct (but we won't show this as it's not a failure mode)
                    most_common_failure_mode = 'no_failures'
                elif failure_counts['compilation_error'] >= failure_counts['incorrect']:
                    # Compilation errors are the dominant failure mode
                    most_common_failure_mode = 'compilation_error'
                else:
                    # Incorrect outputs are the dominant failure mode
                    most_common_failure_mode = 'incorrect'
                
                analysis[model][variant] = {
                    'total_attempts': total,
                    'correct_count': failure_counts['correct'],
                    'incorrect_count': failure_counts['incorrect'],
                    'compilation_error_count': failure_counts['compilation_error'],
                    'correct_pct': percentages['correct'],
                    'incorrect_pct': percentages['incorrect'],
                    'compilation_error_pct': percentages['compilation_error'],
                    'most_common_failure_mode': most_common_failure_mode,
                    'total_failures': total_failures,
                    'failure_rate': total_failures / total * 100 if total > 0 else 0
                }
    
    return analysis

def create_improved_failure_heatmap(analysis, output_dir='.'):
    """Create improved heatmap showing most common failure mode for each model-variant combination."""
    
    # Parse variants into difficulty and complexity for better organization
    difficulties = ['low', 'medium', 'none']
    complexities = ['very_easy', 'easy', 'moderate', 'hard', 'very_hard', 'none']
    
    models = list(analysis.keys())
    
    # Create one subplot per model
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('Most Common Outcome by Model and Variant\\n(Across All 20 Problems × 4 Years = 80 Attempts)', fontsize=20, fontweight='bold')
    
    # Color mapping for failure modes only
    mode_colors = {
        'incorrect': 0,         # Orange for incorrect output failures
        'compilation_error': 1, # Red for compilation error failures
        'no_failures': 2        # Light gray for variants with no failures (rare)
    }
    
    # Create custom colormap - focus on failure types
    colors = ['orange', 'red', 'lightgray']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    for idx, model in enumerate(models[:6]):  # Show up to 6 models
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create matrix for this model
        matrix = np.full((len(difficulties), len(complexities)), -1, dtype=float)
        annotations = np.full((len(difficulties), len(complexities)), '', dtype=object)
        
        for i, difficulty in enumerate(difficulties):
            for j, complexity in enumerate(complexities):
                variant = f"{difficulty}_{complexity}"
                
                if variant in analysis[model]:
                    data = analysis[model][variant]
                    failure_mode = data['most_common_failure_mode']
                    matrix[i, j] = mode_colors[failure_mode]
                    
                    # Create annotation with failure percentages (among all attempts)
                    if failure_mode == 'no_failures':
                        annotations[i, j] = f"NF\\n{data['correct_pct']:.0f}%"
                    elif failure_mode == 'incorrect':
                        annotations[i, j] = f"I\\n{data['incorrect_pct']:.0f}%"
                    elif failure_mode == 'compilation_error':
                        annotations[i, j] = f"C\\n{data['compilation_error_pct']:.0f}%"
        
        # Create heatmap for this model
        mask = matrix == -1  # Mask missing data
        
        sns.heatmap(matrix, 
                   mask=mask,
                   annot=annotations,
                   fmt='',
                   xticklabels=complexities,
                   yticklabels=difficulties,
                   cmap=cmap,
                   vmin=0, vmax=2,
                   cbar=False,
                   ax=ax,
                   square=True,
                   linewidths=0.5)
        
        ax.set_title(f'{model}\\n({len([v for v in analysis[model].values()])} variants)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Complexity →', fontsize=10)
        ax.set_ylabel('Difficulty →', fontsize=10)
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Hide unused subplots
    for idx in range(len(models), 6):
        row = idx // 3
        col = idx % 3
        axes[row, col].set_visible(False)
    
    # Add a custom legend focused on failure modes
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='orange', label='I Incorrect Output (Dominant Failure)'), 
        plt.Rectangle((0,0),1,1, facecolor='red', label='C Compilation Error (Dominant Failure)'),
        plt.Rectangle((0,0),1,1, facecolor='lightgray', label='NF No Failures (All Correct)')
    ]
    
    fig.legend(handles=legend_elements, 
              loc='center', 
              bbox_to_anchor=(0.85, 0.15),
              fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    
    plt.savefig(f"{output_dir}/improved_failure_mode_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved improved failure mode heatmap: {output_dir}/improved_failure_mode_heatmap.png")

def create_summary_statistics(analysis):
    """Create summary statistics table."""
    summary_data = []
    
    for model, model_data in analysis.items():
        for variant, data in model_data.items():
            # Parse variant
            if '_' in variant:
                parts = variant.split('_', 1)
                difficulty = parts[0]
                complexity = parts[1]
            else:
                difficulty = variant
                complexity = 'none'
            
            summary_data.append({
                'Model': model,
                'Difficulty': difficulty,
                'Complexity': complexity,
                'Variant': variant,
                'Total_Attempts': data['total_attempts'],
                'Correct_Count': data['correct_count'],
                'Incorrect_Count': data['incorrect_count'],
                'Compilation_Error_Count': data['compilation_error_count'],
                'Correct_Pct': f"{data['correct_pct']:.1f}%",
                'Incorrect_Pct': f"{data['incorrect_pct']:.1f}%",
                'Compilation_Error_Pct': f"{data['compilation_error_pct']:.1f}%",
                'Dominant_Failure_Mode': data['most_common_failure_mode'],
                'Failure_Rate_Pct': f"{data['failure_rate']:.1f}%"
            })
    
    return pd.DataFrame(summary_data)

def analyze_patterns_by_model(analysis):
    """Analyze patterns by model across all variants."""
    print("\\n🔍 MODEL FAILURE MODE PATTERNS:")
    print("=" * 80)
    
    for model, model_data in analysis.items():
        # Count failure modes across all variants for this model
        failure_mode_counts = defaultdict(int)
        total_variants = len(model_data)
        
        for variant_data in model_data.values():
            failure_mode_counts[variant_data['most_common_failure_mode']] += 1
        
        print(f"\\n📊 {model}:")
        print(f"   Total Variants: {total_variants}")
        
        for mode in ['incorrect', 'compilation_error', 'no_failures']:
            count = failure_mode_counts[mode]
            pct = count / total_variants * 100 if total_variants > 0 else 0
            
            if mode == 'incorrect':
                emoji = '�'
                desc = 'Incorrect Output (Dominant Failure)'
            elif mode == 'compilation_error':
                emoji = '�'
                desc = 'Compilation Error (Dominant Failure)'
            else:
                emoji = '⚪'
                desc = 'No Failures (All Correct)'
            
            print(f"   {emoji} {desc:35}: {count:2}/{total_variants} variants ({pct:5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Improved failure mode analysis for E2H evaluation")
    parser.add_argument("--results-dir", default="results", help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="results", help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return 1
    
    output_dir.mkdir(exist_ok=True)
    
    print("🔄 Loading evaluation results...")
    data = load_evaluation_results(results_dir)
    
    if not data:
        print("No evaluation results found!")
        return 1
    
    print(f"📊 Found data for {len(data)} models")
    
    print("🔍 Analyzing failure modes (prioritizing failures over success)...")
    analysis = analyze_failure_modes_aggregated(data)
    
    # Create summary table
    print("📋 Creating summary statistics...")
    summary_df = create_summary_statistics(analysis)
    
    # Save summary
    summary_df.to_csv(output_dir / "improved_failure_analysis.csv", index=False)
    print(f"✓ Saved improved failure analysis: {output_dir}/improved_failure_analysis.csv")
    
    # Save JSON for programmatic access
    with open(output_dir / "improved_failure_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved improved failure analysis: {output_dir}/improved_failure_analysis.json")
    
    # Create improved heatmap
    print("🎨 Creating improved failure mode heatmap...")
    create_improved_failure_heatmap(analysis, output_dir)
    
    # Analyze patterns
    analyze_patterns_by_model(analysis)
    
    print(f"\\n✅ Improved failure mode analysis completed!")
    print(f"📁 Results saved in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())