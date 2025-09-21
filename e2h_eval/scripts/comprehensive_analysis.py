#!/usr/bin/env python3
"""
Comprehensive analysis script for E2H 60-problem evaluation results.
Generates detailed statistics, visualizations, and insights.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def load_all_results():
    """Load all result files and organize by model and year"""
    results = {}
    
    for file in os.listdir('results'):
        if file.endswith('_variants_results.jsonl'):
            # Parse filename: model_year_variants_results.jsonl
            parts = file.replace('_variants_results.jsonl', '').split('_')
            if len(parts) >= 2:
                year = parts[-1]
                model = '_'.join(parts[:-1])
                
                with open(f'results/{file}') as f:
                    data = [json.loads(line) for line in f]
                
                if model not in results:
                    results[model] = {}
                results[model][year] = data
                
                print(f"Loaded {len(data)} results for {model} {year}")
    
    return results

def analyze_scores(results):
    """Analyze score distributions across models and years"""
    analysis = {}
    
    for model in results:
        analysis[model] = {}
        for year in results[model]:
            data = results[model][year]
            
            # Count scores
            score_counts = Counter()
            total_with_scores = 0
            
            for result in data:
                score = result.get('score')
                if score is not None:
                    score_counts[score] += 1
                    total_with_scores += 1
                else:
                    score_counts['None'] += 1
            
            # Calculate metrics
            passed = score_counts.get(1, 0)
            failed_wrong = score_counts.get(0, 0)
            failed_no_output = score_counts.get(-1, 0)
            missing_scores = score_counts.get('None', 0)
            
            analysis[model][year] = {
                'total': len(data),
                'with_scores': total_with_scores,
                'passed': passed,
                'failed_wrong': failed_wrong,
                'failed_no_output': failed_no_output,
                'missing_scores': missing_scores,
                'pass_rate': passed / total_with_scores if total_with_scores > 0 else 0,
                'score_coverage': total_with_scores / len(data) if len(data) > 0 else 0
            }
    
    return analysis

def create_summary_report(analysis):
    """Create comprehensive summary report"""
    print("\n" + "="*80)
    print("E2H 60-PROBLEM EVALUATION SUMMARY REPORT")
    print("="*80)
    
    # Overall statistics
    total_samples = 0
    total_with_scores = 0
    total_passed = 0
    
    for model in analysis:
        for year in analysis[model]:
            stats = analysis[model][year]
            total_samples += stats['total']
            total_with_scores += stats['with_scores']
            total_passed += stats['passed']
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Samples with scores: {total_with_scores:,}")
    print(f"  Score coverage: {total_with_scores/total_samples*100:.1f}%")
    print(f"  Overall pass rate: {total_passed/total_with_scores*100:.1f}%")
    
    # Model performance
    print(f"\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    model_stats = {}
    for model in analysis:
        model_total_samples = 0
        model_total_with_scores = 0
        model_total_passed = 0
        
        for year in analysis[model]:
            stats = analysis[model][year]
            model_total_samples += stats['total']
            model_total_with_scores += stats['with_scores']
            model_total_passed += stats['passed']
        
        model_pass_rate = model_total_passed / model_total_with_scores if model_total_with_scores > 0 else 0
        model_coverage = model_total_with_scores / model_total_samples if model_total_samples > 0 else 0
        
        model_stats[model] = {
            'pass_rate': model_pass_rate,
            'coverage': model_coverage,
            'samples': model_total_samples,
            'passed': model_total_passed
        }
        
        print(f"{model:<30} Pass Rate: {model_pass_rate*100:5.1f}% Coverage: {model_coverage*100:5.1f}%")
    
    # Year comparison
    print(f"\nYEAR COMPARISON:")
    print("-" * 40)
    
    year_stats = defaultdict(lambda: {'total': 0, 'with_scores': 0, 'passed': 0})
    for model in analysis:
        for year in analysis[model]:
            stats = analysis[model][year]
            year_stats[year]['total'] += stats['total']
            year_stats[year]['with_scores'] += stats['with_scores']
            year_stats[year]['passed'] += stats['passed']
    
    for year in sorted(year_stats.keys()):
        stats = year_stats[year]
        pass_rate = stats['passed'] / stats['with_scores'] if stats['with_scores'] > 0 else 0
        print(f"Year {year}: {pass_rate*100:5.1f}% pass rate ({stats['passed']}/{stats['with_scores']} passed)")
    
    return model_stats, year_stats

def create_visualizations(analysis):
    """Create comprehensive visualizations"""
    print("\nGenerating visualizations...")
    
    # Prepare data for visualization
    models = []
    years = []
    pass_rates = []
    coverages = []
    
    for model in analysis:
        for year in analysis[model]:
            stats = analysis[model][year]
            models.append(model)
            years.append(year)
            pass_rates.append(stats['pass_rate'] * 100)
            coverages.append(stats['score_coverage'] * 100)
    
    df = pd.DataFrame({
        'Model': models,
        'Year': years,
        'Pass_Rate': pass_rates,
        'Coverage': coverages
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Pass rate heatmap by model and year
    pivot_pass = df.pivot(index='Model', columns='Year', values='Pass_Rate')
    sns.heatmap(pivot_pass, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0,0], cbar_kws={'label': 'Pass Rate (%)'})
    axes[0,0].set_title('Pass Rate by Model and Year (%)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Model')
    
    # 2. Coverage heatmap by model and year
    pivot_cov = df.pivot(index='Model', columns='Year', values='Coverage')
    sns.heatmap(pivot_cov, annot=True, fmt='.1f', cmap='Blues', ax=axes[0,1], cbar_kws={'label': 'Coverage (%)'})
    axes[0,1].set_title('Score Coverage by Model and Year (%)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Model')
    
    # 3. Model comparison (average across years)
    model_avg = df.groupby('Model')[['Pass_Rate', 'Coverage']].mean().sort_values('Pass_Rate', ascending=True)
    
    x_pos = range(len(model_avg))
    axes[1,0].barh(x_pos, model_avg['Pass_Rate'], alpha=0.7, color='lightgreen')
    axes[1,0].set_yticks(x_pos)
    axes[1,0].set_yticklabels([m.replace('_', '\n') for m in model_avg.index], fontsize=10)
    axes[1,0].set_xlabel('Average Pass Rate (%)')
    axes[1,0].set_title('Model Performance Ranking', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, v in enumerate(model_avg['Pass_Rate']):
        axes[1,0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # 4. Year trend analysis
    year_avg = df.groupby('Year')[['Pass_Rate', 'Coverage']].mean()
    
    axes[1,1].plot(year_avg.index, year_avg['Pass_Rate'], marker='o', linewidth=3, markersize=8, label='Pass Rate', color='green')
    axes[1,1].plot(year_avg.index, year_avg['Coverage'], marker='s', linewidth=3, markersize=8, label='Coverage', color='blue')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Percentage (%)')
    axes[1,1].set_title('Performance Trends Across Years', fontsize=14, fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/e2h_60_problems_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive analysis chart: results/e2h_60_problems_analysis.png")

def analyze_task_variants(results):
    """Analyze performance across different task variants"""
    print("\nAnalyzing task variant performance...")
    
    variant_stats = defaultdict(lambda: {'total': 0, 'passed': 0})
    
    for model in results:
        for year in results[model]:
            for result in results[model][year]:
                task_id = result['task_id']
                score = result.get('score')
                
                if score is not None:
                    # Extract variant info: E2H_CF1031A_low_easy
                    parts = task_id.split('_')
                    if len(parts) >= 4:
                        difficulty = parts[2]  # low, medium, high
                        complexity = parts[3]  # easy, moderate, hard, very_easy, very_hard, none
                        variant = f"{difficulty}_{complexity}"
                        
                        variant_stats[variant]['total'] += 1
                        if score == 1:
                            variant_stats[variant]['passed'] += 1
    
    print("\nVARIANT PERFORMANCE:")
    print("-" * 50)
    
    # Sort by pass rate
    sorted_variants = sorted(variant_stats.items(), 
                           key=lambda x: x[1]['passed']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                           reverse=True)
    
    for variant, stats in sorted_variants:
        if stats['total'] > 0:
            pass_rate = stats['passed'] / stats['total'] * 100
            print(f"{variant:<20} {pass_rate:5.1f}% ({stats['passed']:3d}/{stats['total']:3d})")
    
    return dict(variant_stats)

def main():
    """Main analysis function"""
    print("Starting comprehensive E2H 60-problem analysis...")
    
    # Load all results
    results = load_all_results()
    
    # Analyze scores
    analysis = analyze_scores(results)
    
    # Create summary report
    model_stats, year_stats = create_summary_report(analysis)
    
    # Analyze variants
    variant_stats = analyze_task_variants(results)
    
    # Create visualizations
    create_visualizations(analysis)
    
    # Save detailed analysis to JSON
    detailed_analysis = {
        'summary': {
            'total_models': len(results),
            'total_years': len(set(year for model in results for year in results[model])),
            'model_performance': model_stats,
            'year_trends': dict(year_stats),
            'variant_performance': variant_stats
        },
        'detailed_stats': analysis
    }
    
    with open('results/detailed_analysis_60_problems.json', 'w') as f:
        json.dump(detailed_analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: results/detailed_analysis_60_problems.json")
    print("Analysis complete!")

if __name__ == "__main__":
    main()