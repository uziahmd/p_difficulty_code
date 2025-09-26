# E2H-Codeforces Evaluation Toolkit

A comprehensive evaluation framework for testing AI models on 60 competitive programming problems from Codeforces using multiple prompt variants and I/O-based testing.

## 🎯 What This Code Does

This toolkit evaluates AI models on competitive programming by:

- **Problem Testing**: Tests 5 AI models (GPT-5 Mini, Gemini 2.5 Flash, Qwen3-14B, DeepSeek variants) on 60 Codeforces problems
- **Variant Analysis**: Uses 18 different prompt variants per problem (3 difficulty levels × 6 complexity levels) to find optimal prompting strategies
- **Comprehensive Evaluation**: Runs 21,600 total evaluations (5 models × 4 years × 60 problems × 18 variants) with sandboxed code execution
- **Performance Analysis**: Generates detailed visualizations showing pass rates, failure modes, and optimal prompt variants per model

## 📊 Key Results

**Best performing models:** Gemini 2.5 Flash (60.1% pass rate) and GPT-5 Mini (59.4% pass rate) significantly outperform others. **Medium difficulty prompts** consistently work best across all models, while "none" (no difficulty specified) prompts perform worst. The evaluation achieved 100% score coverage across all 21,600 test cases.

## 🚀 Quick Start

```bash
cd e2h_eval
python3 scripts/run_full_variant_evaluation.py  # Run complete evaluation
python3 scripts/variant_analysis.py             # Generate performance heatmaps  
python3 scripts/improved_failure_analysis.py    # Analyze failure patterns
```

## 📁 Repository Structure

```
e2h_eval/
├── data/
│   ├── E2H-Codeforces.json      # 60 problems dataset
│   └── eval_202*/               # Evaluation logs (21,600 files)
├── problems/
│   └── e2h_problems.jsonl       # Problem definitions
├── samples/
│   └── *_variants.jsonl         # Extracted samples (20 files)
├── results/
│   ├── *_results.jsonl          # Evaluation results (20 files)
│   ├── variant_heatmap_*.png    # Performance heatmaps
│   └── failure_mode_*.png       # Failure analysis charts
└── scripts/
    ├── run_full_variant_evaluation.py  # Main evaluation pipeline
    ├── variant_analysis.py             # Performance analysis
    └── improved_failure_analysis.py    # Failure mode analysis
```
