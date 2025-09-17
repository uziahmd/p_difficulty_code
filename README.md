# E2H-Codeforces Evaluation Toolkit

A comprehensive evaluation framework for testing AI models on competitive programming problems from Codeforces, implemented using the HumanEval methodology with I/O-based testing and pass@k metrics.

## 🎯 Overview

This toolkit evaluates AI models on 20 competitive programming problems using **multiple generation attempts** to calculate proper **pass@k metrics**. The system uses sandboxed execution with timeout handling to ensure reliable and secure evaluation.

## 📁 Repository Structure

```
p_difficulty/
├── README.md                          # This documentation
├── INDEX.md                           # Problem index and metadata
├── index.csv                          # CSV version of problem index
└── e2h_eval/                          # Main evaluation toolkit
    ├── IMPLEMENTATION_STATUS.md       # Implementation status and plan
    ├── data/                          # Original dataset and generation logs
    │   ├── E2H-Codeforces.json       # Complete problem dataset
    │   └── logs_202X/                 # Solution logs by generation (4 generations)
    │       ├── gpt-5-mini-2025-08-07_E2H-Codeforces/  # GPT model generations
    │       └── qwen3-14b_E2H-Codeforces/               # Qwen model generations
    ├── engine/                        # Core evaluation engine
    │   ├── __init__.py               
    │   ├── harness.py                # Sandboxed test execution
    │   └── reliability_guard.py      # Security and resource limits
    ├── problems/                      # Problem definitions
    │   └── e2h_problems.jsonl        # HumanEval-style problem format
    ├── samples/                       # Extracted solution samples
    │   └── *.jsonl                   # Solutions by model and generation
    ├── scripts/                       # Evaluation and analysis scripts
    │   ├── build_problems_jsonl.py   # Convert problems to HumanEval format
    │   ├── extract_samples_from_logs.py # Extract solutions from generation logs
    │   ├── run_eval.py               # Single evaluation runner
    │   ├── run_full_evaluation.py    # Complete evaluation pipeline
    │   ├── pass_k_analysis.py        # Pass@k metrics and visualizations
    │   └── summarize.py              # Results summarization
    └── results/                       # Evaluation results and analysis
        ├── summary_runs.csv          # Aggregated results summary
        ├── *_results.jsonl           # Detailed results by model/generation
        ├── pass_k_analysis.png       # Pass@k comparison charts
        ├── execution_time_analysis.png # Timing analysis
        └── pass_k_comprehensive_dashboard.png # Complete dashboard
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas matplotlib seaborn numpy
```

### Run Complete Evaluation Pipeline
```bash
cd e2h_eval
python3 scripts/run_full_evaluation.py
```

### Generate Pass@k Analysis and Visualizations
```bash
python3 scripts/pass_k_analysis.py
```

### Generate Summary Statistics
```bash
python3 scripts/summarize.py
```

## 📊 Key Results

### Pass@k Performance Summary
- **GPT-5-mini-2025-08-07**:
  - pass@1: 50.0% (single attempt)
  - pass@2: 65.0% (best of 2 attempts)
  - pass@3: 65.0% (best of 3 attempts)
  - pass@4: 70.0% (best of 4 attempts)

- **Qwen3-14b**: 0.0% across all pass@k metrics (failed all attempts)

### Key Insights
- **Multiple attempts significantly help**: GPT-5-mini improves from 50% to 70% with 4 attempts
- **Diminishing returns observed**: Major improvement from pass@1 to pass@2, then plateau
- **Model capability gaps**: Vast difference between GPT and Qwen performance
- **Execution efficiency**: Fast execution doesn't correlate with higher success rates

## 🧪 Dataset and Methodology

### Dataset Details
- **20 unique problems** from Codeforces
- **2 models evaluated**: GPT-5-mini-2025-08-07, Qwen3-14b
- **4 generations per model**: logs_2025, logs_2026, logs_2027, logs_2028
- **160 total evaluations**: 20 problems × 2 models × 4 generations
- **Rating range**: 712-2451 (beginner to expert level)
- **Algorithm categories**: Math, Implementation, Graph Theory, etc.

### Evaluation Methodology
- **HumanEval-style I/O testing**: Solutions tested against expected input/output
- **Sandboxed execution**: Isolated environment with security restrictions
- **Timeout handling**: 30-second execution limit per test
- **Pass@k calculation**: Probability that at least one of k attempts succeeds
- **Performance timing**: Millisecond-precision execution time measurement

## 📈 Analysis Features

### Pass@k Metrics
- **Multiple attempt analysis**: Pass@1, pass@2, pass@3, pass@4 calculations
- **Success rate improvement**: Quantifies benefit of multiple generations
- **Model comparison**: Side-by-side performance across different attempt counts
- **Statistical significance**: Proper pass@k probability calculations

### Execution Time Analysis
- **Performance profiling**: Millisecond-precision timing for successful runs
- **Speed vs accuracy**: Correlation analysis between execution time and success
- **Generation comparison**: Timing trends across different model generations
- **Problem complexity**: Execution time patterns by problem difficulty

### Comprehensive Visualizations
- **Pass@k comparison charts**: Bar plots and line graphs showing improvement curves
- **Execution time distributions**: Box plots and histograms of timing data
- **Performance dashboards**: Multi-panel views combining metrics
- **Model capability heatmaps**: Visual representation of success patterns

## 🔧 Core Components

### Evaluation Engine (`engine/`)
- **`harness.py`**: Main evaluation orchestrator with multiprocessing isolation
- **`reliability_guard.py`**: Security sandbox with system call restrictions
- **Features**:
  - Temporary directory isolation
  - Resource usage limits (memory, CPU time)
  - Disabled dangerous system calls
  - Timeout handling with graceful termination
  - Error capture and categorization

### Data Processing Pipeline
- **Sample extraction**: Parse generation logs into standardized format
- **Problem conversion**: Transform Codeforces problems to HumanEval format
- **Result aggregation**: Combine results across generations for pass@k calculation
- **Metadata integration**: Include problem ratings, tags, and difficulty information

### Analysis Scripts (`scripts/`)
- **`pass_k_analysis.py`**: Core pass@k calculation and visualization engine
- **`run_eval.py`**: Single model evaluation with sandboxed execution
- **`run_full_evaluation.py`**: Complete pipeline for all models and generations
- **`summarize.py`**: Generate statistical summaries and CSV reports
- **`extract_samples_from_logs.py`**: Parse and standardize solution logs

## 📝 Usage Examples

### Evaluate Single Model on All Problems
```bash
python3 scripts/run_eval.py \
  --samples samples/gpt-5-mini-2025-08-07_2025.jsonl \
  --output results/custom_eval.jsonl
```

### Run Complete Pass@k Analysis
```bash
python3 scripts/pass_k_analysis.py
# Generates:
# - pass_k_analysis.png
# - execution_time_analysis.png  
# - pass_k_comprehensive_dashboard.png
```

### Extract New Samples from Logs
```bash
python3 scripts/extract_samples_from_logs.py
```

### Generate Summary Statistics
```python
from scripts.summarize import load_jsonl, calculate_summary_stats

# Load results
results = load_jsonl('results/gpt-5-mini-2025-08-07_2025_results.jsonl')

# Calculate statistics
stats = calculate_summary_stats(results)
print(f"Pass rate: {stats['pass_rate']:.3f}")
print(f"Average time: {stats['avg_ms']:.1f}ms")
```

## 🧠 Understanding Pass@k

Pass@k measures the probability that **at least one** of k attempts succeeds on a problem:

```
Pass@k = Pr[∃ i ∈ {1,...,k} : attempt_i succeeds]
```

### Why Pass@k Matters
- **Realistic AI assessment**: Models often need multiple attempts for complex problems
- **Generation diversity**: Different attempts may explore different solution approaches
- **Practical deployment**: Real-world usage often involves multiple query attempts
- **Capability measurement**: Distinguishes between "never solves" vs "sometimes solves"

### Our Results Interpretation
- **pass@1 = 50%**: GPT-5-mini solves half the problems on first attempt
- **pass@4 = 70%**: With 4 attempts, success rate increases to 70%
- **Improvement = 20%**: Multiple attempts provide significant benefit
- **Qwen3-14b = 0%**: No improvement with more attempts (fundamental failure)

## 🏆 Key Research Insights

### Model Capabilities
1. **GPT-5-mini shows strong competitive programming ability** with 70% pass@4
2. **Qwen3-14b fails systematically** across all problems and attempts
3. **Multiple attempts significantly improve performance** for capable models
4. **Code generation patterns differ fundamentally** between models

### Technical Findings
1. **Execution time varies widely** (3ms to 135ms) but doesn't correlate with success
2. **Pass@k improvement plateaus** between attempt 2 and 3, with final boost at attempt 4
3. **Sandboxed evaluation is critical** for reliable and secure code execution
4. **Problem difficulty impacts success rates** but some "easy" problems remain challenging

### Algorithmic Insights
1. **Mathematical reasoning translates well** to competitive programming success
2. **Implementation problems** are generally well-handled by successful models
3. **Complex algorithms and data structures** remain challenging even for top models
4. **Code structure patterns** (function definition vs execution) critically impact success

## 🤝 Contributing

This toolkit provides a comprehensive foundation for evaluating AI models on competitive programming tasks. The modular design supports:

### Extension Opportunities
- **Adding new models**: Simply place generation logs in `data/logs_202X/` format
- **Expanding problem sets**: Add new problems to `problems/e2h_problems.jsonl`
- **Custom metrics**: Extend `pass_k_analysis.py` with additional evaluation criteria
- **Visualization enhancement**: Add new charts and analysis dimensions

### Research Applications
- **Model comparison studies**: Systematic evaluation across multiple AI systems
- **Algorithm capability assessment**: Fine-grained analysis of specific programming skills
- **Training data impact**: Compare models trained on different code datasets
- **Pass@k methodology**: Apply to other code generation evaluation tasks

### Development Workflow
1. **Add new data**: Place solution logs in standardized format
2. **Extract samples**: Run `extract_samples_from_logs.py`
3. **Evaluate**: Execute `run_full_evaluation.py` 
4. **Analyze**: Generate insights with `pass_k_analysis.py`
5. **Visualize**: Create charts and dashboards

## 🔒 Security and Reliability

### Sandboxed Execution
- **Process isolation**: Each code execution runs in separate process
- **Filesystem isolation**: Temporary directories prevent file system access
- **Resource limits**: Memory and CPU time constraints prevent resource exhaustion
- **System call filtering**: Dangerous operations (network, file access) are blocked
- **Timeout protection**: Hard limits prevent infinite loops and hanging processes

### Error Handling
- **Comprehensive error capture**: Syntax errors, runtime exceptions, assertion failures
- **Execution state tracking**: Success, failure, timeout status for each test
- **Graceful degradation**: Partial results when some tests fail
- **Detailed logging**: Full error messages and stack traces for debugging

## 📊 Generated Output Files

### Core Results
- **`*_results.jsonl`**: Detailed per-problem results with timing and error information
- **`summary_runs.csv`**: Aggregated statistics across all models and generations
- **`*_samples.jsonl`**: Extracted and standardized solution code from generation logs

### Visualizations
- **`pass_k_analysis.png`**: Bar charts and line plots of pass@k metrics
- **`execution_time_analysis.png`**: Box plots and distributions of execution timing
- **`pass_k_comprehensive_dashboard.png`**: Multi-panel dashboard with key insights

### Analysis Data
- **CSV exports**: Problem-level and aggregated results for further analysis
- **Statistical summaries**: Pass rates, timing statistics, error categorization
- **Performance profiles**: Model capabilities across different problem types

## 📄 License and Attribution

This toolkit is designed for educational and research use. The competitive programming problems are derived from Codeforces and retain their original licensing. The evaluation methodology builds upon the HumanEval framework for code generation assessment.

### Citation
If you use this toolkit in your research, please cite the E2H-Codeforces dataset and methodology. The pass@k evaluation approach follows established practices in code generation evaluation literature.

### Acknowledgments
- **Codeforces**: Original competitive programming problems and test cases
- **HumanEval**: Evaluation methodology and I/O-based testing framework
- **OpenAI**: Pass@k metrics definition and calculation methodology