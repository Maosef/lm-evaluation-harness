#!/usr/bin/env python3
"""
Six-model comparison analysis for BBH, IFEval, MMLU, TruthfulQA MC2, GPQA, GSM8K benchmarks.

Models:
1. deepseek-ai/DeepSeek-R1-Distill-Llama-8B (base)
2. huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated
3. huihui-ai/Huihui-Qwen3-14B-abliterated-v2
4. huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2
5. huihui-ai/Meta-Llama-3.1-8B-Instruct-abliterated
6. p-e-w/Llama-3.1-8B-Instruct-heretic
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import glob

# Define the results directory
RESULTS_DIR = Path('/home/ec2-user/git/lm-evaluation-harness/results/bbh_ifeval_mmlu_truthfulqa_mc2_gpqa_gsm8k_results')

# Define model configurations
MODELS = {
    'DeepSeek-R1-8B\n(Base)': 'deepseek-ai__DeepSeek-R1-Distill-Llama-8B',
    'DeepSeek-R1-8B\n(Abliterated)': 'huihui-ai__DeepSeek-R1-Distill-Llama-8B-abliterated',
    'Qwen3-14B\n(Abliterated)': 'huihui-ai__Huihui-Qwen3-14B-abliterated-v2',
    'GPT-OSS-20B\n(Abliterated)': 'huihui-ai__Huihui-gpt-oss-20b-mxfp4-abliterated-v2',
    'Llama-3.1-8B\n(Abliterated)': 'huihui-ai__Meta-Llama-3.1-8B-Instruct-abliterated',
    'Llama-3.1-8B\n(Heretic)': 'p-e-w__Llama-3.1-8B-Instruct-heretic',
}

# Load result files
def load_results():
    """Load all result JSON files."""
    results = {}
    for model_name, model_dir in MODELS.items():
        model_path = RESULTS_DIR / model_dir
        json_files = list(model_path.glob('results_*.json'))
        if json_files:
            # Take the most recent file
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file) as f:
                results[model_name] = json.load(f)
            print(f"✓ Loaded {model_name}: {latest_file.name}")
        else:
            print(f"✗ No results found for {model_name}")
    return results

# Extract metrics
def extract_metrics(results):
    """Extract key metrics from all models."""
    metrics = {}

    for model_name, data in results.items():
        r = data['results']
        metrics[model_name] = {
            # Main benchmarks
            'bbh': r.get('bbh', {}).get('exact_match,get-answer', None),
            'gsm8k': r.get('gsm8k', {}).get('exact_match,flexible-extract', None),
            'mmlu_overall': r.get('mmlu', {}).get('acc,none', None),
            'truthfulqa_mc2': r.get('truthfulqa_mc2', {}).get('acc,none', None),

            # MMLU categories
            'mmlu_humanities': r.get('mmlu_humanities', {}).get('acc,none', None),
            'mmlu_stem': r.get('mmlu_stem', {}).get('acc,none', None),
            'mmlu_social_sciences': r.get('mmlu_social_sciences', {}).get('acc,none', None),
            'mmlu_other': r.get('mmlu_other', {}).get('acc,none', None),

            # IFEval
            'ifeval_strict': r.get('ifeval', {}).get('prompt_level_strict_acc,none', None),
            'ifeval_loose': r.get('ifeval', {}).get('prompt_level_loose_acc,none', None),

            # GPQA
            'gpqa_diamond': r.get('gpqa_diamond_n_shot', {}).get('acc,none', None),
            'gpqa_extended': r.get('gpqa_extended_n_shot', {}).get('acc,none', None),
            'gpqa_main': r.get('gpqa_main_n_shot', {}).get('acc,none', None),
        }

    return metrics

def print_comparison_table(metrics):
    """Print detailed comparison table."""
    print("\n" + "=" * 150)
    print("SIX-MODEL COMPARISON: BBH, IFEval, MMLU, TruthfulQA, GPQA, GSM8K")
    print("=" * 150)

    # Header
    model_names = list(metrics.keys())
    header = f"{'Benchmark':<25}"
    for name in model_names:
        short_name = name.replace('\n', ' ')[:20]
        header += f" {short_name:>20}"
    print(header)
    print("-" * 150)

    # Main benchmarks
    main_metrics = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2', 'ifeval_loose']
    for key in main_metrics:
        row = f"{key:<25}"
        for model_name in model_names:
            val = metrics[model_name].get(key)
            if val is not None:
                row += f" {val*100:19.2f}%"
            else:
                row += f" {'N/A':>20}"
        print(row)

    print("\n" + "-" * 150)
    print("MMLU CATEGORIES")
    print("-" * 150)

    mmlu_cats = ['mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other']
    for key in mmlu_cats:
        row = f"{key:<25}"
        for model_name in model_names:
            val = metrics[model_name].get(key)
            if val is not None:
                row += f" {val*100:19.2f}%"
            else:
                row += f" {'N/A':>20}"
        print(row)

    print("\n" + "-" * 150)
    print("GPQA VARIANTS")
    print("-" * 150)

    gpqa_variants = ['gpqa_diamond', 'gpqa_extended', 'gpqa_main']
    for key in gpqa_variants:
        row = f"{key:<25}"
        for model_name in model_names:
            val = metrics[model_name].get(key)
            if val is not None:
                row += f" {val*100:19.2f}%"
            else:
                row += f" {'N/A':>20}"
        print(row)

    print("=" * 150)

def print_summary_statistics(metrics):
    """Print summary statistics."""
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    for model_name, model_data in metrics.items():
        # Calculate averages (excluding None values)
        basic_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2']
        basic_values = [model_data[k] for k in basic_keys if model_data.get(k) is not None]

        full_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2',
                     'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main']
        full_values = [model_data[k] for k in full_keys if model_data.get(k) is not None]

        if basic_values:
            avg_basic = np.mean(basic_values) * 100
        else:
            avg_basic = None

        if full_values:
            avg_full = np.mean(full_values) * 100
        else:
            avg_full = None

        print(f"\n{model_name.replace(chr(10), ' ')}:")
        if avg_basic is not None:
            print(f"  Basic Avg (BBH, GSM8K, MMLU, TruthfulQA):          {avg_basic:.2f}%")
        if avg_full is not None:
            print(f"  Comprehensive Avg (+ IFEval, GPQA):                {avg_full:.2f}%")

    print("\n" + "=" * 100)

def print_abliteration_analysis(metrics):
    """Analyze abliteration impact for DeepSeek-R1."""
    print("\n" + "=" * 100)
    print("ABLITERATION IMPACT ANALYSIS: DeepSeek-R1-Distill-Llama-8B")
    print("=" * 100)

    base_model = 'DeepSeek-R1-8B\n(Base)'
    abl_model = 'DeepSeek-R1-8B\n(Abliterated)'

    if base_model not in metrics or abl_model not in metrics:
        print("Base or abliterated model not found in results.")
        return

    print(f"\n{'Benchmark':<30} {'Base':<15} {'Abliterated':<15} {'Change':<15}")
    print("-" * 100)

    all_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2', 'ifeval_loose',
                'gpqa_diamond', 'gpqa_extended', 'gpqa_main']

    for key in all_keys:
        base_val = metrics[base_model].get(key)
        abl_val = metrics[abl_model].get(key)

        if base_val is not None and abl_val is not None:
            diff = (abl_val - base_val) * 100
            print(f"{key:<30} {base_val*100:5.2f}%         {abl_val*100:5.2f}%         {diff:+.2f}%")

    print("=" * 100)

def create_visualizations(metrics):
    """Create comprehensive visualization plots."""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Six-Model Comparison: BBH, IFEval, MMLU, TruthfulQA, GPQA, GSM8K',
                 fontsize=18, fontweight='bold')

    model_names = list(metrics.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']

    # 1. Main benchmarks comparison
    ax1 = fig.add_subplot(gs[0, :])
    benchmarks = ['BBH', 'GSM8K', 'MMLU', 'TruthfulQA\nMC2']
    benchmark_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2']

    x = np.arange(len(benchmarks))
    width = 0.13

    for i, model_name in enumerate(model_names):
        scores = [metrics[model_name].get(key, 0) * 100 for key in benchmark_keys]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax1.bar(x + offset, scores, width, label=model_name.replace('\n', ' '),
                      alpha=0.85, color=colors[i])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=7)

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Main Benchmarks Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    ax1.legend(fontsize=9, loc='upper left', ncol=2)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. MMLU categories
    ax2 = fig.add_subplot(gs[1, 0])
    categories = ['Humanities', 'STEM', 'Social\nSciences', 'Other']
    category_keys = ['mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other']

    x = np.arange(len(categories))
    width = 0.13

    for i, model_name in enumerate(model_names):
        scores = [metrics[model_name].get(key, 0) * 100 for key in category_keys]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax2.bar(x + offset, scores, width, label=model_name.replace('\n', ' '),
               alpha=0.85, color=colors[i])

    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('MMLU Categories', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    # 3. GPQA variants
    ax3 = fig.add_subplot(gs[1, 1])
    gpqa_variants = ['Diamond', 'Extended', 'Main']
    gpqa_keys = ['gpqa_diamond', 'gpqa_extended', 'gpqa_main']

    x = np.arange(len(gpqa_variants))
    width = 0.13

    for i, model_name in enumerate(model_names):
        scores = [metrics[model_name].get(key, 0) * 100 for key in gpqa_keys]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax3.bar(x + offset, scores, width, label=model_name.replace('\n', ' '),
               alpha=0.85, color=colors[i])

    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('GPQA Variants', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gpqa_variants)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)

    # 4. IFEval
    ax4 = fig.add_subplot(gs[1, 2])
    ifeval_types = ['Strict', 'Loose']
    ifeval_keys = ['ifeval_strict', 'ifeval_loose']

    x = np.arange(len(ifeval_types))
    width = 0.13

    for i, model_name in enumerate(model_names):
        scores = [metrics[model_name].get(key, 0) * 100 for key in ifeval_keys]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax4.bar(x + offset, scores, width, label=model_name.replace('\n', ' '),
               alpha=0.85, color=colors[i])

    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('IFEval: Instruction Following', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(ifeval_types)
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 100)

    # 5. Abliteration impact for DeepSeek-R1
    ax5 = fig.add_subplot(gs[2, :2])
    base_model = 'DeepSeek-R1-8B\n(Base)'
    abl_model = 'DeepSeek-R1-8B\n(Abliterated)'

    if base_model in metrics and abl_model in metrics:
        impact_benchmarks = ['BBH', 'GSM8K', 'MMLU', 'TruthfulQA', 'IFEval',
                            'GPQA\nDiamond', 'GPQA\nExtended', 'GPQA\nMain']
        impact_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2', 'ifeval_loose',
                      'gpqa_diamond', 'gpqa_extended', 'gpqa_main']

        impact = []
        for key in impact_keys:
            base_val = metrics[base_model].get(key, 0)
            abl_val = metrics[abl_model].get(key, 0)
            impact.append((abl_val - base_val) * 100)

        colors_impact = ['#27AE60' if d >= 0 else '#E74C3C' for d in impact]
        bars = ax5.barh(impact_benchmarks, impact, color=colors_impact, alpha=0.8)
        ax5.set_xlabel('Performance Change (%)', fontsize=12, fontweight='bold')
        ax5.set_title('DeepSeek-R1-8B: Abliteration Impact (Abliterated - Base)',
                     fontsize=13, fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1.0)
        ax5.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, impact):
            ax5.text(val, bar.get_y() + bar.get_height()/2.,
                    f' {val:+.1f}%',
                    ha='left' if val >= 0 else 'right',
                    va='center', fontsize=10, fontweight='bold')

    # 6. Overall average comparison
    ax6 = fig.add_subplot(gs[2, 2])
    avg_names = ['Basic\nAverage', 'Comprehensive\nAverage']
    x = np.arange(len(avg_names))
    width = 0.13

    basic_avgs = []
    comp_avgs = []

    for model_name in model_names:
        basic_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2']
        basic_values = [metrics[model_name].get(k, 0) for k in basic_keys
                       if metrics[model_name].get(k) is not None]
        basic_avg = np.mean(basic_values) * 100 if basic_values else 0

        comp_keys = ['bbh', 'gsm8k', 'mmlu_overall', 'truthfulqa_mc2', 'ifeval_loose',
                    'gpqa_diamond', 'gpqa_extended', 'gpqa_main']
        comp_values = [metrics[model_name].get(k, 0) for k in comp_keys
                      if metrics[model_name].get(k) is not None]
        comp_avg = np.mean(comp_values) * 100 if comp_values else 0

        basic_avgs.append(basic_avg)
        comp_avgs.append(comp_avg)

    for i, model_name in enumerate(model_names):
        scores = [basic_avgs[i], comp_avgs[i]]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax6.bar(x + offset, scores, width, label=model_name.replace('\n', ' '),
                      alpha=0.85, color=colors[i])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=7)

    ax6.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Average Performance', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(avg_names)
    ax6.legend(fontsize=7, ncol=2)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim(0, 100)

    # 7. Ranking by comprehensive average
    ax7 = fig.add_subplot(gs[3, :])

    # Sort models by comprehensive average
    model_scores = []
    for i, model_name in enumerate(model_names):
        score = comp_avgs[i]
        model_scores.append((model_name.replace('\n', ' '), score, colors[i]))

    model_scores.sort(key=lambda x: x[1], reverse=True)

    ranked_names = [x[0] for x in model_scores]
    ranked_scores = [x[1] for x in model_scores]
    ranked_colors = [x[2] for x in model_scores]

    bars = ax7.barh(ranked_names, ranked_scores, color=ranked_colors, alpha=0.8)
    ax7.set_xlabel('Comprehensive Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax7.set_title('Model Ranking by Comprehensive Average', fontsize=14, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, ranked_scores):
        ax7.text(score, bar.get_y() + bar.get_height()/2.,
                f' {score:.2f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_file = RESULTS_DIR / 'six_model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")

def main():
    """Main execution function."""
    print("Loading evaluation results...")
    results = load_results()

    if not results:
        print("No results loaded. Exiting.")
        return

    print("\nExtracting metrics...")
    metrics = extract_metrics(results)

    print_comparison_table(metrics)
    print_summary_statistics(metrics)
    print_abliteration_analysis(metrics)

    print("\nCreating visualizations...")
    create_visualizations(metrics)

    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
