#!/usr/bin/env python3
"""
Four-way comparison of language models:
1. huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2 (abliterated GPT-OSS)
2. openai/gpt-oss-20b (base GPT-OSS)
3. Qwen/Qwen3-14B (base Qwen3)
4. huihui-ai/Huihui-Qwen3-14B-abliterated-v2 (abliterated Qwen3)

NOTE: Qwen3 (base) was evaluated with limit=50 samples per task.
      Other models were evaluated on full datasets.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load result files
abliterated_gpt = json.load(open('lm_eval_results/ifeval_arc_challenge_gsm8k_mmlu_gpqa_results/huihui-ai__Huihui-gpt-oss-20b-mxfp4-abliterated-v2/results_2025-11-14T18-44-57.626713.json'))
base_gpt = json.load(open('lm_eval_results/ifeval_arc_challenge_gsm8k_mmlu_gpqa_results/openai__gpt-oss-20b/results_2025-11-14T17-58-04.419412.json'))
base_qwen3 = json.load(open('lm_eval_results/ifeval_arc_challenge_gsm8k_mmlu_gpqa_results/Qwen__Qwen3-14B/results_2025-11-14T19-27-13.219827.json'))
abliterated_qwen3 = json.load(open('lm_eval_results/ifeval_arc_challenge_gsm8k_mmlu_gpqa_results/huihui-ai__Huihui-Qwen3-14B-abliterated-v2/results_2025-11-14T22-06-11.949284.json'))

# Extract metrics for all four models
def extract_metrics():
    metrics = {
        'GPT-OSS\nAbliterated': {
            'arc_challenge': abliterated_gpt['results']['arc_challenge']['acc_norm,none'],
            'gsm8k': abliterated_gpt['results']['gsm8k']['exact_match,flexible-extract'],
            'mmlu_overall': abliterated_gpt['results']['mmlu']['acc,none'],
            'mmlu_humanities': abliterated_gpt['results']['mmlu_humanities']['acc,none'],
            'mmlu_stem': abliterated_gpt['results']['mmlu_stem']['acc,none'],
            'mmlu_social_sciences': abliterated_gpt['results']['mmlu_social_sciences']['acc,none'],
            'mmlu_other': abliterated_gpt['results']['mmlu_other']['acc,none'],
            'ifeval_strict': abliterated_gpt['results']['ifeval']['prompt_level_strict_acc,none'],
            'ifeval_loose': abliterated_gpt['results']['ifeval']['prompt_level_loose_acc,none'],
            'gpqa_diamond': abliterated_gpt['results']['gpqa_diamond_n_shot']['acc,none'],
            'gpqa_extended': abliterated_gpt['results']['gpqa_extended_n_shot']['acc,none'],
            'gpqa_main': abliterated_gpt['results']['gpqa_main_n_shot']['acc,none'],
        },
        'GPT-OSS\nBase': {
            'arc_challenge': base_gpt['results']['arc_challenge']['acc_norm,none'],
            'gsm8k': base_gpt['results']['gsm8k']['exact_match,flexible-extract'],
            'mmlu_overall': base_gpt['results']['mmlu']['acc,none'],
            'mmlu_humanities': base_gpt['results']['mmlu_humanities']['acc,none'],
            'mmlu_stem': base_gpt['results']['mmlu_stem']['acc,none'],
            'mmlu_social_sciences': base_gpt['results']['mmlu_social_sciences']['acc,none'],
            'mmlu_other': base_gpt['results']['mmlu_other']['acc,none'],
            'ifeval_strict': base_gpt['results']['ifeval']['prompt_level_strict_acc,none'],
            'ifeval_loose': base_gpt['results']['ifeval']['prompt_level_loose_acc,none'],
            'gpqa_diamond': base_gpt['results']['gpqa_diamond_n_shot']['acc,none'],
            'gpqa_extended': base_gpt['results']['gpqa_extended_n_shot']['acc,none'],
            'gpqa_main': base_gpt['results']['gpqa_main_n_shot']['acc,none'],
        },
        'Qwen3-14B\nBase': {
            'arc_challenge': base_qwen3['results']['arc_challenge']['acc_norm,none'],
            'gsm8k': base_qwen3['results']['gsm8k']['exact_match,flexible-extract'],
            'mmlu_overall': base_qwen3['results']['mmlu']['acc,none'],
            'mmlu_humanities': base_qwen3['results']['mmlu_humanities']['acc,none'],
            'mmlu_stem': base_qwen3['results']['mmlu_stem']['acc,none'],
            'mmlu_social_sciences': base_qwen3['results']['mmlu_social_sciences']['acc,none'],
            'mmlu_other': base_qwen3['results']['mmlu_other']['acc,none'],
            'ifeval_strict': base_qwen3['results']['ifeval']['prompt_level_strict_acc,none'],
            'ifeval_loose': base_qwen3['results']['ifeval']['prompt_level_loose_acc,none'],
            'gpqa_diamond': base_qwen3['results']['gpqa_diamond_n_shot']['acc,none'],
            'gpqa_extended': base_qwen3['results']['gpqa_extended_n_shot']['acc,none'],
            'gpqa_main': base_qwen3['results']['gpqa_main_n_shot']['acc,none'],
        },
        'Qwen3-14B\nAbliterated': {
            'arc_challenge': abliterated_qwen3['results']['arc_challenge']['acc_norm,none'],
            'gsm8k': abliterated_qwen3['results']['gsm8k']['exact_match,flexible-extract'],
            'mmlu_overall': abliterated_qwen3['results']['mmlu']['acc,none'],
            'mmlu_humanities': abliterated_qwen3['results']['mmlu_humanities']['acc,none'],
            'mmlu_stem': abliterated_qwen3['results']['mmlu_stem']['acc,none'],
            'mmlu_social_sciences': abliterated_qwen3['results']['mmlu_social_sciences']['acc,none'],
            'mmlu_other': abliterated_qwen3['results']['mmlu_other']['acc,none'],
            'ifeval_strict': abliterated_qwen3['results']['ifeval']['prompt_level_strict_acc,none'],
            'ifeval_loose': abliterated_qwen3['results']['ifeval']['prompt_level_loose_acc,none'],
            'gpqa_diamond': abliterated_qwen3['results']['gpqa_diamond_n_shot']['acc,none'],
            'gpqa_extended': abliterated_qwen3['results']['gpqa_extended_n_shot']['acc,none'],
            'gpqa_main': abliterated_qwen3['results']['gpqa_main_n_shot']['acc,none'],
        }
    }
    return metrics

metrics = extract_metrics()

# Print comparison table
print("=" * 120)
print("FOUR-WAY MODEL COMPARISON")
print("=" * 120)
print("NOTE: Base Qwen3-14B was evaluated with limit=50 samples per task")
print("      Other models were evaluated on full datasets")
print("=" * 120)
print(f"{'Benchmark':<25} {'GPT-OSS Abl':<15} {'GPT-OSS Base':<15} {'Qwen3 Base':<15} {'Qwen3 Abl':<15}")
print("-" * 120)

for key in ['arc_challenge', 'gsm8k', 'mmlu_overall', 'mmlu_humanities', 'mmlu_stem',
            'mmlu_social_sciences', 'mmlu_other', 'ifeval_strict', 'ifeval_loose',
            'gpqa_diamond', 'gpqa_extended', 'gpqa_main']:
    gpt_abl = metrics['GPT-OSS\nAbliterated'][key]
    gpt_base = metrics['GPT-OSS\nBase'][key]
    qwen_base = metrics['Qwen3-14B\nBase'][key]
    qwen_abl = metrics['Qwen3-14B\nAbliterated'][key]

    print(f"{key:<25} {gpt_abl*100:5.2f}%         {gpt_base*100:5.2f}%         {qwen_base*100:5.2f}%         {qwen_abl*100:5.2f}%")

print("=" * 120)

# Calculate averages
print("\n" + "=" * 120)
print("SUMMARY STATISTICS")
print("=" * 120)

for model_name, model_data in metrics.items():
    avg_basic = np.mean([model_data[k] for k in ['arc_challenge', 'gsm8k', 'mmlu_overall']])
    avg_full = np.mean([
        model_data[k] for k in [
            'arc_challenge', 'gsm8k', 'mmlu_overall',
            'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main'
        ]
    ])
    print(f"\n{model_name}:")
    print(f"  Basic Avg (ARC, GSM8K, MMLU):        {avg_basic*100:.2f}%")
    print(f"  Comprehensive Avg (+ IFEval, GPQA):  {avg_full*100:.2f}%")

print("\n" + "=" * 120)

# Abliteration impact analysis
print("\nABLITERATION IMPACT ANALYSIS")
print("=" * 120)

print("\nGPT-OSS Abliteration Impact:")
for key in ['arc_challenge', 'gsm8k', 'mmlu_overall', 'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main']:
    diff = (metrics['GPT-OSS\nAbliterated'][key] - metrics['GPT-OSS\nBase'][key]) * 100
    print(f"  {key:<25} {diff:+.2f}%")

print("\nQwen3-14B Abliteration Impact:")
for key in ['arc_challenge', 'gsm8k', 'mmlu_overall', 'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main']:
    diff = (metrics['Qwen3-14B\nAbliterated'][key] - metrics['Qwen3-14B\nBase'][key]) * 100
    print(f"  {key:<25} {diff:+.2f}%")

print("=" * 120)

# Create comprehensive bar plot visualization
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Four-Way Model Comparison: GPT-OSS vs Qwen3-14B (Base vs Abliterated)\nNote: Base Qwen3 evaluated on 50 samples/task; others on full datasets',
             fontsize=16, fontweight='bold')

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
model_names = ['GPT-OSS\nAbliterated', 'GPT-OSS\nBase', 'Qwen3-14B\nBase', 'Qwen3-14B\nAbliterated']

# 1. Main benchmarks
ax1 = fig.add_subplot(gs[0, :])
benchmarks = ['ARC Challenge', 'GSM8K', 'MMLU Overall']
benchmark_keys = ['arc_challenge', 'gsm8k', 'mmlu_overall']

x = np.arange(len(benchmarks))
width = 0.2

for i, model_name in enumerate(model_names):
    scores = [metrics[model_name][key] * 100 for key in benchmark_keys]
    bars = ax1.bar(x + i*width - 1.5*width, scores, width, label=model_name, alpha=0.85, color=colors[i])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)

ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Main Benchmarks Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(benchmarks)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# 2. MMLU categories
ax2 = fig.add_subplot(gs[1, 0])
categories = ['Humanities', 'STEM', 'Social Sci', 'Other']
category_keys = ['mmlu_humanities', 'mmlu_stem', 'mmlu_social_sciences', 'mmlu_other']

x = np.arange(len(categories))
width = 0.2

for i, model_name in enumerate(model_names):
    scores = [metrics[model_name][key] * 100 for key in category_keys]
    bars = ax2.bar(x + i*width - 1.5*width, scores, width, label=model_name, alpha=0.85, color=colors[i])

ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('MMLU Categories', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# 3. GPQA variants
ax3 = fig.add_subplot(gs[1, 1])
gpqa_variants = ['Diamond', 'Extended', 'Main']
gpqa_keys = ['gpqa_diamond', 'gpqa_extended', 'gpqa_main']

x = np.arange(len(gpqa_variants))
width = 0.2

for i, model_name in enumerate(model_names):
    scores = [metrics[model_name][key] * 100 for key in gpqa_keys]
    bars = ax3.bar(x + i*width - 1.5*width, scores, width, label=model_name, alpha=0.85, color=colors[i])

ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('GPQA Variants', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(gpqa_variants)
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# 4. IFEval
ax4 = fig.add_subplot(gs[1, 2])
ifeval_types = ['Strict', 'Loose']
ifeval_keys = ['ifeval_strict', 'ifeval_loose']

x = np.arange(len(ifeval_types))
width = 0.2

for i, model_name in enumerate(model_names):
    scores = [metrics[model_name][key] * 100 for key in ifeval_keys]
    bars = ax4.bar(x + i*width - 1.5*width, scores, width, label=model_name, alpha=0.85, color=colors[i])

ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('IFEval: Instruction Following', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(ifeval_types)
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# 5. Abliteration impact for GPT-OSS
ax5 = fig.add_subplot(gs[2, 0])
impact_benchmarks = ['ARC', 'GSM8K', 'MMLU', 'IFEval', 'GPQA\nDiamond', 'GPQA\nExtended', 'GPQA\nMain']
impact_keys = ['arc_challenge', 'gsm8k', 'mmlu_overall', 'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main']

gpt_impact = [(metrics['GPT-OSS\nAbliterated'][key] - metrics['GPT-OSS\nBase'][key]) * 100 for key in impact_keys]
colors_impact = ['#27AE60' if d >= 0 else '#E74C3C' for d in gpt_impact]

bars = ax5.barh(impact_benchmarks, gpt_impact, color=colors_impact, alpha=0.8)
ax5.set_xlabel('Performance Change (%)', fontsize=11, fontweight='bold')
ax5.set_title('GPT-OSS: Abliteration Impact', fontsize=12, fontweight='bold')
ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax5.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, gpt_impact)):
    ax5.text(val, bar.get_y() + bar.get_height()/2.,
            f' {val:+.1f}%',
            ha='left' if val >= 0 else 'right',
            va='center', fontsize=9, fontweight='bold')

# 6. Abliteration impact for Qwen3
ax6 = fig.add_subplot(gs[2, 1])
qwen_impact = [(metrics['Qwen3-14B\nAbliterated'][key] - metrics['Qwen3-14B\nBase'][key]) * 100 for key in impact_keys]
colors_impact = ['#27AE60' if d >= 0 else '#E74C3C' for d in qwen_impact]

bars = ax6.barh(impact_benchmarks, qwen_impact, color=colors_impact, alpha=0.8)
ax6.set_xlabel('Performance Change (%)', fontsize=11, fontweight='bold')
ax6.set_title('Qwen3-14B: Abliteration Impact', fontsize=12, fontweight='bold')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax6.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, qwen_impact)):
    ax6.text(val, bar.get_y() + bar.get_height()/2.,
            f' {val:+.1f}%',
            ha='left' if val >= 0 else 'right',
            va='center', fontsize=9, fontweight='bold')

# 7. Overall average comparison
ax7 = fig.add_subplot(gs[2, 2])
avg_names = ['Basic\nAverage', 'Comprehensive\nAverage']
x = np.arange(len(avg_names))
width = 0.2

basic_avgs = []
comp_avgs = []

for model_name in model_names:
    basic_avg = np.mean([metrics[model_name][k] for k in ['arc_challenge', 'gsm8k', 'mmlu_overall']]) * 100
    comp_avg = np.mean([
        metrics[model_name][k] for k in [
            'arc_challenge', 'gsm8k', 'mmlu_overall',
            'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main'
        ]
    ]) * 100
    basic_avgs.append(basic_avg)
    comp_avgs.append(comp_avg)

for i, model_name in enumerate(model_names):
    scores = [basic_avgs[i], comp_avgs[i]]
    bars = ax7.bar(x + i*width - 1.5*width, scores, width, label=model_name, alpha=0.85, color=colors[i])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=7)

ax7.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax7.set_title('Average Performance', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(avg_names)
ax7.legend(fontsize=7)
ax7.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_file = 'lm_eval_results/four_way_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")
