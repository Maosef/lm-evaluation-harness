#!/usr/bin/env python3
"""
Compare evaluation results between two GPT-OSS models:
1. huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2 (abliterated)
2. openai/gpt-oss-20b (base model)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the result files
abliterated_all = json.load(open('lm_eval_results/ifeval_arc_challenge_gsm8k_mmlu_gpqa_results/huihui-ai__Huihui-gpt-oss-20b-mxfp4-abliterated-v2/results_2025-11-14T18-44-57.626713.json'))
base_all = json.load(open('lm_eval_results/ifeval_arc_challenge_gsm8k_mmlu_gpqa_results/openai__gpt-oss-20b/results_2025-11-14T17-58-04.419412.json'))

# Extract key metrics
def extract_metrics():
    metrics = {
        'abliterated': {
            'arc_challenge': abliterated_all['results']['arc_challenge']['acc_norm,none'],
            'gsm8k': abliterated_all['results']['gsm8k']['exact_match,flexible-extract'],
            'mmlu_overall': abliterated_all['results']['mmlu']['acc,none'],
            'mmlu_humanities': abliterated_all['results']['mmlu_humanities']['acc,none'],
            'mmlu_stem': abliterated_all['results']['mmlu_stem']['acc,none'],
            'mmlu_social_sciences': abliterated_all['results']['mmlu_social_sciences']['acc,none'],
            'mmlu_other': abliterated_all['results']['mmlu_other']['acc,none'],
            'ifeval_strict': abliterated_all['results']['ifeval']['prompt_level_strict_acc,none'],
            'ifeval_loose': abliterated_all['results']['ifeval']['prompt_level_loose_acc,none'],
            'gpqa_diamond': abliterated_all['results']['gpqa_diamond_n_shot']['acc,none'],
            'gpqa_extended': abliterated_all['results']['gpqa_extended_n_shot']['acc,none'],
            'gpqa_main': abliterated_all['results']['gpqa_main_n_shot']['acc,none'],
        },
        'base': {
            'arc_challenge': base_all['results']['arc_challenge']['acc_norm,none'],
            'gsm8k': base_all['results']['gsm8k']['exact_match,flexible-extract'],
            'mmlu_overall': base_all['results']['mmlu']['acc,none'],
            'mmlu_humanities': base_all['results']['mmlu_humanities']['acc,none'],
            'mmlu_stem': base_all['results']['mmlu_stem']['acc,none'],
            'mmlu_social_sciences': base_all['results']['mmlu_social_sciences']['acc,none'],
            'mmlu_other': base_all['results']['mmlu_other']['acc,none'],
            'ifeval_strict': base_all['results']['ifeval']['prompt_level_strict_acc,none'],
            'ifeval_loose': base_all['results']['ifeval']['prompt_level_loose_acc,none'],
            'gpqa_diamond': base_all['results']['gpqa_diamond_n_shot']['acc,none'],
            'gpqa_extended': base_all['results']['gpqa_extended_n_shot']['acc,none'],
            'gpqa_main': base_all['results']['gpqa_main_n_shot']['acc,none'],
        }
    }
    return metrics

metrics = extract_metrics()

# Print comparison table
print("=" * 80)
print("EVALUATION RESULTS COMPARISON")
print("=" * 80)
print(f"{'Benchmark':<30} {'Abliterated':<15} {'Base Model':<15} {'Diff':<15}")
print("-" * 80)

for key in metrics['abliterated'].keys():
    abl = metrics['abliterated'][key]
    base = metrics['base'].get(key, None)
    if base is not None:
        diff = abl - base
        diff_str = f"{diff:+.4f} ({diff*100:+.2f}%)"
        print(f"{key:<30} {abl:.4f} ({abl*100:.2f}%){'':<2} {base:.4f} ({base*100:.2f}%){'':<2} {diff_str}")
    else:
        print(f"{key:<30} {abl:.4f} ({abl*100:.2f}%){'':<2} {'N/A':<15} {'N/A':<15}")

# IFEval comparison (now both models have it)
print("\nIFEval Metrics:")
print("-" * 80)
for key in ['ifeval_strict', 'ifeval_loose']:
    abl = metrics['abliterated'][key]
    base = metrics['base'][key]
    diff = abl - base
    diff_str = f"{diff:+.4f} ({diff*100:+.2f}%)"
    print(f"{key:<30} {abl:.4f} ({abl*100:.2f}%){'':<2} {base:.4f} ({base*100:.2f}%){'':<2} {diff_str}")

# GPQA comparison
print("\nGPQA Metrics:")
print("-" * 80)
for key in ['gpqa_diamond', 'gpqa_extended', 'gpqa_main']:
    abl = metrics['abliterated'][key]
    base = metrics['base'][key]
    diff = abl - base
    diff_str = f"{diff:+.4f} ({diff*100:+.2f}%)"
    print(f"{key:<30} {abl:.4f} ({abl*100:.2f}%){'':<2} {base:.4f} ({base*100:.2f}%){'':<2} {diff_str}")

print("=" * 80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('GPT-OSS Model Comparison: Abliterated vs Base', fontsize=16, fontweight='bold')

# 1. Overall benchmark comparison
ax1 = axes[0, 0]
benchmarks = ['ARC\nChallenge', 'GSM8K', 'MMLU\nOverall']
abl_scores = [
    metrics['abliterated']['arc_challenge'] * 100,
    metrics['abliterated']['gsm8k'] * 100,
    metrics['abliterated']['mmlu_overall'] * 100,
]
base_scores = [
    metrics['base']['arc_challenge'] * 100,
    metrics['base']['gsm8k'] * 100,
    metrics['base']['mmlu_overall'] * 100,
]

x = np.arange(len(benchmarks))
width = 0.35

bars1 = ax1.bar(x - width/2, abl_scores, width, label='Abliterated', alpha=0.8, color='#2E86AB')
bars2 = ax1.bar(x + width/2, base_scores, width, label='Base Model', alpha=0.8, color='#A23B72')

ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Main Benchmarks Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(benchmarks)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

# 2. MMLU category breakdown
ax2 = axes[0, 1]
categories = ['Humanities', 'STEM', 'Social\nSciences', 'Other']
abl_mmlu = [
    metrics['abliterated']['mmlu_humanities'] * 100,
    metrics['abliterated']['mmlu_stem'] * 100,
    metrics['abliterated']['mmlu_social_sciences'] * 100,
    metrics['abliterated']['mmlu_other'] * 100,
]
base_mmlu = [
    metrics['base']['mmlu_humanities'] * 100,
    metrics['base']['mmlu_stem'] * 100,
    metrics['base']['mmlu_social_sciences'] * 100,
    metrics['base']['mmlu_other'] * 100,
]

x = np.arange(len(categories))
bars1 = ax2.bar(x - width/2, abl_mmlu, width, label='Abliterated', alpha=0.8, color='#2E86AB')
bars2 = ax2.bar(x + width/2, base_mmlu, width, label='Base Model', alpha=0.8, color='#A23B72')

ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('MMLU Categories Breakdown', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

# 3. Performance difference (Delta)
ax3 = axes[1, 0]
all_metrics = ['ARC', 'GSM8K', 'MMLU\nOverall', 'MMLU\nHumanities', 'MMLU\nSTEM', 'MMLU\nSocial', 'MMLU\nOther']
differences = [
    (metrics['abliterated']['arc_challenge'] - metrics['base']['arc_challenge']) * 100,
    (metrics['abliterated']['gsm8k'] - metrics['base']['gsm8k']) * 100,
    (metrics['abliterated']['mmlu_overall'] - metrics['base']['mmlu_overall']) * 100,
    (metrics['abliterated']['mmlu_humanities'] - metrics['base']['mmlu_humanities']) * 100,
    (metrics['abliterated']['mmlu_stem'] - metrics['base']['mmlu_stem']) * 100,
    (metrics['abliterated']['mmlu_social_sciences'] - metrics['base']['mmlu_social_sciences']) * 100,
    (metrics['abliterated']['mmlu_other'] - metrics['base']['mmlu_other']) * 100,
]

colors = ['#27AE60' if d >= 0 else '#E74C3C' for d in differences]
bars = ax3.barh(all_metrics, differences, color=colors, alpha=0.8)

ax3.set_xlabel('Performance Difference (Abliterated - Base) %', fontsize=11, fontweight='bold')
ax3.set_title('Performance Delta: Abliterated vs Base', fontsize=12, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, differences)):
    ax3.text(val, bar.get_y() + bar.get_height()/2.,
            f' {val:+.1f}%',
            ha='left' if val >= 0 else 'right',
            va='center', fontsize=9, fontweight='bold')

# 4. Summary statistics table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

summary_data = [
    ['Metric', 'Abliterated', 'Base', 'Delta'],
    ['', '', '', ''],
    ['ARC Challenge', f"{metrics['abliterated']['arc_challenge']*100:.2f}%",
     f"{metrics['base']['arc_challenge']*100:.2f}%",
     f"{(metrics['abliterated']['arc_challenge']-metrics['base']['arc_challenge'])*100:+.2f}%"],
    ['GSM8K', f"{metrics['abliterated']['gsm8k']*100:.2f}%",
     f"{metrics['base']['gsm8k']*100:.2f}%",
     f"{(metrics['abliterated']['gsm8k']-metrics['base']['gsm8k'])*100:+.2f}%"],
    ['MMLU Overall', f"{metrics['abliterated']['mmlu_overall']*100:.2f}%",
     f"{metrics['base']['mmlu_overall']*100:.2f}%",
     f"{(metrics['abliterated']['mmlu_overall']-metrics['base']['mmlu_overall'])*100:+.2f}%"],
    ['  - Humanities', f"{metrics['abliterated']['mmlu_humanities']*100:.2f}%",
     f"{metrics['base']['mmlu_humanities']*100:.2f}%",
     f"{(metrics['abliterated']['mmlu_humanities']-metrics['base']['mmlu_humanities'])*100:+.2f}%"],
    ['  - STEM', f"{metrics['abliterated']['mmlu_stem']*100:.2f}%",
     f"{metrics['base']['mmlu_stem']*100:.2f}%",
     f"{(metrics['abliterated']['mmlu_stem']-metrics['base']['mmlu_stem'])*100:+.2f}%"],
    ['  - Social Sciences', f"{metrics['abliterated']['mmlu_social_sciences']*100:.2f}%",
     f"{metrics['base']['mmlu_social_sciences']*100:.2f}%",
     f"{(metrics['abliterated']['mmlu_social_sciences']-metrics['base']['mmlu_social_sciences'])*100:+.2f}%"],
    ['  - Other', f"{metrics['abliterated']['mmlu_other']*100:.2f}%",
     f"{metrics['base']['mmlu_other']*100:.2f}%",
     f"{(metrics['abliterated']['mmlu_other']-metrics['base']['mmlu_other'])*100:+.2f}%"],
    ['', '', '', ''],
    ['IFEval (Strict)', f"{metrics['abliterated']['ifeval_strict']*100:.2f}%",
     f"{metrics['base']['ifeval_strict']*100:.2f}%",
     f"{(metrics['abliterated']['ifeval_strict']-metrics['base']['ifeval_strict'])*100:+.2f}%"],
    ['IFEval (Loose)', f"{metrics['abliterated']['ifeval_loose']*100:.2f}%",
     f"{metrics['base']['ifeval_loose']*100:.2f}%",
     f"{(metrics['abliterated']['ifeval_loose']-metrics['base']['ifeval_loose'])*100:+.2f}%"],
    ['', '', '', ''],
    ['GPQA Diamond', f"{metrics['abliterated']['gpqa_diamond']*100:.2f}%",
     f"{metrics['base']['gpqa_diamond']*100:.2f}%",
     f"{(metrics['abliterated']['gpqa_diamond']-metrics['base']['gpqa_diamond'])*100:+.2f}%"],
    ['GPQA Extended', f"{metrics['abliterated']['gpqa_extended']*100:.2f}%",
     f"{metrics['base']['gpqa_extended']*100:.2f}%",
     f"{(metrics['abliterated']['gpqa_extended']-metrics['base']['gpqa_extended'])*100:+.2f}%"],
    ['GPQA Main', f"{metrics['abliterated']['gpqa_main']*100:.2f}%",
     f"{metrics['base']['gpqa_main']*100:.2f}%",
     f"{(metrics['abliterated']['gpqa_main']-metrics['base']['gpqa_main'])*100:+.2f}%"],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.35, 0.2, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style the header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code the delta column
for i in range(2, len(summary_data)):  # Color all data rows
    if i < len(summary_data) and summary_data[i][3]:  # Skip empty rows
        cell = table[(i, 3)]
        text = summary_data[i][3]
        if text.startswith('+'):
            cell.set_facecolor('#D5F4E6')
        elif text.startswith('-'):
            cell.set_facecolor('#FADBD8')

ax4.set_title('Detailed Performance Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('lm_eval_results/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: lm_eval_results/model_comparison.png")

# Calculate average performance
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

abl_avg_basic = np.mean([metrics['abliterated'][k] for k in ['arc_challenge', 'gsm8k', 'mmlu_overall']])
base_avg_basic = np.mean([metrics['base'][k] for k in ['arc_challenge', 'gsm8k', 'mmlu_overall']])

print(f"\nAverage Performance (ARC, GSM8K, MMLU):")
print(f"  Abliterated: {abl_avg_basic*100:.2f}%")
print(f"  Base Model:  {base_avg_basic*100:.2f}%")
print(f"  Difference:  {(abl_avg_basic-base_avg_basic)*100:+.2f}%")

# Calculate comprehensive average including IFEval and GPQA
abl_avg_full = np.mean([
    metrics['abliterated'][k] for k in [
        'arc_challenge', 'gsm8k', 'mmlu_overall',
        'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main'
    ]
])
base_avg_full = np.mean([
    metrics['base'][k] for k in [
        'arc_challenge', 'gsm8k', 'mmlu_overall',
        'ifeval_loose', 'gpqa_diamond', 'gpqa_extended', 'gpqa_main'
    ]
])

print(f"\nAverage Performance (ARC, GSM8K, MMLU, IFEval, GPQA):")
print(f"  Abliterated: {abl_avg_full*100:.2f}%")
print(f"  Base Model:  {base_avg_full*100:.2f}%")
print(f"  Difference:  {(abl_avg_full-base_avg_full)*100:+.2f}%")

print("\n" + "=" * 80)
