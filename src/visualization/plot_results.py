#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Linear Attention vs. Softmax Attention
Team 3 — Adaptive Attention Computation Benchmark

Generates 4 high-resolution, publication-grade figures:
  1. fig1_efficiency_scaling_curves.png: Latency, Peak VRAM, and Speedup scaling vs N (64 to 65,536)
  2. fig2_retrieval_accuracy_curves.png: Needle-in-a-haystack retrieval accuracy (Global vs Local)
  3. fig3_tradeoff_and_noninferiority.png: Pareto frontier and 95% bootstrap forest plot
  4. fig4_sota_deltanet_comparison.png: SOTA Gated DeltaNet vs Linear Attention on Associative Recall & Throughput
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

# Set matplotlib publication styles
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'lines.linewidth': 2.0,
    'lines.markersize': 6.5,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Standardized color scheme
COLOR_MAP = {
    'softmax (naive)': '#d95f02',       # Vermilion / Orange-Red
    'flash/SDPA (exact)': '#1f78b4',     # Steel Blue
    'linear (ReLU+1)': '#e31a1c',        # Crimson Red
    'sparse (local w=64)': '#33a02c',    # Emerald Green
    'Gated DeltaNet': '#6a3d9a',         # Royal Purple
    'Softmax / SDPA (Exact)': '#1f78b4', # Steel Blue
    'Linear Attention (ReLU+1)': '#e31a1c' # Crimson Red
}

MARKER_MAP = {
    'softmax (naive)': 's',
    'flash/SDPA (exact)': 'o',
    'linear (ReLU+1)': '^',
    'sparse (local w=64)': 'D',
    'Gated DeltaNet': 'p',
    'Softmax / SDPA (Exact)': 'o',
    'Linear Attention (ReLU+1)': '^'
}

LABEL_MAP = {
    'softmax (naive)': 'Naive Softmax (O(N²))',
    'flash/SDPA (exact)': 'PyTorch SDPA (Fused Exact)',
    'linear (ReLU+1)': 'Linear Attention (ReLU+1)',
    'sparse (local w=64)': 'Sparse Attention (Local w=64)',
    'Gated DeltaNet': 'Gated DeltaNet (O(1) State)',
    'Softmax / SDPA (Exact)': 'Softmax / SDPA (Exact)',
    'Linear Attention (ReLU+1)': 'Linear Attention (ReLU+1)'
}


def load_datasets(data_dir: Path):
    """Loads and validates all 5 benchmark CSV datasets."""
    files = {
        'efficiency': data_dir / 'efficiency_raw.csv',
        'retrieval': data_dir / 'retrieval_raw.csv',
        'noninferiority': data_dir / 'noninferiority.csv',
        'deltanet': data_dir / 'deltanet_comparison.csv',
        'qwen': data_dir / 'qwen_prefill_results.csv'
    }
    
    dfs = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
        dfs[name] = pd.read_csv(path)
        print(f"[OK] Loaded {name}: {len(dfs[name])} rows from {path.name}")
    
    return dfs


def plot_fig1_efficiency_scaling(efficiency_df: pd.DataFrame, qwen_df: pd.DataFrame, output_path: Path):
    """
    Figure 1: Computational Scaling Curves & Speedup Profile.
    Panel A: Latency (ms) vs Sequence Length N (log-log) with scaling exponents (alpha).
    Panel B: Peak VRAM Consumption (MB) vs Sequence Length N with T4 Hardware Ceiling.
    Panel C: Empirical Speedup over SDPA across Sequence Lengths (Layer-level & End-to-end Qwen2.5-0.5B).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # -------------------------------------------------------------
    # Panel A: Latency vs N (Log-Log)
    # -------------------------------------------------------------
    ax1 = axes[0]
    methods = ['softmax (naive)', 'flash/SDPA (exact)', 'sparse (local w=64)', 'linear (ReLU+1)']
    
    slopes = {}
    for method in methods:
        sub = efficiency_df[(efficiency_df['method'] == method) & (efficiency_df['status'] == 'ok')].sort_values('N')
        if len(sub) == 0:
            continue
        
        # Calculate scaling exponent alpha on asymptotic regime (N >= 512)
        asymp = sub[sub['N'] >= 512]
        if len(asymp) >= 2:
            alpha, _ = np.polyfit(np.log10(asymp['N']), np.log10(asymp['median_ms']), 1)
        else:
            alpha, _ = np.polyfit(np.log10(sub['N']), np.log10(sub['median_ms']), 1)
        slopes[method] = alpha
        
        lbl = f"{LABEL_MAP.get(method, method)} ($\\alpha={alpha:.2f}$)"
        color = COLOR_MAP.get(method, '#333333')
        marker = MARKER_MAP.get(method, 'o')
        
        ax1.plot(sub['N'], sub['median_ms'], marker=marker, color=color, label=lbl,
                 linewidth=2.2, markersize=7)
        ax1.fill_between(sub['N'], sub['q25_ms'], sub['q75_ms'], color=color, alpha=0.15)
        
    # Annotate OOM for Naive Softmax
    ax1.scatter([32768, 65536], [175.608, 175.608], marker='x', s=100, color='#d95f02', zorder=5)
    ax1.annotate('OOM (Out of Memory\n> 16 GB VRAM)', xy=(32768, 175.608), xytext=(8000, 450),
                 arrowprops=dict(facecolor='#d95f02', arrowstyle='->', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#d95f02',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee8c8', edgecolor='#d95f02', alpha=0.9))
    
    # Annotate 164x speedup at 65536
    ax1.annotate('164.4× Speedup\n(9.7ms vs 1602ms)', xy=(65536, 9.741), xytext=(12000, 1.2),
                 arrowprops=dict(facecolor='#e31a1c', arrowstyle='->', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#b30000',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee0d2', edgecolor='#e31a1c', alpha=0.9))
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence Length $N$ (tokens)')
    ax1.set_ylabel('Execution Latency (ms)')
    ax1.set_title('(A) Latency Scaling vs Sequence Length $N$')
    ax1.grid(True)
    ax1.legend(loc='upper left', framealpha=0.92, fontsize=8.8)
    
    # -------------------------------------------------------------
    # Panel B: Peak VRAM vs N (Log-Log)
    # -------------------------------------------------------------
    ax2 = axes[1]
    for method in methods:
        sub = efficiency_df[(efficiency_df['method'] == method) & (efficiency_df['status'] == 'ok')].sort_values('N')
        if len(sub) == 0:
            continue
        color = COLOR_MAP.get(method, '#333333')
        marker = MARKER_MAP.get(method, 'o')
        lbl = LABEL_MAP.get(method, method)
        ax2.plot(sub['N'], sub['peak_total_mb'], marker=marker, color=color, label=lbl,
                 linewidth=2.2, markersize=7)
    
    # Hardware ceiling line
    ax2.axhline(y=16384, color='#7f0000', linestyle='--', linewidth=1.8, label='Tesla T4 Limit (16 GB)')
    ax2.text(70, 17500, 'T4 VRAM Capacity: 16 GB', color='#7f0000', fontweight='bold', fontsize=9.5)
    
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Sequence Length $N$ (tokens)')
    ax2.set_ylabel('Peak Total GPU Memory (MB)')
    ax2.set_title('(B) Peak VRAM Footprint vs Sequence Length $N$')
    ax2.grid(True)
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.8)
    
    # -------------------------------------------------------------
    # Panel C: Speedup vs Flash/SDPA (Layer-Level & Qwen End-to-End)
    # -------------------------------------------------------------
    ax3 = axes[2]
    
    # Calculate layer speedup relative to flash/SDPA
    sdpa_sub = efficiency_df[(efficiency_df['method'] == 'flash/SDPA (exact)') & (efficiency_df['status'] == 'ok')].set_index('N')['median_ms']
    lin_sub = efficiency_df[(efficiency_df['method'] == 'linear (ReLU+1)') & (efficiency_df['status'] == 'ok')].set_index('N')['median_ms']
    common_n = sdpa_sub.index.intersection(lin_sub.index)
    speedup_layer = sdpa_sub.loc[common_n] / lin_sub.loc[common_n]
    
    ax3.plot(common_n, speedup_layer, marker='^', color='#e31a1c', linewidth=2.4, markersize=8,
             label='Layer-Level Speedup (Linear vs SDPA)')
    
    # Add Qwen2.5-0.5B prefill speedup points
    qwen_lin = qwen_df[qwen_df['method'] == 'Qwen2.5-0.5B (Patched Linear Attention)']
    ax3.plot(qwen_lin['sequence_length_N'], qwen_lin['speedup_vs_sdpa'], marker='s', color='#2b83ba',
             linewidth=2.2, markersize=7.5, linestyle='-.', label='End-to-End Qwen2.5-0.5B Prefill')
    
    qwen_adapt = qwen_df[qwen_df['method'] == 'Qwen2.5-0.5B (Patched Adaptive Rank)']
    ax3.plot(qwen_adapt['sequence_length_N'], qwen_adapt['speedup_vs_sdpa'], marker='d', color='#984ea3',
             linewidth=2.0, markersize=7, linestyle=':', label='End-to-End Qwen Adaptive Rank')
    
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2, label='Parity (1.0× Speedup)')
    
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Sequence Length $N$ (tokens)')
    ax3.set_ylabel('Speedup Factor vs SDPA (Ratio)')
    ax3.set_title('(C) Relative Speedup Factor vs SDPA')
    ax3.grid(True)
    ax3.legend(loc='upper left', framealpha=0.92, fontsize=8.8)
    
    # Speedup values callout
    for n_val, sp in zip([4096, 16384, 65536], [speedup_layer[4096], speedup_layer[16384], speedup_layer[65536]]):
        ax3.annotate(f"{sp:.1f}×", xy=(n_val, sp), xytext=(n_val*0.65, sp*1.35),
                     fontsize=9.5, fontweight='bold', color='#b30000')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close('all')
    print(f"[SAVED] Figure 1: {output_path}")


def plot_fig2_retrieval_accuracy(retrieval_df: pd.DataFrame, output_path: Path):
    """
    Figure 2: Needle-in-a-Haystack Retrieval Accuracy Curves.
    Panel A: Global Target (Random Distance needle across sequence length).
    Panel B: Local Target (Needle strictly within distance <= 64).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    modes = ['global target', 'local target (|dist|<=64)']
    titles = [
        '(A) Global Needle Retrieval (Arbitrary Key-Value Distance)',
        r'(B) Local Needle Retrieval (Distance $\leq 64$ tokens)'
    ]
    methods = ['flash/SDPA (exact)', 'softmax (naive)', 'sparse (local w=64)', 'linear (ReLU+1)']
    
    for ax, mode, title in zip(axes, modes, titles):
        mode_df = retrieval_df[retrieval_df['mode'] == mode]
        
        for method in methods:
            sub = mode_df[mode_df['method'] == method]
            if len(sub) == 0:
                continue
            
            stats = sub.groupby('N')['acc'].agg(['mean', 'std']).reset_index()
            color = COLOR_MAP.get(method, '#333333')
            marker = MARKER_MAP.get(method, 'o')
            lbl = LABEL_MAP.get(method, method)
            
            # Line plot with error points
            ax.plot(stats['N'], stats['mean'] * 100.0, marker=marker, color=color,
                    label=lbl, linewidth=2.4, markersize=8)
            ax.fill_between(stats['N'], (stats['mean'] - stats['std']) * 100.0,
                            (stats['mean'] + stats['std']) * 100.0,
                            color=color, alpha=0.15)
        
        # Random Chance Line (16 classes = 6.25%)
        ax.axhline(y=6.25, color='gray', linestyle=':', linewidth=1.8, label='Random Chance Baseline (6.25%)')
        
        # Formatting
        ax.set_xscale('log', base=2)
        ax.set_ylim(-2, 108)
        ax.set_xlabel('Sequence Length $N$ (tokens)')
        ax.set_ylabel('Retrieval Accuracy (%)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc='lower left' if 'Local' in title else 'center right', framealpha=0.92, fontsize=8.8)
        
        # Annotations
        if mode == 'global target':
            ax.annotate('Softmax / SDPA: 100% Perfect Retrieval', xy=(512, 100), xytext=(200, 85),
                        arrowprops=dict(facecolor='#1f78b4', arrowstyle='->', lw=1.5),
                        fontsize=9.5, fontweight='bold', color='#08519c',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#deebf7', edgecolor='#1f78b4'))
            ax.annotate('Linear Attention Collapses\nto 14.2% at N=1024', xy=(1024, 14.2), xytext=(256, 32),
                        arrowprops=dict(facecolor='#e31a1c', arrowstyle='->', lw=1.5),
                        fontsize=9.5, fontweight='bold', color='#b30000',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fee0d2', edgecolor='#e31a1c'))
        else:
            ax.annotate('Sparse Attention: 100%\n(Needle inside window w=64)', xy=(1024, 100), xytext=(256, 85),
                        arrowprops=dict(facecolor='#33a02c', arrowstyle='->', lw=1.5),
                        fontsize=9.5, fontweight='bold', color='#006d2c',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#e5f5e0', edgecolor='#33a02c'))
            ax.annotate('Linear Attention Collapses\nEven on Local Needles (14.1%)\n(Recurrent crosstalk noise)',
                        xy=(1024, 14.1), xytext=(200, 35),
                        arrowprops=dict(facecolor='#e31a1c', arrowstyle='->', lw=1.5),
                        fontsize=9.2, fontweight='bold', color='#b30000',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fee0d2', edgecolor='#e31a1c'))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close('all')
    print(f"[SAVED] Figure 2: {output_path}")


def plot_fig3_tradeoff_and_noninferiority(retrieval_df: pd.DataFrame, efficiency_df: pd.DataFrame,
                                         noninf_df: pd.DataFrame, output_path: Path):
    """
    Figure 3: Efficiency vs Quality Tradeoff & Formal Non-Inferiority Forest Plot.
    Panel A: Pareto Frontier (Latency Speedup vs Retrieval Accuracy).
    Panel B: Forest Plot of 95% Bootstrap CIs for Difference Delta vs Margin (-0.05).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    
    # -------------------------------------------------------------
    # Panel A: Pareto Frontier (Speedup vs Accuracy)
    # -------------------------------------------------------------
    ax1 = axes[0]
    
    # Match efficiency and global retrieval
    ret_mean = retrieval_df[retrieval_df['mode'] == 'global target'].groupby(['method', 'N'])['acc'].mean().reset_index()
    eff_sub = efficiency_df[efficiency_df['status'] == 'ok'][['method', 'N', 'median_ms']].copy()
    
    # Baseline latency is SDPA
    sdpa_times = eff_sub[eff_sub['method'] == 'flash/SDPA (exact)'].set_index('N')['median_ms']
    
    merged = pd.merge(ret_mean, eff_sub, on=['method', 'N'])
    merged['speedup'] = merged.apply(lambda r: sdpa_times.get(r['N'], np.nan) / r['median_ms'] if r['N'] in sdpa_times else np.nan, axis=1)
    
    for method in ['flash/SDPA (exact)', 'softmax (naive)', 'sparse (local w=64)', 'linear (ReLU+1)']:
        sub = merged[merged['method'] == method].sort_values('N')
        if len(sub) == 0:
            continue
        color = COLOR_MAP.get(method, '#333333')
        marker = MARKER_MAP.get(method, 'o')
        lbl = LABEL_MAP.get(method, method)
        
        ax1.plot(sub['speedup'], sub['acc'] * 100.0, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        scatter = ax1.scatter(sub['speedup'], sub['acc'] * 100.0, color=color, marker=marker,
                              s=np.log2(sub['N']) * 14, label=lbl, edgecolor='black', linewidth=0.6, zorder=4)
        
        # Label points with N
        for _, row in sub.iterrows():
            if row['N'] in [256, 1024, 4096]:
                ax1.annotate(f"N={int(row['N'])}", xy=(row['speedup'], row['acc'] * 100.0),
                             xytext=(row['speedup']*1.08, row['acc'] * 100.0 - 2.5),
                             fontsize=8.2, color=color, fontweight='semibold')
    
    # Quadrant shading
    ax1.axhline(y=95.0, color='gray', linestyle=':', alpha=0.6)
    ax1.axvline(x=2.0, color='gray', linestyle=':', alpha=0.6)
    
    # Text annotations for quadrants
    ax1.text(0.35, 102, 'High Quality, Low Speedup\n(Exact Softmax / SDPA)',
             fontsize=9.0, color='#1f78b4', fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor='#e6f2ff', alpha=0.7))
    ax1.text(6.0, 10, 'High Speedup, Severe Collapse\n(Vanilla Linear Attention)',
             fontsize=9.0, color='#b30000', fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', facecolor='#ffe6e6', alpha=0.7))
    
    ax1.set_xscale('log')
    ax1.set_xlim(0.2, 25)
    ax1.set_ylim(0, 108)
    ax1.set_xlabel('Speedup Factor vs SDPA (Log Scale)')
    ax1.set_ylabel('Retrieval Accuracy (%)')
    ax1.set_title('(A) Pareto Landscape: Speedup vs Retrieval Quality')
    ax1.grid(True)
    ax1.legend(loc='lower left', framealpha=0.92, fontsize=8.8)
    
    # -------------------------------------------------------------
    # Panel B: Non-Inferiority Forest Plot (Delta vs Margin delta_0)
    # -------------------------------------------------------------
    ax2 = axes[1]
    
    # Filter non-inferiority records
    # Order by N and method
    lin_records = noninf_df[noninf_df['method'] == 'linear (ReLU+1)'].copy()
    lin_records['label'] = lin_records.apply(lambda r: f"N={r['N']} ({'Global' if 'global' in r['mode'] else 'Local'})", axis=1)
    lin_records = lin_records.sort_values(by=['mode', 'N'], ascending=[True, False]).reset_index(drop=True)
    
    y_pos = np.arange(len(lin_records))
    
    # Shaded Inferiority / Non-inferiority zones
    ax2.axvspan(-1.05, -0.05, color='#fee0d2', alpha=0.45, label='Inferiority Zone (Delta < -0.05)')
    ax2.axvspan(-0.05, 0.05, color='#e5f5e0', alpha=0.45, label='Non-Inferiority Margin Zone')
    
    # Non-inferiority margin reference line
    ax2.axvline(x=-0.05, color='#de2d26', linestyle='--', linewidth=2.0, label='Margin -delta_0 = -0.05')
    ax2.axvline(x=0.0, color='black', linestyle='-', linewidth=1.2, alpha=0.7, label='Zero Parity (Delta = 0.0)')
    
    # Plot forest points & CIs
    for i, row in lin_records.iterrows():
        delta = row['delta']
        ci_lo = row['ci_lo']
        ci_hi = row['ci_hi']
        is_global = 'global' in row['mode']
        color = '#b30000' if is_global else '#e6550d'
        
        # Horizontal error bar
        ax2.errorbar(delta, y_pos[i], xerr=[[delta - ci_lo], [ci_hi - delta]],
                     fmt='o', color=color, ecolor=color, elinewidth=2.2, capsize=4.5,
                     markersize=7.5, zorder=5)
        # Annotate text
        ax2.text(ci_hi + 0.02, y_pos[i] - 0.15, f"{delta:+.3f} [{ci_lo:.3f}, {ci_hi:.3f}]",
                 fontsize=8.2, color=color, fontweight='semibold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(lin_records['label'])
    ax2.set_xlim(-1.0, 0.15)
    ax2.set_xlabel('Paired Accuracy Difference: $\\Delta = \\mathrm{Acc}_{\\mathrm{linear}} - \\mathrm{Acc}_{\\mathrm{softmax}}$')
    ax2.set_ylabel('Benchmark Evaluation Setup (Sequence Length & Mode)')
    ax2.set_title('(B) Forest Plot: 95% Bootstrap CIs vs Margin ($\\delta_0 = 0.05$)')
    ax2.grid(True)
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.6)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close('all')
    print(f"[SAVED] Figure 3: {output_path}")


def plot_fig4_sota_deltanet_comparison(deltanet_df: pd.DataFrame, output_path: Path):
    """
    Figure 4: SOTA Gated DeltaNet vs Linear Attention on Associative Recall & Throughput.
    Panel A: Multi-Query Associative Recall Accuracy vs Sequence Length N.
    Panel B: Throughput (Tokens / Second) Scaling vs Sequence Length N.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # -------------------------------------------------------------
    # Panel A: Associative Recall Accuracy vs N
    # -------------------------------------------------------------
    ax1 = axes[0]
    
    methods = ['Softmax / SDPA (Exact)', 'Gated DeltaNet', 'Linear Attention (ReLU+1)']
    
    for method in methods:
        sub = deltanet_df[deltanet_df['method'] == method].sort_values('sequence_length_N')
        if len(sub) == 0:
            continue
        
        color = COLOR_MAP.get(method, '#333333')
        marker = MARKER_MAP.get(method, 'o')
        lbl = LABEL_MAP.get(method, method)
        
        ax1.plot(sub['sequence_length_N'], sub['associative_recall_acc'] * 100.0,
                 marker=marker, color=color, label=lbl, linewidth=2.4, markersize=8)
    
    # Annotate DeltaNet breakthrough
    ax1.annotate('Gated DeltaNet: 96.5% Recall at N=1024\n(Preserves memory via delta rule S = S(I - β kkᵀ) + β vkᵀ)',
                 xy=(1024, 96.5), xytext=(1200, 75),
                 arrowprops=dict(facecolor='#6a3d9a', arrowstyle='->', lw=1.6),
                 fontsize=9.2, fontweight='bold', color='#491e70',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2e6ff', edgecolor='#6a3d9a', alpha=0.95))
    
    # Annotate Linear Attention collapse
    ax1.annotate('Linear Attention Collapses\nto 14.2% at N=1024 (7.1% at 16k)',
                 xy=(1024, 14.2), xytext=(1500, 28),
                 arrowprops=dict(facecolor='#e31a1c', arrowstyle='->', lw=1.6),
                 fontsize=9.2, fontweight='bold', color='#b30000',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee0d2', edgecolor='#e31a1c', alpha=0.95))
    
    ax1.set_xscale('log', base=2)
    ax1.set_ylim(-2, 108)
    ax1.set_xlabel('Sequence Length $N$ (tokens)')
    ax1.set_ylabel('Associative Recall Accuracy (%)')
    ax1.set_title('(A) Associative Recall: DeltaNet vs Linear Attention')
    ax1.grid(True)
    ax1.legend(loc='lower left', framealpha=0.92, fontsize=9.0)
    
    # -------------------------------------------------------------
    # Panel B: Throughput (Tokens / sec) vs Sequence Length N
    # -------------------------------------------------------------
    ax2 = axes[1]
    
    for method in methods:
        sub = deltanet_df[deltanet_df['method'] == method].sort_values('sequence_length_N')
        if len(sub) == 0:
            continue
        
        color = COLOR_MAP.get(method, '#333333')
        marker = MARKER_MAP.get(method, 'o')
        lbl = LABEL_MAP.get(method, method)
        
        # Convert to Millions of tokens/sec
        throughput_m = sub['throughput_tokens_sec'] / 1e6
        ax2.plot(sub['sequence_length_N'], throughput_m,
                 marker=marker, color=color, label=lbl, linewidth=2.4, markersize=8)
    
    # Annotation on SDPA throughput crash
    ax2.annotate('SDPA Throughput Degrades\n(O(N²) FLOPs at long context)',
                 xy=(16384, 0.143), xytext=(4000, 1.2),
                 arrowprops=dict(facecolor='#1f78b4', arrowstyle='->', lw=1.5),
                 fontsize=9.0, fontweight='bold', color='#08519c',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#deebf7', edgecolor='#1f78b4'))
    
    # Annotation on DeltaNet maintaining linear throughput
    ax2.annotate('DeltaNet Maintains Linear O(N) Scaling\n(~3.0M - 3.8M tokens/sec)',
                 xy=(8192, 3.828), xytext=(2000, 4.8),
                 arrowprops=dict(facecolor='#6a3d9a', arrowstyle='->', lw=1.5),
                 fontsize=9.0, fontweight='bold', color='#491e70',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#f2e6ff', edgecolor='#6a3d9a'))
    
    # Add Recurrent Memory footprint box
    table_text = (
        "Recurrent State Footprint:\n"
        "• Linear Attention: 16 KB (Fixed O(1))\n"
        "• Gated DeltaNet: 16 KB (Fixed O(1))\n"
        "• Softmax KV Cache: 512 KB → 8,192 KB (O(N))"
    )
    ax2.text(0.04, 0.05, table_text, transform=ax2.transAxes,
             fontsize=8.8, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', edgecolor='#b2b2b2', alpha=0.95))
    
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length $N$ (tokens)')
    ax2.set_ylabel('Throughput ($10^6$ tokens / sec)')
    ax2.set_title('(B) Processing Throughput Scaling vs Sequence Length $N$')
    ax2.grid(True)
    ax2.legend(loc='center right', framealpha=0.92, fontsize=9.0)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close('all')
    print(f"[SAVED] Figure 4: {output_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / 'data'
    figures_dir = repo_root / 'docs_markdown' / 'figures'
    
    print(f"=== Publication Visualization Suite ===")
    print(f"Loading data from: {data_dir}")
    print(f"Target figures directory: {figures_dir}")
    
    dfs = load_datasets(data_dir)
    
    # Generate all 4 figures
    fig1_path = figures_dir / 'fig1_efficiency_scaling_curves.png'
    plot_fig1_efficiency_scaling(dfs['efficiency'], dfs['qwen'], fig1_path)
    
    fig2_path = figures_dir / 'fig2_retrieval_accuracy_curves.png'
    plot_fig2_retrieval_accuracy(dfs['retrieval'], fig2_path)
    
    fig3_path = figures_dir / 'fig3_tradeoff_and_noninferiority.png'
    plot_fig3_tradeoff_and_noninferiority(dfs['retrieval'], dfs['efficiency'], dfs['noninferiority'], fig3_path)
    
    fig4_path = figures_dir / 'fig4_sota_deltanet_comparison.png'
    plot_fig4_sota_deltanet_comparison(dfs['deltanet'], fig4_path)
    
    # Verification
    print("\n=== Verifying Generated Figures ===")
    for p in [fig1_path, fig2_path, fig3_path, fig4_path]:
        if p.exists() and p.stat().st_size > 0:
            print(f"  [VERIFIED] {p.name}: {p.stat().st_size / 1024:.1f} KB")
        else:
            raise RuntimeError(f"Failed to generate figure: {p}")
            
    print("\nAll 4 publication figures successfully generated and verified!")


if __name__ == '__main__':
    main()
