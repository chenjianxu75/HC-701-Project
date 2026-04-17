"""
Baseline Results Visualization — Publication-Quality Figures
Generates grouped bar charts, line charts, and radar charts
for the Kvasir-trained baseline experiments.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from pathlib import Path

# ── Use non-interactive backend ──────────────────────────────────────
matplotlib.use('Agg')

# ── Global Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Color Palette ────────────────────────────────────────────────────
COLORS = {
    'YOLOv8n-seg':  '#6366f1',   # Indigo
    'YOLOv11s-seg': '#f59e0b',   # Amber
    'RT-DETR-L':    '#10b981',   # Emerald
}

DATASET_COLORS = {
    'Kvasir':  '#6366f1',
    'CVC':     '#f59e0b',
    'ETIS':    '#ef4444',
}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

# Evolution data (trained on Kvasir)
evolution = {
    'YOLOv8n-seg': {
        'epochs': [5, 50, 100],
        'Kvasir': [88.5, 94.6, 95.4],
        'CVC':    [38.6, 42.0, 43.3],
        'ETIS':   [35.2, 38.0, 39.5],
    },
    'YOLOv11s-seg': {
        'epochs': [5, 50, 100],
        'Kvasir': [91.1, 95.4, 95.8],
        'CVC':    [42.6, 82.7, 82.4],
        'ETIS':   [53.5, 78.0, 77.8],
    },
    'RT-DETR-L': {
        'epochs': [100],
        'Kvasir': [96.3],
        'CVC':    [87.7],
        'ETIS':   [87.9],
    },
}

# Detailed metrics at 100ep (trained on Kvasir)
detailed_kvasir = {
    'YOLOv8n-seg':  {'P': 93.08, 'R': 91.01, 'mAP50': 95.4, 'mAP50-95': 77.74},
    'YOLOv11s-seg': {'P': 92.75, 'R': 92.77, 'mAP50': 95.8, 'mAP50-95': 79.16},
    'RT-DETR-L':    {'P': 92.65, 'R': 91.41, 'mAP50': 96.3, 'mAP50-95': 79.93},
}

detailed_cvc = {
    'YOLOv8n-seg':  {'P': 41.25, 'R': 44.52, 'mAP50': 43.3, 'mAP50-95': 24.11},
    'YOLOv11s-seg': {'P': 83.15, 'R': 85.22, 'mAP50': 82.4, 'mAP50-95': 64.12},
    'RT-DETR-L':    {'P': 80.44, 'R': 85.33, 'mAP50': 87.7, 'mAP50-95': 68.91},
}

detailed_etis = {
    'YOLOv8n-seg':  {'P': 41.11, 'R': 38.65, 'mAP50': 39.5, 'mAP50-95': 22.05},
    'YOLOv11s-seg': {'P': 76.54, 'R': 79.12, 'mAP50': 77.8, 'mAP50-95': 57.34},
    'RT-DETR-L':    {'P': 81.25, 'R': 84.60, 'mAP50': 87.9, 'mAP50-95': 68.11},
}


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Grouped Bar Chart — 100ep Flagship Models × 3 Datasets
# ═══════════════════════════════════════════════════════════════════════
def plot_grouped_bar():
    models = ['YOLOv8n-seg', 'YOLOv11s-seg', 'RT-DETR-L']
    datasets = ['Kvasir', 'CVC', 'ETIS']
    
    values = np.array([
        [95.4, 43.3, 39.5],   # v8n
        [95.8, 82.4, 77.8],   # v11s
        [96.3, 87.7, 87.9],   # RT-DETR
    ])

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ds in enumerate(datasets):
        bars = ax.bar(x + (i - 1) * width, values[:, i], width,
                      label=ds, color=list(DATASET_COLORS.values())[i],
                      edgecolor='white', linewidth=0.8, zorder=3)
        # Add value labels on top
        for bar, val in zip(bars, values[:, i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Mask mAP@0.5 (%)')
    ax.set_title('100-Epoch Flagship Models — Cross-Domain Generalization (Trained on Kvasir)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.legend(title='Test Dataset', loc='upper left')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.3, label='80% threshold')

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig1_grouped_bar_100ep.png'
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Line Chart — Performance Evolution Across Epochs
# ═══════════════════════════════════════════════════════════════════════
def plot_epoch_evolution():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    datasets = ['Kvasir', 'CVC', 'ETIS']
    titles = [
        'In-Domain (Kvasir → Kvasir)',
        'Cross-Domain (Kvasir → CVC)',
        'Extreme Cross-Domain (Kvasir → ETIS)',
    ]

    for idx, (ds, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx]
        for model in ['YOLOv8n-seg', 'YOLOv11s-seg']:
            data = evolution[model]
            ax.plot(data['epochs'], data[ds],
                    marker='o', linewidth=2.5, markersize=8,
                    color=COLORS[model], label=model, zorder=3)
            # Annotate last point
            ax.annotate(f"{data[ds][-1]:.1f}%",
                        xy=(data['epochs'][-1], data[ds][-1]),
                        xytext=(8, -5), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=COLORS[model])

        # RT-DETR only has 100ep, plot as a star marker
        rt_val = evolution['RT-DETR-L'][ds][0]
        ax.scatter([100], [rt_val], marker='*', s=200,
                   color=COLORS['RT-DETR-L'], zorder=4, label='RT-DETR-L')
        ax.annotate(f"{rt_val:.1f}%",
                    xy=(100, rt_val),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=COLORS['RT-DETR-L'])

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Training Epochs')
        ax.set_xticks([5, 50, 100])
        ax.grid(alpha=0.3, zorder=0)
        if idx == 0:
            ax.set_ylabel('Mask mAP@0.5 (%)')
            ax.legend(loc='lower right')

    axes[0].set_ylim(25, 102)
    plt.suptitle('Performance Evolution by Training Duration (Trained on Kvasir)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig2_epoch_evolution.png'
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Radar Chart — Multi-Metric Profile at 100ep
# ═══════════════════════════════════════════════════════════════════════
def plot_radar():
    categories = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             subplot_kw=dict(polar=True))
    
    dataset_details = [
        ('Kvasir (In-Domain)', detailed_kvasir),
        ('CVC (Cross-Domain)', detailed_cvc),
        ('ETIS (Extreme)', detailed_etis),
    ]

    models = ['YOLOv8n-seg', 'YOLOv11s-seg', 'RT-DETR-L']

    for ax_idx, (ds_name, ds_data) in enumerate(dataset_details):
        ax = axes[ax_idx]
        
        for model in models:
            vals = [ds_data[model][k] for k in ['P', 'R', 'mAP50', 'mAP50-95']]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=model, color=COLORS[model])
            ax.fill(angles, vals, alpha=0.1, color=COLORS[model])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7, color='gray')
        ax.set_title(ds_name, fontweight='bold', pad=20)
        
        if ax_idx == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)

    plt.suptitle('Multi-Metric Capability Profile at 100 Epochs (Trained on Kvasir)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig3_radar_profile.png'
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Generalization Drop Waterfall
# ═══════════════════════════════════════════════════════════════════════
def plot_generalization_drop():
    """Shows how much each model drops from in-domain to cross-domain."""
    models = ['YOLOv8n-seg', 'YOLOv11s-seg', 'RT-DETR-L']
    
    in_domain = [95.4, 95.8, 96.3]
    cvc_gen =   [43.3, 82.4, 87.7]
    etis_gen =  [39.5, 77.8, 87.9]
    
    drop_cvc  = [ind - gen for ind, gen in zip(in_domain, cvc_gen)]
    drop_etis = [ind - gen for ind, gen in zip(in_domain, etis_gen)]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, drop_cvc, width,
                   label='Kvasir → CVC Drop', color='#f59e0b', edgecolor='white', zorder=3)
    bars2 = ax.bar(x + width/2, drop_etis, width,
                   label='Kvasir → ETIS Drop', color='#ef4444', edgecolor='white', zorder=3)

    for bar, val in zip(bars1, drop_cvc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'-{val:.1f}pp', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#b45309')
    for bar, val in zip(bars2, drop_etis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'-{val:.1f}pp', ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#dc2626')

    ax.set_ylabel('Performance Drop (percentage points)')
    ax.set_title('Cross-Domain Generalization Gap at 100 Epochs (Trained on Kvasir)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, 65)

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig4_generalization_drop.png'
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating publication-quality figures...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    plot_grouped_bar()
    plot_epoch_evolution()
    plot_radar()
    plot_generalization_drop()
    
    print(f"\n✅ All 4 figures saved to: {OUTPUT_DIR}")
