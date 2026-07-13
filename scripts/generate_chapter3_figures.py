# -*- coding: utf-8 -*-
"""
Chapter 3 - I2V-CompBench Evaluation System Design
Generate 7 academic-style concept figures.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Force UTF-8 stdout for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============ Style Setup ============
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'Times New Roman', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

PALETTE = sns.color_palette("muted")
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN = PALETTE[:6]
GRAY = (0.55, 0.58, 0.62)
LIGHT_GRAY = (0.92, 0.93, 0.95)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '\u4e09\u9636\u6bb5\u5b9e\u73b0\u9700\u6c42', '\u8bba\u6587', 'figures', 'ch3')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  [OK] {name}")


# ================================================================
# Fig 3-1: Core Research Questions
# ================================================================
def fig3_1_core_questions():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("\u56fe3-1  \u6838\u5fc3\u7814\u7a76\u95ee\u9898\u903b\u8f91\u5173\u7cfb",
                 fontsize=14, fontweight='bold', pad=12)

    questions = [
        ("\u95ee\u9898\u4e00\nWhat", "\u8bc4\u4ec0\u4e48\uff1f", "\u7ec4\u5408\u80fd\u529b", BLUE),
        ("\u95ee\u9898\u4e8c\nHow", "\u600e\u4e48\u63a7\u5236\uff1f", "Preserve-Transform", ORANGE),
        ("\u95ee\u9898\u4e09\nJudge", "\u600e\u4e48\u5224\u5b9a\uff1f", "\u8bc1\u636e\u94fe", GREEN),
        ("\u95ee\u9898\u56db\nMeasure", "\u600e\u4e48\u91cf\u5316\uff1f", "\u6307\u6807\u4f53\u7cfb", PURPLE),
    ]

    box_w, box_h = 2.0, 2.8
    gap = 0.7
    start_x = 0.3

    for i, (title, q, ans, color) in enumerate(questions):
        x = start_x + i * (box_w + gap)
        y = 0.2
        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.1",
                             facecolor=(*color[:3], 0.12),
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + box_w/2, y + box_h - 0.5, title,
                ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        ax.text(x + box_w/2, y + box_h/2, q,
                ha='center', va='center', fontsize=12, color='#333333')
        ax.text(x + box_w/2, y + 0.5, f"\u2192 {ans}",
                ha='center', va='center', fontsize=10, color=GRAY, style='italic')
        if i < 3:
            ax.annotate('', xy=(x + box_w + gap - 0.05, y + box_h/2 + 0.2),
                        xytext=(x + box_w + 0.05, y + box_h/2 + 0.2),
                        arrowprops=dict(arrowstyle='->', lw=2, color=GRAY))

    save_fig(fig, 'fig3_1_core_questions.png')


# ================================================================
# Fig 3-2: Validity Risks Quadrant
# ================================================================
def fig3_2_validity_risks():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.set_title("\u56fe3-2  \u8bc4\u6d4b\u6709\u6548\u6027\u98ce\u9669\u56db\u8c61\u9650\u77e9\u9635",
                 fontsize=14, fontweight='bold', pad=16)

    quadrants = [
        (-1.15, 0.02, 1.1, 1.1, (*ORANGE[:3], 0.12), "\u6784\u5ff5\u504f\u79bb",
         "\u8bc4\u6d4b\u4e0d\u53cd\u6620\n\u771f\u5b9e\u80fd\u529b"),
        (0.05, 0.02, 1.1, 1.1, (*RED[:3], 0.12), "\u6d4b\u91cf\u4e0d\u53ef\u9760",
         "\u91cd\u590d\u6d4b\u91cf\n\u4e0d\u4e00\u81f4"),
        (-1.15, -1.12, 1.1, 1.1, (*BLUE[:3], 0.12), "\u5f52\u56e0\u6df7\u6742",
         "\u65e0\u6cd5\u5f52\u56e0\u5230\n\u6307\u5b9a\u7ef4\u5ea6"),
        (0.05, -1.12, 1.1, 1.1, (*PURPLE[:3], 0.12), "\u5206\u5e03\u4ee3\u8868\u6027\u4e0d\u8db3",
         "\u6837\u672c\u4e0d\u4ee3\u8868\n\u5b9e\u9645\u573a\u666f"),
    ]
    colors_edge = [ORANGE, RED, BLUE, PURPLE]

    for i, (x, y, w, h, fc, title, desc) in enumerate(quadrants):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                              facecolor=fc, edgecolor=colors_edge[i], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.7, title,
                ha='center', va='center', fontsize=13, fontweight='bold', color=colors_edge[i])
        ax.text(x + w/2, y + h*0.3, desc,
                ha='center', va='center', fontsize=10, color='#444444')

    # Axes
    ax.annotate('', xy=(1.2, -0.55), xytext=(-1.2, -0.55),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'))
    ax.annotate('', xy=(-0.55, 1.2), xytext=(-0.55, -1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'))
    ax.text(0.4, -0.62, "\u5916\u5728\u6709\u6548\u6027 \u2192", fontsize=10, ha='center', color='#333333')
    ax.text(-1.1, -0.62, "\u2190 \u5185\u5728\u6709\u6548\u6027", fontsize=10, ha='center', color='#333333')
    ax.text(-0.62, 0.7, "\u9ad8\n\u98ce\n\u9669", fontsize=9, ha='center', color='#333333')
    ax.text(-0.62, -0.8, "\u4f4e\n\u98ce\n\u9669", fontsize=9, ha='center', color='#333333')

    save_fig(fig, 'fig3_2_validity_risks.png')


# ================================================================
# Fig 3-3: Preserve-Transform Dual Axis
# ================================================================
def fig3_3_preserve_transform():
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Transform \u7a0b\u5ea6\uff08\u4f4e \u2192 \u9ad8\uff09", fontsize=12)
    ax.set_ylabel("Preserve \u7ea6\u675f\uff08\u5c11 \u2192 \u591a\uff09", fontsize=12)
    ax.set_title("\u56fe3-3  Preserve\u2013Transform \u53cc\u8f74\u6846\u67b6",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['\u4f4e', '\u4e2d', '\u9ad8'])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['\u5c11', '\u4e2d', '\u591a'])
    ax.grid(True, alpha=0.3)

    dims = [
        (0.45, 0.85, "Attribute\nBinding", BLUE),
        (0.78, 0.60, "Action\nBinding", ORANGE),
        (0.80, 0.45, "Motion\nBinding", GREEN),
        (0.50, 0.25, "Background\nDynamics", PURPLE),
        (0.90, 0.88, "View\nTransformation", RED),
    ]

    for x, y, label, color in dims:
        ax.scatter(x, y, s=200, color=color, zorder=5, edgecolors='white', linewidths=1.5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(12, -5),
                    fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor=color, linewidth=0.5))

    ax.text(0.05, -0.18,
            '\u6848\u4f8b: "\u7ea2\u8272\u8f7f\u8f66\u4ece\u9759\u6b62\u5230\u884c\u9a76" \u2014 '
            'Transform: \u8fd0\u52a8\u72b6\u6001\u53d8\u5316;  '
            'Preserve: \u8f66\u8eab\u989c\u8272\u3001\u80cc\u666f\u9053\u8def\u3001\u76f8\u673a\u4f4d\u7f6e',
            fontsize=8.5, color='#555555', style='italic')

    save_fig(fig, 'fig3_3_preserve_transform.png')


# ================================================================
# Fig 3-4: Five Dimensions Capability Matrix
# ================================================================
def fig3_4_five_dimensions():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("\u56fe3-4  \u4e94\u7ef4\u5ea6 \u00d7 C\u2080\u5206\u91cf \u5173\u6ce8\u5ea6\u77e9\u9635",
                 fontsize=14, fontweight='bold', pad=12)

    dims = ['Attribute\nBinding', 'Action\nBinding', 'Motion\nBinding',
            'Background\nDynamics', 'View\nTransformation']
    components = ['\u5b9e\u4f53 E', '\u5c5e\u6027 A', '\u5173\u7cfb R',
                  '\u72b6\u6001 S', '\u80cc\u666f B', '\u76f8\u673a K']

    # Attention matrix: 2=core, 1=secondary, 0=not involved
    data = np.array([
        [2, 2, 1, 1, 0, 0],
        [2, 1, 1, 2, 0, 0],
        [2, 0, 1, 2, 1, 1],
        [1, 0, 0, 1, 2, 0],
        [1, 0, 0, 0, 1, 2],
    ])

    cmap = plt.cm.Blues
    ax.imshow(data, cmap=cmap, aspect='auto', vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=11)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims, fontsize=10)

    labels_map = {2: "\u6838\u5fc3", 1: "\u6b21\u8981", 0: "\u2014"}
    for i in range(len(dims)):
        for j in range(len(components)):
            val = data[i, j]
            color = 'white' if val == 2 else '#333333'
            ax.text(j, i, labels_map[val], ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold' if val == 2 else 'normal')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0.9), label='\u6838\u5fc3\u5173\u6ce8'),
        Patch(facecolor=cmap(0.45), label='\u6b21\u8981\u5173\u6ce8'),
        Patch(facecolor=cmap(0.05), label='\u4e0d\u6d89\u53ca'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    save_fig(fig, 'fig3_4_five_dimensions.png')


# ================================================================
# Fig 3-5: Success/Failure Comparison
# ================================================================
def fig3_5_dimension_examples():
    fig, axes = plt.subplots(5, 2, figsize=(9, 11))
    fig.suptitle("\u56fe3-5  \u4e94\u7ef4\u5ea6\u6210\u529f / \u5931\u8d25\u5bf9\u6bd4\u793a\u610f",
                 fontsize=14, fontweight='bold', y=0.98)

    examples = [
        ("Attribute Binding",
         "\u7ea2\u8272\u8f7f\u8f66\u4fdd\u6301\u7ea2\u8272\n\u5c5e\u6027\u6b63\u786e\u7ed1\u5b9a\u76ee\u6807\u4e3b\u4f53",
         "\u7ea2\u8272\u53d8\u84dd\u8272\n\u5168\u5c40\u8272\u8c03\u6f02\u79fb\u6216\u9519\u8bef\u4e3b\u4f53\u53d8\u8272"),
        ("Action Binding",
         "\u4eba\u7269\u6b63\u786e\u6267\u884c\u8dd1\u6b65\u52a8\u4f5c\n\u59ff\u6001\u6301\u7eed\u53d8\u5316",
         "\u4eba\u7269\u9759\u6b62\u672a\u52a8\n\u76ee\u6807\u4e3b\u4f53\u6f0f\u6267\u884c"),
        ("Motion Binding",
         "\u6c7d\u8f66\u6cbf\u9053\u8def\u5411\u53f3\u884c\u9a76\n\u4e3b\u4f53\u4ea7\u751f\u771f\u5b9e\u4f4d\u79fb",
         "\u76f8\u673a\u5de6\u79fb\u9020\u6210\u76f8\u5bf9\u53f3\u79fb\n\u4ee3\u7406\u6ee1\u8db3"),
        ("Background Dynamics",
         "\u5929\u7a7a\u9010\u6e10\u51fa\u73b0\u4e4c\u4e91\n\u524d\u666f\u4e3b\u4f53\u4fdd\u6301\u4e0d\u53d8",
         "\u5168\u5c40\u753b\u9762\u53d8\u6697\n\u524d\u666f\u4eba\u7269\u540c\u65f6\u6539\u53d8"),
        ("View Transformation",
         "\u955c\u5934\u63a8\u8fd1\uff08dolly-in\uff09\n\u900f\u89c6\u5173\u7cfb\u6b63\u786e\u53d8\u5316",
         "\u4e8c\u7ef4\u7f29\u653e\u5192\u5145\u63a8\u8fd1\n\u65e0\u900f\u89c6\u53d8\u5316"),
    ]

    for i, (dim, success, failure) in enumerate(examples):
        # Success column
        ax_s = axes[i, 0]
        ax_s.set_xlim(0, 1)
        ax_s.set_ylim(0, 1)
        ax_s.axis('off')
        rect = FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.03",
                              facecolor=(*GREEN[:3], 0.08), edgecolor=GREEN, linewidth=2)
        ax_s.add_patch(rect)
        ax_s.text(0.5, 0.78, f"PASS  {dim}", ha='center', va='center',
                  fontsize=10, fontweight='bold', color=GREEN)
        ax_s.text(0.5, 0.4, success, ha='center', va='center', fontsize=9, color='#333333')

        # Failure column
        ax_f = axes[i, 1]
        ax_f.set_xlim(0, 1)
        ax_f.set_ylim(0, 1)
        ax_f.axis('off')
        rect = FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.03",
                              facecolor=(*RED[:3], 0.08), edgecolor=RED, linewidth=2)
        ax_f.add_patch(rect)
        ax_f.text(0.5, 0.78, f"FAIL  {dim}", ha='center', va='center',
                  fontsize=10, fontweight='bold', color=RED)
        ax_f.text(0.5, 0.4, failure, ha='center', va='center', fontsize=9, color='#333333')

    axes[0, 0].text(0.5, 1.1, "\u6210\u529f\uff08Success\uff09", ha='center', fontsize=12,
                    fontweight='bold', color=GREEN, transform=axes[0, 0].transAxes)
    axes[0, 1].text(0.5, 1.1, "\u5931\u8d25\uff08Failure\uff09", ha='center', fontsize=12,
                    fontweight='bold', color=RED, transform=axes[0, 1].transAxes)
    fig.subplots_adjust(hspace=0.15, wspace=0.08)
    save_fig(fig, 'fig3_5_dimension_examples.png')


# ================================================================
# Fig 3-6: Evidence Chain
# ================================================================
def fig3_6_evidence_chain():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("\u56fe3-6  \u8bc1\u636e\u94fe\u4e09\u6bb5\u5f0f\u7ed3\u6784",
                 fontsize=14, fontweight='bold', pad=12)

    stages = [
        ("\u521d\u59cb\u8bc1\u636e\n(Initial Evidence)",
         "\u56fe\u50cf\u5185\u5bb9\n\u2022 \u4e3b\u4f53\u8eab\u4efd\n\u2022 \u521d\u59cb\u5c5e\u6027\n\u2022 \u7a7a\u95f4\u5e03\u5c40",
         BLUE),
        ("\u6307\u4ee4\u8bc1\u636e\n(Instruction Evidence)",
         "Prompt\u8981\u6c42\n\u2022 \u76ee\u6807\u4e3b\u4f53\n\u2022 \u53d8\u5316\u7c7b\u578b\n\u2022 \u76ee\u6807\u53c2\u6570",
         ORANGE),
        ("\u7ed3\u679c\u8bc1\u636e\n(Result Evidence)",
         "\u89c6\u9891\u8f93\u51fa\n\u2022 \u53d8\u5316\u662f\u5426\u53d1\u751f\n\u2022 \u4fdd\u6301\u662f\u5426\u6ee1\u8db3\n\u2022 \u8fc7\u7a0b\u662f\u5426\u8fde\u7eed",
         GREEN),
    ]

    box_w, box_h = 2.8, 2.8
    gap = 0.6
    start_x = 0.5

    for i, (title, content, color) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        y = 0.3
        rect = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.08",
                              facecolor=(*color[:3], 0.10), edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + box_h - 0.35, f"Stage {i+1}",
                fontsize=8, color=color, fontweight='bold', alpha=0.7)
        ax.text(x + box_w/2, y + box_h - 0.7, title,
                ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        ax.text(x + box_w/2, y + box_h/2 - 0.4, content,
                ha='center', va='center', fontsize=9, color='#333333')
        if i < 2:
            ax.annotate('', xy=(x + box_w + gap - 0.1, y + box_h/2),
                        xytext=(x + box_w + 0.1, y + box_h/2),
                        arrowprops=dict(arrowstyle='->', lw=2.5, color=GRAY))

    ax.text(5.0, -0.8,
            "\u53ef\u5224\u5b9a\u6027\u6761\u4ef6\uff1a\u4e09\u6bb5\u8bc1\u636e\u5b8c\u6574 \u2192 "
            "\u5224\u65ad\u4e0d\u4f9d\u8d56\u8bc4\u6d4b\u5668\u7684\u989d\u5916\u731c\u6d4b",
            ha='center', va='center', fontsize=10, color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT_GRAY,
                      edgecolor=GRAY, linewidth=1))

    save_fig(fig, 'fig3_6_evidence_chain.png')


# ================================================================
# Fig 3-7: System Overview
# ================================================================
def fig3_7_system_overview():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-2, 5.5)
    ax.axis('off')
    ax.set_title("\u56fe3-7  I2V-CompBench \u7cfb\u7edf\u603b\u6846\u67b6",
                 fontsize=14, fontweight='bold', pad=14)

    stages = [
        ("\u771f\u5b9e\u89c6\u9891\n\u6570\u636e\u6e90", 0.3, BLUE),
        ("\u5148\u9a8c\u5206\u6790\n(Phase 1)", 2.5, BLUE),
        ("\u8bc4\u6d4b\u9898\u76ee\u6784\u5efa\n(Phase 2)", 5.0, ORANGE),
        ("\u8d28\u91cf\u63a7\u5236\n(Ch.5)", 7.5, GREEN),
        ("\u6a21\u578b\u8bc4\u6d4b\n(Phase 3)", 9.8, PURPLE),
        ("\u7ed3\u679c\u5206\u6790", 11.8, RED),
    ]

    box_w, box_h = 1.8, 1.8
    y_main = 1.5

    for label, x, color in stages:
        rect = FancyBboxPatch((x, y_main), box_w, box_h, boxstyle="round,pad=0.06",
                              facecolor=(*color[:3], 0.12), edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y_main + box_h/2, label,
                ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + box_w
        x2 = stages[i+1][1]
        ax.annotate('', xy=(x2, y_main + box_h/2),
                    xytext=(x1, y_main + box_h/2),
                    arrowprops=dict(arrowstyle='->', lw=2, color=GRAY))

    # Top: Core concept
    ax.text(6.5, 4.5,
            "\u6838\u5fc3\u6982\u5ff5: Preserve\u2013Transform \u53cc\u8f74\u6846\u67b6 + \u4e94\u7ef4\u7ec4\u5408\u80fd\u529b",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=(*BLUE[:3], 0.06),
                      edgecolor=BLUE, linewidth=1.5, linestyle='--'))
    ax.annotate('', xy=(6.5, y_main + box_h + 0.05), xytext=(6.5, 4.0),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=BLUE, linestyle='dashed'))

    # Bottom: Output
    ax.text(6.5, -0.8,
            "\u8f93\u51fa: 1500\u6761\u7ed3\u6784\u5316Benchmark\u6837\u672c (5\u7ef4 \u00d7 300\u6761)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=(*GREEN[:3], 0.06),
                      edgecolor=GREEN, linewidth=1.5, linestyle='--'))
    ax.annotate('', xy=(6.5, -0.3), xytext=(6.5, y_main - 0.05),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=GREEN, linestyle='dashed'))

    # Chapter labels
    for x, y, txt, c in [(1.5, y_main-0.4, "Ch.3-4", BLUE),
                          (5.9, y_main-0.4, "Ch.4", ORANGE),
                          (8.4, y_main-0.4, "Ch.5", GREEN),
                          (10.7, y_main-0.4, "Ch.6", PURPLE)]:
        ax.text(x, y, txt, ha='center', fontsize=8, color=c, alpha=0.7)

    save_fig(fig, 'fig3_7_system_overview.png')


# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Chapter 3 figures generation started")
    print("=" * 60)

    fig3_1_core_questions()
    fig3_2_validity_risks()
    fig3_3_preserve_transform()
    fig3_4_five_dimensions()
    fig3_5_dimension_examples()
    fig3_6_evidence_chain()
    fig3_7_system_overview()

    print("=" * 60)
    print(f"  All 7 figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
