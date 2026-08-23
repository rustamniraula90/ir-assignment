import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

QUERIES = [
    "community nursing\nintervention",
    "physical activity\nolder adults",
    "health\ninequalities",
    "mental health\nyoung people",
    "bariatric\nsurgery",
]

TP = [5, 10, 7, 7, 10]
FP = [5, 0, 3, 3, 0]
FN = [129, 81, 16, 54, 15]
TN = [2057, 2105, 2170, 2132, 2171]

PRECISION = [0.50, 1.00, 0.70, 0.70, 1.00]
RECALL = [0.037, 0.110, 0.304, 0.115, 0.400]
F1 = [0.069, 0.198, 0.424, 0.197, 0.571]
P_AT_5 = [0.80, 1.00, 0.80, 1.00, 1.00]
AP = [0.925, 1.000, 0.826, 0.962, 1.000]

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)

matrix = np.array([TP, FP, FN, TN], dtype=float).T

fig, ax = plt.subplots(figsize=(6.5, 4.6))
im = ax.imshow(
    matrix, cmap="Blues",
    norm=SymLogNorm(linthresh=1, vmin=0, vmax=matrix.max()),
)
ax.set_xticks(range(4))
ax.set_xticklabels(["TP", "FP", "FN", "TN"])
ax.set_yticks(range(5))
ax.set_yticklabels(QUERIES, fontsize=8)
ax.set_title("Search engine confusion matrix per query\n(judged over the full 2,196-publication collection)")
for i in range(5):
    for j in range(4):
        value = int(matrix[i, j])
        ax.text(
            j, i, f"{value:,}", ha="center", va="center", fontsize=9,
            color="white" if value > 300 else "black",
        )
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Publications (log scale)")
fig.tight_layout()
savefig(fig, "se_confusion_heatmap.png")

metrics = {"Precision": PRECISION, "Recall": RECALL, "$F_1$": F1, "P@5": P_AT_5, "AP": AP}
x = np.arange(len(QUERIES))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 4.8))
for i, (name, values) in enumerate(metrics.items()):
    ax.bar(x + (i - 2) * width, values, width, label=name)
ax.set_xticks(x)
ax.set_xticklabels(QUERIES, fontsize=8)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.set_title("Search engine effectiveness per test query")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False)
fig.tight_layout()
savefig(fig, "se_effectiveness_bars.png")

print("\nDone.")
