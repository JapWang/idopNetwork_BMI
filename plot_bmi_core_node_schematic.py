"""Schematic of core-node transitions across BMI strata."""

import matplotlib.pyplot as plt

bmi_labels = ["Underweight", "Normal\nweight", "Overweight", "Obesity"]

core_nodes = [
    ["MCV", "Hb"],
    ["cr", "Hb"],
    ["RDW-SD", "uric"],
    ["RBC", "Hb", "uric"],
]

main_core = {
    0: ["MCV", "Hb"],
    1: ["cr", "Hb"],
    2: ["RDW-SD", "uric"],
    3: ["RBC", "Hb", "uric"],
}

x_coords = []
y_coords = []
labels = []
sizes = []

for i, nodes in enumerate(core_nodes):
    x = i
    n = len(nodes)
    ys = list(range(n))
    ys = [y - (n - 1) / 2 for y in ys]

    for node, y in zip(nodes, ys):
        x_coords.append(x)
        y_coords.append(y)
        labels.append(node)
        if node in main_core.get(i, []):
            sizes.append(1800)
        else:
            sizes.append(900)

fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")

ax.scatter(
    x_coords,
    y_coords,
    s=sizes,
    c="#FF6B6B",
    alpha=0.9,
    edgecolors="#34495E",
    linewidths=1.2,
)

for x, y, lab in zip(x_coords, y_coords, labels):
    ax.text(x, y, lab, ha="center", va="center", fontsize=10, color="black")

ax.set_xticks(range(len(bmi_labels)))
ax.set_xticklabels(bmi_labels, fontsize=11)
ax.set_yticks([])
ax.set_ylabel("Core nodes", fontsize=11)
ax.set_xlim(-0.7, len(bmi_labels) - 0.3)
ax.set_ylim(-2, 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_title(
    "Evolution of core nodes across BMI strata\n"
    "MCV/Hb → cr/Hb → RDW-SD/uric → Hb/RBC/uric",
    fontsize=12,
)

plt.tight_layout()
plt.savefig("results/core_nodes_bmi_flow.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved to results/core_nodes_bmi_flow.png")
