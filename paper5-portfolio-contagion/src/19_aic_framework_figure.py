"""
Paper 5 -- fig1_framework (pipeline overview)
=============================================
Visual abstract emphasising the automation flow: data ingestion ->
copula network -> risk metrics -> decision output, with labelled
inter-layer arrows naming the data object handed to each stage.

Output:
  results/figures/fig1_framework.{pdf,png}
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "../results/figures/"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 10,
    "axes.titlesize" : 12,
    "figure.dpi"     : 150,
})

fig, ax = plt.subplots(figsize=(15.5, 8))
ax.set_xlim(0, 17.2)
ax.set_ylim(0, 9)
ax.axis("off")

fig.suptitle("Automated Construction-Cost Contagion Pipeline:"
             " From Public Procurement Data to Per-Project Procurement Triggers",
             fontsize=13, fontweight="bold", y=0.97)

# ---- Layer rows ----
LAYERS = [
    {"y": 6.6, "fc": "#E3F2FD", "ec": "#1565C0",
     "title": "LAYER 1  --  AUTOMATED DATA INGESTION (monthly cron)",
     "boxes": [
         ("Diavgeia API\n(public procurement\nrecords, EL Δ.2.2)",      "0.4"),
         ("ELSTAT SPC23\n(material price\nindices)",                    "3.7"),
         ("FRED PPI\n(global commodity\nbenchmarks)",                    "7.0"),
         ("EU OpenTender\n+ World Bank IEG\n(external validation)",     "10.3"),
         ("Cross-country\npanel (13 EU/OECD\ncountries)",               "13.6"),
     ]},
    {"y": 4.6, "fc": "#FFF8E1", "ec": "#F57F17",
     "title": "LAYER 2  --  COPULA NETWORK + SYSTEM-VAR ANALYTICS",
     "boxes": [
         ("Rolling pair-Gumbel\nλ_U(t)\n(24-month windows)",            "0.4"),
         ("Source/Receiver\ndecomposition\n(NET = OUT − IN)",            "3.7"),
         ("Portfolio ES\n+ contingency\nallocation",                     "7.0"),
         ("LSTM ensemble\n(dual-target,\n5 seeds)",                     "10.3"),
         ("HMM regime\nclassification\n(2-state Gaussian)",             "13.6"),
     ]},
    {"y": 2.6, "fc": "#FFEBEE", "ec": "#C62828",
     "title": "LAYER 3  --  RISK METRICS + COMPOSITE INDEX",
     "boxes": [
         ("Contagion Index\n(network-mean λ_U)",                        "0.4"),
         ("Network density,\nspillover, role",                          "3.7"),
         ("ES_99(portfolio),\nDiv. benefit",                            "7.0"),
         ("CSRI\nz-score index",                                         "10.3"),
         ("Vine-conditional\nFEVD (3.11x)",                             "13.6"),
     ]},
    {"y": 0.6, "fc": "#E8F5E9", "ec": "#2E7D32",
     "title": "LAYER 4  --  AUTOMATED DECISION OUTPUT (per project, per month)",
     "boxes": [
         ("Per-project\ncontingency rule\n(2.5%–4.9%)",                "0.4"),
         ("Type-prioritised\nprocurement\n(roads first)",                "3.7"),
         ("Crisis alert\nif CSRI > +1σ\n(P(crisis) >= 0.5)",            "7.0"),
         ("Source-aware\nhedging signals\n(roads, pipelines)",          "10.3"),
         ("Project DB\nupdate +\nemail trigger",                        "13.6"),
     ]},
]

BOX_W = 2.6
BOX_H = 1.2

for layer in LAYERS:
    # Layer title bar
    ax.text(0.0, layer["y"] + BOX_H + 0.18, layer["title"],
            ha="left", va="bottom", fontsize=10, fontweight="bold",
            color=layer["ec"])
    # Boxes
    for body, x in layer["boxes"]:
        x = float(x)
        rect = FancyBboxPatch((x, layer["y"]), BOX_W, BOX_H,
                              boxstyle="round,pad=0.08",
                              facecolor=layer["fc"], edgecolor=layer["ec"],
                              linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + BOX_W / 2, layer["y"] + BOX_H / 2, body,
                ha="center", va="center", fontsize=8.2)

# ---- Big vertical arrows on right side, between layers,
# ----  each labelled with the data object handed to the next layer ----
FLOW_LABELS = [
    ((6.6, 5.8), "monthly type-level\ncost returns"),
    ((4.6, 3.8), "λ_U(t) network, NET flows,\nES, P(crisis)"),
    ((2.6, 1.8), "CSRI(t) + source/receiver\nroles"),
]
for (y_top, y_bot), lab in FLOW_LABELS:
    arr = FancyArrowPatch((16.7, y_top), (16.7, y_bot),
                          arrowstyle="-|>", mutation_scale=26,
                          lw=3.0, color="#37474F")
    ax.add_patch(arr)
    ax.text(16.5, (y_top + y_bot) / 2, lab,
            ha="right", va="center", fontsize=7.8, style="italic",
            color="#37474F")

# ---- Cycle-time annotation ----
ax.text(8.0, -0.55,
        "End-to-end cycle: ≤ 5 minutes per monthly portfolio update — "
        "deterministic, fully scripted, reproducible",
        ha="center", va="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#FFFDE7", edgecolor="#F57F17", linewidth=1.5))

# ---- Reproducibility tag ----
ax.text(0.0, -0.55,
        "github.com/dimitrioschronis/construction-risk-suite",
        ha="left", va="center", fontsize=8, style="italic",
        color="#37474F")

# ---- Output stat tag ----
ax.text(17.2, -0.55,
        "1,310 contracts  •  €4.51 B  •  Δ.2.2 awards 2014–2024",
        ha="right", va="center", fontsize=8, style="italic",
        color="#37474F")

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig1_framework.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig1_framework.pdf / .png  (AiC-style upgrade)")
