"""
Paper 5 -- QW3: Upgraded AiC-style fig1_framework
==================================================
Visual abstract emphasising AUTOMATION FLOW: data ingestion ->
copula network -> risk metrics -> decision output. Tailored to
AiC editor's first-glance review (the figure that follows the
abstract).

Output:
  results/figures/fig1_framework.{pdf,png}    (replaces v1)
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

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 16)
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
         ("Cross-country\npanel (17 EU/OECD\ncountries)",               "13.6"),
     ]},
    {"y": 4.6, "fc": "#FFF8E1", "ec": "#F57F17",
     "title": "LAYER 2  --  COPULA NETWORK + SYSTEM-VAR ANALYTICS",
     "boxes": [
         ("Rolling pair-Gumbel\nλ_U(t)\n(36-month windows)",            "0.4"),
         ("Source/Receiver\ndecomposition\n(NET = OUT − IN)",            "3.7"),
         ("Portfolio ES\n+ contingency\nallocation",                     "7.0"),
         ("LSTM dual-target\nensemble\n(P4 architecture)",              "10.3"),
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

# ---- Big vertical arrows on right side, between layers ----
for y_top, y_bot in [(6.6, 5.8), (4.6, 3.8), (2.6, 1.8)]:
    arr = FancyArrowPatch((15.4, y_top), (15.4, y_bot),
                          arrowstyle="->", mutation_scale=22,
                          lw=2.5, color="#37474F")
    ax.add_patch(arr)

# ---- Cycle-time annotation ----
ax.text(8.0, -0.55,
        "End-to-end cycle: ≤ 5 minutes per portfolio update — "
        "deterministic, reproducible, no human in the loop",
        ha="center", va="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#FFFDE7", edgecolor="#F57F17", linewidth=1.5))

# ---- Reproducibility tag ----
ax.text(0.0, -0.55,
        "github.com/dimitrioschronis/construction-risk-suite",
        ha="left", va="center", fontsize=8, style="italic",
        color="#37474F")

# ---- Output stat tag ----
ax.text(16.0, -0.55,
        "1,310 projects  •  €4.5 B  •  276 months  •  29 references",
        ha="right", va="center", fontsize=8, style="italic",
        color="#37474F")

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig1_framework.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig1_framework.pdf / .png  (AiC-style upgrade)")
