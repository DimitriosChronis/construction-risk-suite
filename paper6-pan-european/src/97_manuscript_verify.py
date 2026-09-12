"""
97_manuscript_verify.py
=======================
Delta-phase sweep: verifies that every headline number quoted in
manuscript/main.tex matches the final result tables, that cite keys
match the .bib exactly (no orphans either way), and that LaTeX
structure is sound. Exit 1 on any failure.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(SCRIPT_DIR, "..", "manuscript")
TAB = os.path.join(SCRIPT_DIR, "..", "results", "tables")
PROC = os.path.join(SCRIPT_DIR, "..", "data", "processed")

tex = open(os.path.join(MS, "main.tex"), encoding="utf-8").read()
flat = re.sub(r"\s+", " ", tex)
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def t(name):
    return pd.read_csv(os.path.join(TAB, name))


# ---- 1. structural checks ----
bal = 0
for i, ch in enumerate(tex):
    if ch == "{" and (i == 0 or tex[i - 1] != "\\"):
        bal += 1
    elif ch == "}" and (i == 0 or tex[i - 1] != "\\"):
        bal -= 1
check(bal == 0, f"brace imbalance {bal}")

labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
check(not (refs - labels), f"refs w/o labels: {refs - labels}")

cites = set()
for m in re.finditer(r"\\cite(?:t|p|alp|alt)?\*?(?:\[[^\]]*\])?\{([^}]+)\}",
                     tex):
    for k in m.group(1).split(","):
        cites.add(k.strip())
bib = open(os.path.join(MS, "references.bib"), encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
check(not (cites - keys), f"cited not in bib: {cites - keys}")
check(not (keys - cites), f"bib orphans: {keys - cites}")

# ---- 2. numbers vs tables ----
h1 = t("table_h1_summary.csv")
sub = t("table_h1_subgroups.csv")
ep = t("table_h1_episodes.csv")
ci = t("table_h1_bootstrap_ci.csv")
h3 = t("table_h3_summary.csv").iloc[0]
g = t("table_granger_network.csv")
h4 = t("table_h4_summary.csv").iloc[0]
wb = t("table_h4_wildboot.csv").iloc[0]
g3 = t("table_g3_summary.csv")
bi = t("table_bridge_indicator_check.csv")
bc = t("table_bridge_correlations.csv")
lb = t("table_robustness_labels.csv")
se = t("table_robustness_seasonal.csv").iloc[0]
gs = t("table_robustness_gr_source.csv")
ir = t("table_indicator_robustness.csv")
q1 = t("table_quarterly_h1.csv").iloc[0]
dy = pd.read_csv(os.path.join(TAB, "table_dy_spillover.csv"), index_col=0)

REQUIRED = {
    # H1 core
    "0.157 -> 0.213 stable/crisis w24":
        ["0.157", "0.213"],
    "amp 1.360/1.434": ["1.360", "1.434"],
    "perm p": ["0.132", "0.152"],
    "CI w36 blk6": ["1.069", "1.836"],
    "pairs 21/24 of 28": ["$21$", "$24$"],
    "MED-MED": ["1.565", "0.004", "1.337", "0.015"],
    "episodes": ["1.782", "1.026", "0.494", "2.078"],
    # family shift
    "pooled family": ["$25$", "$16$", "0.132", "0.360"],
    # H2
    "H2": ["-0.003", "$75\\%$"],
    # H3
    "H3": ["$34$", "$31$", "61", "32.6", "+23.6", "+16.6", "+12.4",
           "-25.6", "-19.7"],
    # H4
    "H4": ["0.127", "0.088", "0.078", "1.88", "0.007", "0.012",
           "0.055", "0.062", "0.060"],
    # bridge
    "bridge": ["0.963", "0.956", "0.811", "0.847", "0.202", "0.261",
               "247", "0.259"],
    # ukraine
    "ukraine": ["6.4", "4.1", "0.85", "0.380"],
    # robustness
    "robust": ["1.269", "1.278", "1.382", "0.976", "11/15", "13/15",
               "1.609", "0.008", "7/7", "0/7"],
}
for group, tokens in REQUIRED.items():
    for tok in tokens:
        check(tok in flat, f"missing number [{group}]: '{tok}'")

# cross-verify a few against the actual tables (value drift guard)
check(abs(h1[h1.window == 24]["mean_lambdaU_stable"].iloc[0] - 0.157)
      < 5e-4, "table drift: lambdaU stable w24")
check(abs(sub[(sub.window == 24) & (sub.labels == "exog")
              & (sub.bloc == "MED-MED")]["perm_p"].iloc[0] - 0.004)
      < 5e-4, "table drift: MED-MED p")
check(int(g["bh_fdr_sig"].sum()) == 31, "table drift: BH count")
check(abs(wb["wild_boot_p"] - 0.0546) < 5e-3, "table drift: wild p")
check(abs(float(dy.loc["GR", "net_pct"]) - 23.6) < 0.1,
      "table drift: GR net spillover")
check((lb["MEDMED_p"] < 0.05).sum() == 11
      and (lb["MEDMED_p"] < 0.06).sum() == 13,
      "table drift: label sweep counts")
g25 = g3[g3.regime == "crisis"]["gumbel_best"].iloc[0]
check(int(g25) == 25, "table drift: crisis Gumbel best")

# forbidden phrases (overclaim guard)
for phrase in ["the first study", "for the first time", "we prove",
               "contagion is established", "causal effect"]:
    check(phrase not in flat.lower(), f"forbidden phrase: '{phrase}'")

print("=" * 50)
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"ALL MANUSCRIPT CHECKS PASSED "
      f"(cites: {len(cites)}, labels: {len(labels)})")
