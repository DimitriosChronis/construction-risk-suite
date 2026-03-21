"""
Module: 09_vine_copula.py
Description: 4-dimensional Vine Copula using pyvinecopulib.
             Replaces the simple Marshall-Olkin Gumbel with a proper R-vine
             that allows different bivariate copulas on each edge (pair-copula).
             Outputs: fitted vine structure, simulation comparison, publication figure.
Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
from scipy.stats import norm, kendalltau
import warnings
warnings.filterwarnings('ignore')

try:
    import pyvinecopulib as pv
    VINE_AVAILABLE = True
except ImportError:
    VINE_AVAILABLE = False
    print("[WARN] pyvinecopulib not installed. Run: pip install pyvinecopulib")
    print("       Falling back to manual C-vine with scipy copulas.")


# --- 1. CONFIGURATION ---
@dataclass
class VineConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    TABLES_DIR: Path = BASE_DIR / 'results' / 'tables'
    FIGURES_DIR: Path = BASE_DIR / 'results' / 'figures'

    NUM_SIMS: int = 20_000
    BASE_COST: float = 2_300_000.0
    PROJECT_MONTHS: int = 24
    SEED: int = 42

    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'Concrete': 0.30, 'Steel': 0.30, 'Fuel_Energy': 0.20, 'PVC_Pipes': 0.20
    })

    # Stable / Crisis split for parameter reporting
    STABLE_YEARS: tuple = (2000, 2007)
    CRISIS_YEARS: tuple = (2008, 2013)


# --- 2. PSEUDO-OBS UTIL ---
def pseudo_obs(data: np.ndarray) -> np.ndarray:
    n, d = data.shape
    u = np.zeros_like(data)
    for j in range(d):
        ranks = np.argsort(np.argsort(data[:, j])) + 1
        u[:, j] = ranks / (n + 1)
    return u


# --- 3. FALLBACK: MANUAL C-VINE (no pyvinecopulib) ---
class ManualCVine:
    """
    Simplified C-vine with Gumbel pair copulas (upper-tail dependence).
    Fitted via Kendall-tau inversion on each bivariate margin.
    """
    def __init__(self, d: int = 4, seed: int = 42):
        self.d = d
        self.rng = np.random.default_rng(seed)
        self.thetas: Dict = {}   # (i,j) -> theta

    def fit(self, u: np.ndarray):
        d = u.shape[1]
        # Tree 1: all pairs sharing variable 0 as root (C-vine)
        for j in range(1, d):
            t, _ = kendalltau(u[:, 0], u[:, j])
            t = np.clip(t, 0.01, 0.99)
            self.thetas[(0, j)] = max(1.0001, 1.0 / (1.0 - t))

        # Tree 2: conditioned pairs
        if d >= 3:
            v01 = self._gumbel_cdf_conditional(u[:, 0], u[:, 1], self.thetas[(0, 1)])
            v02 = self._gumbel_cdf_conditional(u[:, 0], u[:, 2], self.thetas[(0, 2)])
            t, _ = kendalltau(v01, v02)
            t = np.clip(t, 0.01, 0.99)
            self.thetas[('cond', 1, 2)] = max(1.0001, 1.0 / (1.0 - t))

        if d >= 4:
            v03 = self._gumbel_cdf_conditional(u[:, 0], u[:, 3], self.thetas[(0, 3)])
            t, _ = kendalltau(v02, v03)
            t = np.clip(t, 0.01, 0.99)
            self.thetas[('cond', 2, 3)] = max(1.0001, 1.0 / (1.0 - t))

        return self

    @staticmethod
    def _gumbel_cdf_conditional(u1: np.ndarray, u2: np.ndarray,
                                 theta: float) -> np.ndarray:
        """h-function: C(u2|u1) for Gumbel copula (numerical)."""
        u1 = np.clip(u1, 1e-9, 1 - 1e-9)
        u2 = np.clip(u2, 1e-9, 1 - 1e-9)
        l1 = (-np.log(u1)) ** theta
        l2 = (-np.log(u2)) ** theta
        S = l1 + l2
        log_C = -S ** (1.0 / theta)
        # dC/du1 via chain rule
        dlog_C_dS = (1.0 / theta) * S ** (1.0 / theta - 1.0) * (-1.0)
        dS_dl1 = 1.0
        dl1_du1 = -theta * (-np.log(u1)) ** (theta - 1.0) / u1 * (-1.0)
        cond = np.exp(log_C) * dlog_C_dS * dS_dl1 * dl1_du1
        return np.clip(cond, 1e-9, 1 - 1e-9)

    @staticmethod
    def _gumbel_quantile_conditional(p: np.ndarray, u1: np.ndarray,
                                      theta: float, tol: float = 1e-6) -> np.ndarray:
        """Inverse h-function via bisection."""
        lo = np.full_like(p, 1e-9)
        hi = np.ones_like(p) * (1 - 1e-9)
        for _ in range(50):
            mid = (lo + hi) / 2.0
            val = ManualCVine._gumbel_cdf_conditional(u1, mid, theta)
            lo = np.where(val < p, mid, lo)
            hi = np.where(val >= p, mid, hi)
        return (lo + hi) / 2.0

    def simulate(self, n: int) -> np.ndarray:
        """Rosenblatt sequential sampling for C-vine."""
        d = self.d
        rng = self.rng
        u = np.zeros((n, d))
        w = rng.uniform(0, 1, (n, d))

        u[:, 0] = w[:, 0]
        u[:, 1] = self._gumbel_quantile_conditional(
            w[:, 1], u[:, 0], self.thetas[(0, 1)]
        )

        if d >= 3:
            # v1|0 and invert tree-2 pair
            v01 = self._gumbel_cdf_conditional(u[:, 0], u[:, 1], self.thetas[(0, 1)])
            v02_star = self._gumbel_quantile_conditional(
                w[:, 2], v01, self.thetas[('cond', 1, 2)]
            )
            u[:, 2] = self._gumbel_quantile_conditional(
                v02_star, u[:, 0], self.thetas[(0, 2)]
            )

        if d >= 4:
            v02 = self._gumbel_cdf_conditional(u[:, 0], u[:, 2], self.thetas[(0, 2)])
            v03_star = self._gumbel_quantile_conditional(
                w[:, 3], v02, self.thetas[('cond', 2, 3)]
            )
            u[:, 3] = self._gumbel_quantile_conditional(
                v03_star, u[:, 0], self.thetas[(0, 3)]
            )

        return u


# --- 4. VINE COPULA ENGINE ---
class VineCopulaEngine:
    def __init__(self, cfg: VineConfig):
        self.cfg = cfg
        self.cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(cfg.SEED)

    def run(self):
        df = pd.read_csv(self.cfg.DATA_PATH, index_col='Date', parse_dates=True)
        returns_full = np.log(df / df.shift(1)).dropna()

        cols = ['Concrete', 'Steel', 'Fuel_Energy', 'PVC_Pipes']
        returns_full = returns_full[cols]

        # ---- Fit on full sample ----
        u_full = pseudo_obs(returns_full.values)

        print("\n" + "="*65)
        print("4-DIMENSIONAL VINE COPULA ANALYSIS")
        print("="*65)

        vine_model = None
        structure_rows = []

        if VINE_AVAILABLE:
            vine_model, structure_rows = self._fit_pyvine(u_full, cols)
        else:
            vine_model, structure_rows = self._fit_manual_vine(u_full, cols)

        # ---- Simulate ----
        print("\n[Simulating 4D Vine Copula...]")
        u_vine = self._simulate(vine_model, self.cfg.NUM_SIMS)

        # ---- Compute project costs ----
        MAX_VOL = 0.15
        scale = np.sqrt(self.cfg.PROJECT_MONTHS)
        margins = {}
        for col in cols:
            _, std = norm.fit(returns_full[col])
            margins[col] = (0.0, min(std, MAX_VOL) * scale)

        # Also simulate Gumbel (for comparison)
        avg_tau = np.mean([
            kendalltau(u_full[:, i], u_full[:, j])[0]
            for i in range(4) for j in range(i + 1, 4)
        ])
        theta = max(1.0001, 1.0 / (1.0 - np.clip(avg_tau, 0.01, 0.99)))
        alpha = 1.0 / theta
        rng2 = np.random.default_rng(self.cfg.SEED + 1)
        V_g = self._stable(alpha, self.cfg.NUM_SIMS, rng2)[:, None]
        Y_g = rng2.uniform(0, 1, (self.cfg.NUM_SIMS, 4))
        u_gumbel = np.exp(- (-np.log(np.clip(Y_g, 1e-9, 1-1e-9)) / V_g) ** alpha)

        costs_vine   = self._calc_cost(u_vine,   cols, margins)
        costs_gumbel = self._calc_cost(u_gumbel, cols, margins)

        # ---- P-statistics comparison ----
        percs = [50, 75, 85, 90, 95, 99]
        comp_rows = []
        for p in percs:
            cv = np.percentile(costs_vine, p)
            cg = np.percentile(costs_gumbel, p)
            comp_rows.append({
                'Percentile': f'P{p}',
                'Vine Copula (EUR)': round(cv, 0),
                'Gumbel Copula (EUR)': round(cg, 0),
                'Delta (EUR)': round(cv - cg, 0),
                'Delta (%)': round((cv - cg) / cg * 100, 3)
            })

        df_comp = pd.DataFrame(comp_rows)
        print("\nVINE vs GUMBEL COST COMPARISON:")
        print(df_comp.to_string(index=False))

        # ---- Regime-specific fitting ----
        regime_rows = self._regime_fit(returns_full, cols)

        # ---- Save outputs ----
        df_comp.to_csv(self.cfg.TABLES_DIR / 'vine_vs_gumbel.csv', index=False)
        pd.DataFrame(structure_rows).to_csv(
            self.cfg.TABLES_DIR / 'vine_structure.csv', index=False
        )
        pd.DataFrame(regime_rows).to_csv(
            self.cfg.TABLES_DIR / 'vine_regime_params.csv', index=False
        )
        print("[OK] CSVs saved -> results/tables/vine_*.csv")

        # ---- Figures ----
        self._plot_comparison(costs_vine, costs_gumbel, df_comp)
        self._export_latex(df_comp, structure_rows, regime_rows)

        return costs_vine, costs_gumbel, df_comp

    # ------------------------------------------------------------------
    def _fit_pyvine(self, u: np.ndarray, cols):
        print("[pyvinecopulib] Fitting R-vine with automatic structure selection...")
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.gumbel,
                        pv.BicopFamily.clayton, pv.BicopFamily.frank,
                        pv.BicopFamily.joe, pv.BicopFamily.bb1],
            selection_criterion='aic',
            num_threads=1
        )
        # pyvinecopulib 0.7.x API: create empty vine, then select
        d = u.shape[1]
        vine = pv.Vinecop(d)
        # Data must be Fortran-contiguous float64
        u_fort = np.asfortranarray(u.astype(np.float64))
        vine.select(u_fort, controls)
        print(vine)

        # Extract structure summary
        d = u.shape[1]
        rows = []
        pair_copulas = vine.pair_copulas
        mat = vine.matrix
        for tree in range(d - 1):
            for edge in range(d - tree - 1):
                pc = pair_copulas[tree][edge]
                # Safe column-name lookup (1-based matrix indices)
                try:
                    c1 = cols[int(mat[edge, d - 1]) - 1]
                    c2 = cols[int(mat[tree + edge + 1, d - 1]) - 1]
                    pair_name = f"{c1} - {c2}"
                except (IndexError, ValueError):
                    pair_name = f"edge ({tree+1},{edge+1})"
                rows.append({
                    'Tree': tree + 1,
                    'Edge': edge + 1,
                    'Pair': pair_name,
                    'Family': str(pc.family),
                    'Parameters': str([round(float(p), 4) for p in pc.parameters.flatten()]),
                    'tau': round(float(pc.tau), 4),
                })

        return vine, rows

    def _fit_manual_vine(self, u: np.ndarray, cols):
        print("[Manual C-vine] Fitting Gumbel-based C-vine (pyvinecopulib unavailable)...")
        vine = ManualCVine(d=4, seed=self.cfg.SEED)
        vine.fit(u)

        rows = []
        for key, theta in vine.thetas.items():
            tau = 1.0 - 1.0 / theta
            rows.append({
                'Tree': 'C-vine',
                'Edge': str(key),
                'Family': 'Gumbel',
                'theta': round(theta, 4),
                'tau': round(tau, 4),
            })

        print("C-vine parameters:")
        for r in rows:
            print(f"  Edge {r['Edge']}: theta={r['theta']:.4f}, tau={r['tau']:.4f}")

        return vine, rows

    def _simulate(self, vine_model, n: int) -> np.ndarray:
        if VINE_AVAILABLE and isinstance(vine_model, pv.Vinecop):
            try:
                samples = vine_model.simulate(n, seeds=[self.cfg.SEED])
            except TypeError:
                # Older API without seeds kwarg
                samples = vine_model.simulate(n)
            return np.array(samples)
        else:
            return vine_model.simulate(n)

    def _calc_cost(self, u_mat: np.ndarray, cols, margins) -> np.ndarray:
        u_mat = np.clip(u_mat, 1e-6, 1 - 1e-6)
        impact = np.zeros(u_mat.shape[0])
        for i, col in enumerate(cols):
            if col in self.cfg.WEIGHTS:
                mu, std = margins[col]
                ret = norm.ppf(u_mat[:, i], loc=mu, scale=std)
                impact += self.cfg.WEIGHTS[col] * (np.exp(ret) - 1)
        return self.cfg.BASE_COST * (1 + impact)

    @staticmethod
    def _stable(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
        if alpha >= 0.99:
            return np.ones(size)
        U = rng.uniform(1e-4, np.pi - 1e-4, size)
        W = rng.exponential(1.0, size) + 1e-4
        t1 = np.sin(alpha * U) / (np.sin(U) ** (1 / alpha))
        t2 = (np.sin((1 - alpha) * U) / W) ** ((1 - alpha) / alpha)
        return np.clip(t1 * t2, 1e-9, 1e5)

    def _regime_fit(self, returns_full, cols):
        """Fit vine parameters separately for stable and crisis regimes."""
        rows = []
        regimes = {
            'Stable': self.cfg.STABLE_YEARS,
            'Crisis': self.cfg.CRISIS_YEARS,
        }
        for regime, (yr_s, yr_e) in regimes.items():
            sub = returns_full[
                (returns_full.index.year >= yr_s) &
                (returns_full.index.year <= yr_e)
            ]
            if len(sub) < 12:
                continue
            u_sub = pseudo_obs(sub.values)

            if VINE_AVAILABLE:
                controls = pv.FitControlsVinecop(
                    family_set=[pv.BicopFamily.gumbel, pv.BicopFamily.gaussian,
                                pv.BicopFamily.clayton],
                    selection_criterion='aic', num_threads=1
                )
                d_r = u_sub.shape[1]
                vine_r = pv.Vinecop(d_r)
                vine_r.select(np.asfortranarray(u_sub.astype(np.float64)), controls)
                # Summary tau across all pair copulas in tree 1 and tree 2
                n_trees = d_r - 1
                tau_vals = []
                for t in range(n_trees):
                    for e in range(d_r - t - 1):
                        try:
                            tau_vals.append(float(vine_r.pair_copulas[t][e].tau))
                        except Exception:
                            pass
                avg_tau = float(np.mean(tau_vals)) if tau_vals else 0.0
                rows.append({
                    'Regime': f'{regime} ({yr_s}-{yr_e})',
                    'n': len(sub),
                    'Avg pair tau (vine)': round(avg_tau, 4),
                    'Method': 'pyvinecopulib R-vine'
                })
            else:
                # Fallback: avg Kendall tau
                taus = [
                    kendalltau(u_sub[:, i], u_sub[:, j])[0]
                    for i in range(4) for j in range(i + 1, 4)
                ]
                rows.append({
                    'Regime': f'{regime} ({yr_s}-{yr_e})',
                    'n': len(sub),
                    'Avg pair tau (vine)': round(np.mean(taus), 4),
                    'Method': 'Manual C-vine (Gumbel)'
                })
        return rows

    # ------------------------------------------------------------------
    def _plot_comparison(self, costs_vine, costs_gumbel, df_comp):
        """
        Publication figure: CDF comparison (Vine vs Gumbel) + bar chart of P85/P95 gap.
        """
        fig = plt.figure(figsize=(7.2, 5.5))
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               hspace=0.42, wspace=0.38,
                               left=0.10, right=0.97, top=0.91, bottom=0.10)

        ax_cdf = fig.add_subplot(gs[0, :])
        ax_bar = fig.add_subplot(gs[1, 0])
        ax_diff = fig.add_subplot(gs[1, 1])

        # ---- CDF panel ----
        for costs, color, lbl, ls in [
            (costs_gumbel, '#D55E00', 'Marshall-Olkin Gumbel', '--'),
            (costs_vine,   '#0072B2', '4-D Vine Copula',       '-'),
        ]:
            s = np.sort(costs)
            p = np.arange(1, len(s) + 1) / len(s)
            ax_cdf.plot(s, p, color=color, linewidth=1.8, linestyle=ls, label=lbl)

        ax_cdf.axhline(0.85, color='black', linewidth=0.9, linestyle=':', alpha=0.7)
        ax_cdf.axhline(0.95, color='black', linewidth=0.9, linestyle=':', alpha=0.7)
        ax_cdf.text(np.percentile(costs_vine, 0.5), 0.854, 'P85',
                    fontsize=7, color='#333333')
        ax_cdf.text(np.percentile(costs_vine, 0.5), 0.954, 'P95',
                    fontsize=7, color='#333333')
        ax_cdf.set_xlabel('Project Cost (EUR)', fontsize=8)
        ax_cdf.set_ylabel('Cumulative Probability', fontsize=8)
        ax_cdf.set_title('(a) Cost CDF: Vine Copula vs. Marshall-Olkin Gumbel',
                         fontsize=8, fontweight='bold')
        ax_cdf.legend(fontsize=7, frameon=True)
        ax_cdf.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x/1e6:.2f}M')
        )
        ax_cdf.grid(True, alpha=0.3, linewidth=0.4)
        ax_cdf.tick_params(labelsize=7)

        # ---- Bar chart: P85/P95/P99 ----
        sub = df_comp[df_comp['Percentile'].isin(['P85', 'P95', 'P99'])]
        x = np.arange(len(sub))
        w = 0.33
        ax_bar.bar(x - w/2, sub['Gumbel Copula (EUR)'].values / 1e6,
                   width=w, color='#D55E00', alpha=0.8, label='Gumbel')
        ax_bar.bar(x + w/2, sub['Vine Copula (EUR)'].values / 1e6,
                   width=w, color='#0072B2', alpha=0.8, label='Vine')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(sub['Percentile'].values, fontsize=8)
        ax_bar.set_ylabel('Cost (EUR M)', fontsize=8)
        ax_bar.set_title('(b) Cost Quantiles Comparison', fontsize=8, fontweight='bold')
        ax_bar.legend(fontsize=7)
        ax_bar.tick_params(labelsize=7)
        ax_bar.grid(True, alpha=0.3, linewidth=0.4, axis='y')

        # ---- Delta (%) panel ----
        sub2 = df_comp.copy()
        colors_delta = ['#CC0000' if d > 0 else '#009E73'
                        for d in sub2['Delta (%)'].values]
        ax_diff.bar(range(len(sub2)), sub2['Delta (%)'].values,
                    color=colors_delta, alpha=0.85)
        ax_diff.axhline(0, color='black', linewidth=0.8)
        ax_diff.set_xticks(range(len(sub2)))
        ax_diff.set_xticklabels(sub2['Percentile'].values, fontsize=7, rotation=30)
        ax_diff.set_ylabel('Vine - Gumbel (%)', fontsize=8)
        ax_diff.set_title('(c) Relative Difference (Vine vs. Gumbel)',
                          fontsize=8, fontweight='bold')
        ax_diff.tick_params(labelsize=7)
        ax_diff.grid(True, alpha=0.3, linewidth=0.4, axis='y')

        out = self.cfg.FIGURES_DIR / 'fig_vine_copula.png'
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[OK] Figure saved -> {out}")

    def _export_latex(self, df_comp, structure_rows, regime_rows):
        """LaTeX table: Vine vs Gumbel cost percentiles."""
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{4-D Vine Copula vs.\ Marshall-Olkin Gumbel: "
                     r"Simulated Cost Percentiles (Base Cost = \EUR{2.3M})}")
        lines.append(r"\label{tab:vine_comparison}")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Percentile & Vine (EUR) & Gumbel (EUR) & $\Delta$ (EUR) & $\Delta$ (\%) \\")
        lines.append(r"\midrule")
        for _, r in df_comp.iterrows():
            bold_open  = r"\textbf{" if r['Percentile'] in ('P85', 'P95') else ""
            bold_close = "}" if r['Percentile'] in ('P85', 'P95') else ""
            lines.append(
                f"{bold_open}{r['Percentile']}{bold_close} & "
                f"{bold_open}{r['Vine Copula (EUR)']:,.0f}{bold_close} & "
                f"{bold_open}{r['Gumbel Copula (EUR)']:,.0f}{bold_close} & "
                f"{r['Delta (EUR)']:+,.0f} & "
                f"{r['Delta (%)']:+.3f}\\% \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(
            r"\multicolumn{5}{l}{\footnotesize \textit{Note: Vine fitted via AIC selection "
            r"(pyvinecopulib). 20{,}000 simulations, 24-month horizon. "
            r"Bold = key risk thresholds.}} \\"
        )
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        # Regime table
        if regime_rows:
            lines.append("\n\n")
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
            lines.append(r"\caption{4-D Vine Copula: Average Pair-Copula Kendall's $\tau$ by Regime}")
            lines.append(r"\label{tab:vine_regime}")
            lines.append(r"\begin{tabular}{lrrl}")
            lines.append(r"\toprule")
            lines.append(r"Regime & $n$ & Avg.\ pair $\tau$ & Method \\")
            lines.append(r"\midrule")
            for r in regime_rows:
                lines.append(
                    f"{r['Regime']} & {r['n']} & {r['Avg pair tau (vine)']:.4f} & {r['Method']} \\\\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")

        path = self.cfg.TABLES_DIR / 'vine_copula_latex.tex'
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[OK] LaTeX saved -> {path}")


# --- 5. EXECUTION ---
if __name__ == "__main__":
    cfg = VineConfig()
    engine = VineCopulaEngine(cfg)
    engine.run()
