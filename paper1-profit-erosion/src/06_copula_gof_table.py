"""
Module: 06_copula_gof_table.py
Description: Formal Goodness-of-Fit comparison for Gaussian, Gumbel, Clayton,
             and Frank copulas across Stable and Crisis regimes.
             Pseudo-observations via empirical PIT (rankdata / n+1).
             All Archimedean copulas fitted by MLE (not method-of-moments)
             so comparisons are on equal footing with Gaussian MLE.
             Outputs publication-quality AIC/BIC/Log-Likelihood table.
Author: Dimitrios Chronis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import kendalltau, norm, rankdata
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')


# --- 1. CONFIGURATION ---
@dataclass
class GoFConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'clean_returns.csv'
    TABLES_DIR: Path = BASE_DIR / 'results' / 'tables'
    FIGURES_DIR: Path = BASE_DIR / 'results' / 'figures'

    # Regime definitions (user-specified)
    STABLE_START: str = '2014-01-01'
    STABLE_END:   str = '2019-12-31'
    CRISIS_START: str = '2021-01-01'
    CRISIS_END:   str = '2024-12-31'

    SEED: int = 42


# --- 2. PSEUDO-OBSERVATIONS (Probability Integral Transform) ---
def pseudo_obs(data: np.ndarray) -> np.ndarray:
    """
    Convert raw returns to uniform pseudo-observations via empirical PIT.
    Uses scipy.stats.rankdata (handles ties) divided by n+1 to ensure
    strict (0,1) support, following Genest & Favre (2007).
    """
    n, d = data.shape
    u = np.zeros_like(data, dtype=float)
    for j in range(d):
        u[:, j] = rankdata(data[:, j]) / (n + 1)
    return u


# --- 3. COPULA LOG-LIKELIHOOD FUNCTIONS (all fitted by MLE) ---

class GaussianCopula:
    name = "Gaussian"

    @staticmethod
    def fit(u: np.ndarray):
        """MLE via normal-score correlation matrix."""
        x = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        R = np.corrcoef(x.T)
        eigvals = np.linalg.eigvalsh(R)
        if eigvals.min() < 1e-8:
            R += np.eye(R.shape[0]) * (1e-6 - eigvals.min())
        return R

    @staticmethod
    def log_likelihood(u: np.ndarray, R: np.ndarray) -> float:
        n, d = u.shape
        x = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return -np.inf
        R_inv = np.linalg.inv(R)
        quad = np.einsum('ni,ij,nj->n', x, R_inv - np.eye(d), x)
        return float(np.sum(-0.5 * logdet - 0.5 * quad))

    @staticmethod
    def num_params(d: int) -> int:
        return d * (d - 1) // 2


class GumbelCopula:
    name = "Gumbel"

    @staticmethod
    def _bivariate_ll(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
        """Exact bivariate Gumbel copula log-density."""
        u1 = np.clip(u1, 1e-9, 1 - 1e-9)
        u2 = np.clip(u2, 1e-9, 1 - 1e-9)
        lu1 = -np.log(u1)
        lu2 = -np.log(u2)
        S = lu1 ** theta + lu2 ** theta
        S1t = S ** (1.0 / theta)
        log_c = (-S1t
                 + np.log(S1t + theta - 1.0)
                 - (2.0 - 1.0 / theta) * np.log(S)
                 + (theta - 1.0) * (np.log(lu1) + np.log(lu2))
                 - np.log(u1) - np.log(u2))
        mask = np.isfinite(log_c)
        return float(np.sum(log_c[mask]))

    @staticmethod
    def fit(u: np.ndarray):
        """MLE via scipy.optimize.minimize, theta in [1, 20]."""
        d = u.shape[1]
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

        def neg_ll(x):
            theta = float(x[0])
            if theta <= 1.0:
                return 1e12
            total = 0.0
            for i, j in pairs:
                total += GumbelCopula._bivariate_ll(u[:, i], u[:, j], theta)
            return -total

        res = minimize(neg_ll, x0=[2.0], method='L-BFGS-B',
                       bounds=[(1.0001, 20.0)],
                       options={'ftol': 1e-12, 'gtol': 1e-8})
        return float(res.x[0])

    @staticmethod
    def log_likelihood(u: np.ndarray, theta: float) -> float:
        d = u.shape[1]
        ll = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                ll += GumbelCopula._bivariate_ll(u[:, i], u[:, j], theta)
        return ll

    @staticmethod
    def num_params(d: int) -> int:
        return 1


class ClaytonCopula:
    name = "Clayton"

    @staticmethod
    def _bivariate_ll(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
        u1 = np.clip(u1, 1e-9, 1 - 1e-9)
        u2 = np.clip(u2, 1e-9, 1 - 1e-9)
        # c(u1,u2) = (1+theta)(u1*u2)^(-1-theta)*(u1^-theta+u2^-theta-1)^(-2-1/theta)
        inner = u1 ** (-theta) + u2 ** (-theta) - 1.0
        inner = np.maximum(inner, 1e-300)
        log_c = (np.log(1.0 + theta)
                 + (-1.0 - theta) * (np.log(u1) + np.log(u2))
                 + (-2.0 - 1.0 / theta) * np.log(inner))
        mask = np.isfinite(log_c)
        return float(np.sum(log_c[mask]))

    @staticmethod
    def fit(u: np.ndarray):
        """MLE via scipy.optimize.minimize, theta in [0.01, 20]."""
        d = u.shape[1]
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

        def neg_ll(x):
            theta = float(x[0])
            if theta <= 0:
                return 1e12
            total = 0.0
            for i, j in pairs:
                total += ClaytonCopula._bivariate_ll(u[:, i], u[:, j], theta)
            return -total

        res = minimize(neg_ll, x0=[1.0], method='L-BFGS-B',
                       bounds=[(0.01, 20.0)],
                       options={'ftol': 1e-12, 'gtol': 1e-8})
        return float(res.x[0])

    @staticmethod
    def log_likelihood(u: np.ndarray, theta: float) -> float:
        d = u.shape[1]
        ll = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                ll += ClaytonCopula._bivariate_ll(u[:, i], u[:, j], theta)
        return ll

    @staticmethod
    def num_params(d: int) -> int:
        return 1


class FrankCopula:
    name = "Frank"

    @staticmethod
    def _bivariate_ll(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
        u1 = np.clip(u1, 1e-9, 1 - 1e-9)
        u2 = np.clip(u2, 1e-9, 1 - 1e-9)
        if abs(theta) < 1e-6:
            return 0.0
        et = np.exp(theta)
        e1 = np.exp(theta * u1)
        e2 = np.exp(theta * u2)
        numer = theta * (et - 1.0) * np.exp(theta * (u1 + u2))
        denom = ((et - 1.0) + (e1 - 1.0) * (e2 - 1.0)) ** 2
        log_c = np.log(np.abs(numer) + 1e-300) - np.log(np.abs(denom) + 1e-300)
        mask = np.isfinite(log_c)
        return float(np.sum(log_c[mask]))

    @staticmethod
    def fit(u: np.ndarray):
        """MLE via scipy.optimize.minimize, theta in [-20, 20] (allows negative dependence)."""
        d = u.shape[1]
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

        def neg_ll(x):
            theta = float(x[0])
            if abs(theta) < 1e-6:
                return 1e12
            total = 0.0
            for i, j in pairs:
                total += FrankCopula._bivariate_ll(u[:, i], u[:, j], theta)
            return -total

        # Try positive start; also try negative to avoid local minima
        best_res = None
        for x0 in [2.0, -2.0, 5.0]:
            res = minimize(neg_ll, x0=[x0], method='L-BFGS-B',
                           bounds=[(-20.0, 20.0)],
                           options={'ftol': 1e-12, 'gtol': 1e-8})
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        return float(best_res.x[0])

    @staticmethod
    def log_likelihood(u: np.ndarray, theta: float) -> float:
        d = u.shape[1]
        ll = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                ll += FrankCopula._bivariate_ll(u[:, i], u[:, j], theta)
        return ll

    @staticmethod
    def num_params(d: int) -> int:
        return 1


# --- 4. MAIN GOF ENGINE ---
class CopulaGoFAnalyzer:
    COPULAS = [GaussianCopula, GumbelCopula, ClaytonCopula, FrankCopula]

    def __init__(self, cfg: GoFConfig):
        self.cfg = cfg
        self._gaussian_wins_flag = None  # set by _write_flag() if triggered

    def run(self):
        self.cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Load price-level data and compute log-returns
        df = pd.read_csv(self.cfg.DATA_PATH, index_col='Date', parse_dates=True)
        returns = np.log(df / df.shift(1)).dropna()

        cols = ['Concrete', 'Steel', 'Fuel_Energy', 'PVC_Pipes']
        returns = returns[cols]

        regimes = {
            'Stable (2014-2019)': returns[self.cfg.STABLE_START:self.cfg.STABLE_END],
            'Crisis (2021-2024)': returns[self.cfg.CRISIS_START:self.cfg.CRISIS_END],
        }

        print("\nPIT METHOD: scipy.stats.rankdata / (n+1) applied per column")
        print("FIT METHOD: Full MLE via minimize_scalar on composite pairwise log-likelihood")
        print("           (Gaussian: normal-score MLE; Archimedean: direct LL optimisation)")

        rows = []
        for regime_name, regime_data in regimes.items():
            raw = regime_data.values
            # --- PIT: empirical CDF via rankdata / (n+1) ---
            u = pseudo_obs(raw)
            n, d = u.shape

            # Sanity check: confirm all values strictly in (0,1)
            assert u.min() > 0 and u.max() < 1, "PIT output outside (0,1)!"

            print(f"\n{'='*60}")
            print(f"Regime: {regime_name}  (n={n}, d={d})")
            print(f"  PIT check: min={u.min():.4f}, max={u.max():.4f} (should be in (0,1))")
            print(f"{'='*60}")

            regime_rows = []
            for CopClass in self.COPULAS:
                param = CopClass.fit(u)
                ll = CopClass.log_likelihood(u, param)
                k = CopClass.num_params(d)
                aic = 2 * k - 2 * ll
                bic = k * np.log(n) - 2 * ll

                if CopClass.name == 'Gaussian':
                    param_str = f"rho_bar = {np.mean(param[np.triu_indices(d, 1)]):.3f}"
                else:
                    param_str = f"theta = {param:.3f}"

                regime_rows.append({
                    'Regime':           regime_name,
                    'Copula':           CopClass.name,
                    'Parameters':       param_str,
                    'Log-Likelihood':   round(ll, 1),
                    'AIC':              round(aic, 1),
                    'BIC':              round(bic, 1),
                    'k':                k,
                    'n':                n,
                })
                print(f"  {CopClass.name:10s} | theta/rho={param if CopClass.name != 'Gaussian' else np.mean(param[np.triu_indices(d, 1)]):.3f}"
                      f" | LL={ll:10.1f} | AIC={aic:10.1f} | BIC={bic:10.1f}")

            # Identify best per regime
            best_aic = min(regime_rows, key=lambda r: r['AIC'])
            best_bic = min(regime_rows, key=lambda r: r['BIC'])
            print(f"\n  Best AIC: {best_aic['Copula']} (AIC={best_aic['AIC']})")
            print(f"  Best BIC: {best_bic['Copula']} (BIC={best_bic['BIC']})")

            # Honest flag: warn if Gaussian beats Gumbel in Crisis regime
            if 'Crisis' in regime_name:
                gumbel_row = next(r for r in regime_rows if r['Copula'] == 'Gumbel')
                gauss_row  = next(r for r in regime_rows if r['Copula'] == 'Gaussian')
                if gauss_row['AIC'] < gumbel_row['AIC']:
                    delta = gumbel_row['AIC'] - gauss_row['AIC']
                    print(f"\n  [FLAG] Gaussian STILL outperforms Gumbel in {regime_name} after correct PIT+MLE.")
                    print(f"         Gaussian AIC={gauss_row['AIC']}, Gumbel AIC={gumbel_row['AIC']} (delta={delta:.1f})")
                    print(f"         This is an honest empirical result -- symmetric dependence dominates")
                    print(f"         in this regime. Report as-is; do NOT force Gumbel as winner.")
                    self._write_flag(regime_name, gauss_row, gumbel_row)
                else:
                    print(f"\n  [OK] Gumbel outperforms Gaussian in Crisis regime as expected.")

            rows.extend(regime_rows)

        gof_df = pd.DataFrame(rows)
        for ic in ('AIC', 'BIC'):
            gof_df[f'{ic}_Rank'] = gof_df.groupby('Regime')[ic].rank(ascending=True).astype(int)
        gof_df['Best (AIC)'] = gof_df['AIC_Rank'] == 1

        csv_path = self.cfg.TABLES_DIR / 'copula_gof_comparison.csv'
        gof_df.to_csv(csv_path, index=False)
        print(f"\n[OK] CSV saved -> {csv_path}")

        self._render_table_figure(gof_df)
        return gof_df

    def _write_flag(self, regime_name: str, gauss_row: dict, gumbel_row: dict):
        """Write gof_honest_flag.txt if Gaussian beats Gumbel after correct PIT+MLE."""
        flag_path = self.cfg.TABLES_DIR / 'gof_honest_flag.txt'
        delta = gumbel_row['AIC'] - gauss_row['AIC']
        lines = [
            "GOF HONEST FLAG: Gaussian achieves superior marginal fit in Crisis regime",
            "=" * 70,
            f"Regime       : {regime_name}",
            f"Gaussian AIC : {gauss_row['AIC']}",
            f"Gumbel   AIC : {gumbel_row['AIC']}",
            f"Delta (Gumbel - Gaussian) = {delta:.1f} AIC units",
            "",
            "METHOD: PIT via scipy.stats.rankdata/(n+1); MLE via scipy.optimize.minimize",
            "        Gumbel theta in [1,20]; Clayton theta in [0.01,20]; Frank theta in [-20,20]",
            "",
            "HONEST FINDING:",
            "  Gaussian achieves superior marginal fit in this crisis period.",
            "  However, it structurally assumes lambda_U = 0 and cannot capture",
            "  tail co-movement by construction. GoF metrics measure distributional fit,",
            "  not tail dependence adequacy.",
            "",
            "LATEX FOOTNOTE TO ADD:",
            "  'Gaussian achieves superior marginal fit; however, it structurally assumes",
            "   lambda_U = 0 and cannot capture tail co-movement by construction.",
            "   GoF metrics measure distributional fit, not tail dependence adequacy.'",
            "",
            "RECOMMENDATION:",
            "  Report Gaussian as empirically best copula by AIC/BIC for this regime.",
            "  Justify Gumbel selection on the basis of tail dependence theory,",
            "  not on GoF rank. Add the footnote above to Table 2 in the paper.",
            "  Do NOT suppress this result or reverse-engineer the table.",
        ]
        flag_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"  [FLAG FILE] Written -> {flag_path}")
        # Store flag state for LaTeX export
        self._gaussian_wins_flag = {
            'regime': regime_name,
            'gauss_aic': gauss_row['AIC'],
            'gumbel_aic': gumbel_row['AIC'],
            'delta': delta,
        }

    def _render_table_figure(self, gof_df: pd.DataFrame):
        regime_order = ['Stable (2014-2019)', 'Crisis (2021-2024)']
        display_cols = ['Regime', 'Copula', 'Parameters', 'Log-Likelihood', 'AIC', 'BIC']
        display = gof_df[display_cols].copy()
        display['regime_sort'] = display['Regime'].map({r: i for i, r in enumerate(regime_order)})
        display = display.sort_values(['regime_sort', 'AIC']).drop(columns='regime_sort')

        fig_width = 7.0
        n_rows = len(display) + 1
        row_h = 0.32
        fig_height = n_rows * row_h + 0.9
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        col_labels = ['Regime', 'Copula', 'Parameter', 'Log-Lik.', 'AIC', 'BIC']
        col_widths = [0.26, 0.13, 0.18, 0.14, 0.14, 0.15]
        table_data = display.values.tolist()

        the_table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            colWidths=col_widths,
            loc='center',
            cellLoc='center'
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1, 1.45)

        header_color = '#1a1a2e'
        best_color   = '#d4edda'
        crisis_bg    = '#fff3cd'
        stable_bg    = '#f8f9fa'
        alt_color    = '#eef2f7'

        best_per_regime = {}
        for regime in regime_order:
            sub = gof_df[gof_df['Regime'] == regime]
            best_copula = sub.loc[sub['AIC'].idxmin(), 'Copula']
            best_per_regime[regime] = best_copula

        for (row, col), cell in the_table.get_celld().items():
            cell.set_edgecolor('#333333')
            cell.set_linewidth(0.5)

            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color='white', fontweight='bold',
                                    fontfamily='DejaVu Sans', fontsize=8)
            else:
                data_row = display.iloc[row - 1]
                regime = data_row['Regime']
                copula = data_row['Copula']
                is_best = (best_per_regime.get(regime) == copula)
                is_crisis = 'Crisis' in regime

                if is_best:
                    cell.set_facecolor(best_color)
                    cell.set_text_props(fontweight='bold')
                elif is_crisis:
                    bg = crisis_bg if (row % 2 == 0) else '#ffeeba'
                    cell.set_facecolor(bg)
                else:
                    bg = stable_bg if (row % 2 == 0) else alt_color
                    cell.set_facecolor(bg)

                cell.set_text_props(fontfamily='DejaVu Sans', fontsize=8)

        n_stable = len(display[display['Regime'] == regime_order[0]])
        y_sep = 1.0 - (n_stable + 1) / n_rows

        from matplotlib.lines import Line2D
        line = Line2D([0.01, 0.99], [y_sep, y_sep],
                      transform=ax.transAxes,
                      color='#333333', linewidth=1.6, clip_on=False)
        ax.add_artist(line)

        note_text = (
            'Note: Pseudo-observations via empirical PIT (rankdata/(n+1), Genest & Favre 2007). '
            'All copulas fitted by MLE on uniform margins. '
            'Bold/green = best AIC per regime. Lower AIC/BIC = superior fit.'
        )
        fig.text(
            0.5, 0.01,
            note_text,
            ha='center', va='bottom', fontsize=6.5, style='italic',
            fontfamily='DejaVu Sans', color='#444444',
            wrap=True
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1.0])
        out_path = self.cfg.FIGURES_DIR / 'fig_copula_gof_table.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"[OK] Figure saved -> {out_path}")

        latex_path = self.cfg.TABLES_DIR / 'copula_gof_latex.tex'
        self._export_latex(display, best_per_regime, latex_path)

    def _export_latex(self, display: pd.DataFrame, best_per_regime: dict, path: Path):
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Copula Goodness-of-Fit Comparison (PIT + MLE)}")
        lines.append(r"\label{tab:copula_gof}")
        lines.append(r"\begin{tabular}{llcrrr}")
        lines.append(r"\toprule")
        lines.append(r"Regime & Copula & Parameter & Log-Lik. & AIC & BIC \\")
        lines.append(r"\midrule")

        current_regime = None
        for _, row in display.iterrows():
            regime = row['Regime']
            copula = row['Copula']
            is_best = best_per_regime.get(regime) == copula

            if current_regime is not None and regime != current_regime:
                lines.append(r"\midrule")
            current_regime = regime

            regime_cell = regime if (display[display['Regime'] == regime].index[0] == _) else ''
            param_raw = str(row['Parameters'])
            param_str = (param_raw
                         .replace('theta', r'$\theta$')
                         .replace('rho_bar', r'$\bar{\rho}$'))
            ll_str  = f"{row['Log-Likelihood']:.1f}"
            aic_str = f"{row['AIC']:.1f}"
            bic_str = f"{row['BIC']:.1f}"

            if is_best:
                line = (f"\\textbf{{{regime_cell}}} & \\textbf{{{copula}}} & "
                        f"\\textbf{{{param_str}}} & \\textbf{{{ll_str}}} & "
                        f"\\textbf{{{aic_str}}} & \\textbf{{{bic_str}}} \\\\")
            else:
                line = f"{regime_cell} & {copula} & {param_str} & {ll_str} & {aic_str} & {bic_str} \\\\"
            lines.append(line)

        lines.append(r"\bottomrule")
        base_note = (
            r"\multicolumn{6}{l}{\footnotesize \textit{Note: Pseudo-observations via "
            r"empirical PIT (rankdata\,/\,(n+1), Genest \& Favre 2007). "
            r"All copulas fitted by MLE on uniform margins. "
            r"Bold/green shading = best AIC per regime. Lower AIC/BIC = superior fit.}} \\"
        )
        lines.append(base_note)

        # If Gaussian won in Crisis regime, add the mandatory honest-flag footnote
        if self._gaussian_wins_flag is not None:
            flag_note = (
                r"\multicolumn{6}{l}{\footnotesize \textit{"
                r"\dag\ Gaussian achieves superior marginal fit; however, it structurally "
                r"assumes $\lambda_U = 0$ and cannot capture tail co-movement by construction. "
                r"GoF metrics measure distributional fit, not tail dependence adequacy.}} \\"
            )
            lines.append(flag_note)

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[OK] LaTeX table saved -> {path}")


# --- 5. EXECUTION ---
if __name__ == "__main__":
    cfg = GoFConfig()
    analyzer = CopulaGoFAnalyzer(cfg)
    gof_df = analyzer.run()

    print("\n" + "="*70)
    print("COPULA GOODNESS-OF-FIT SUMMARY")
    print("="*70)
    summary = gof_df[['Regime', 'Copula', 'Log-Likelihood', 'AIC', 'BIC', 'AIC_Rank']].copy()
    summary = summary.sort_values(['Regime', 'AIC_Rank'])
    print(summary.to_string(index=False))
    print("="*70)
    print("\nOutput files:")
    print("  CSV    -> results/tables/copula_gof_comparison.csv")
    print("  LaTeX  -> results/tables/copula_gof_latex.tex")
    print("  Figure -> results/figures/fig_copula_gof_table.png")
