#!/usr/bin/env python3
"""
compare_models.py — Paired comparative analysis of RL agent log files.

Metrics
-------
Base:       KILLS, T_ALIVE, A_HP, A_STAMINA, A_DAMAGE
Derived:    S1 = K/T
            S2 = K/A_stamina
            S3 = K^2 / (T * A_damage)
            r_H = A_HP / T
            r_Sigma = A_stamina / T
            r_D = A_damage / T
Composite:  M_H = K * A_HP / T
            M_Sigma = K * A_stamina / T
            M_D = K * A_damage / T

All metrics are higher-is-better. No success/failure categories are reported.
-----------------------------------
python compare_models.py \
    --models P=../pre-ewc+kd-backup/agent_log.log \
             B=../backup/agent_log.log \
             F=../backup-FT/agent_log.log \
    --latex out/ --figures out/figures/ --export out/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
EPS = 1e-9

# All metrics are higher-is-better for this study.
LOWER_IS_BETTER: set[str] = set()

ALL_METRICS = [
    # Base (raw)
    "KILLS", "T_ALIVE", "A_HP", "A_STAMINA", "A_DAMAGE",
    # Simple derived
    "S1_KillRate", "S2_KillPerStamina", "S3_EffectiveKillOutput",
    # Resource rates
    "r_HP", "r_Stamina", "r_Damage",
    # Composite (kill-weighted resource retention)
    "M_HP", "M_Stamina", "M_Damage",
]

DERIVED_METRICS = [
    "S1_KillRate", "S2_KillPerStamina", "S3_EffectiveKillOutput",
    "r_HP", "r_Stamina", "r_Damage",
    "M_HP", "M_Stamina", "M_Damage",
]

COMPOSITE_METRICS = ["M_HP", "M_Stamina", "M_Damage"]

METRIC_DISPLAY = {
    "KILLS":                  r"$N_{\mathrm{kills}}$",
    "T_ALIVE":                r"$T_{\mathrm{alive}}$",
    "A_HP":                   r"$A_{\mathrm{HP}}$",
    "A_STAMINA":              r"$A_{\mathrm{stamina}}$",
    "A_DAMAGE":               r"$A_{\mathrm{damage}}$",
    "S1_KillRate":            r"$S_1 = K/T$",
    "S2_KillPerStamina":      r"$S_2 = K/A_{\sigma}$",
    "S3_EffectiveKillOutput": r"$S_3 = K^2/(T\,A_{\mathrm{dmg}})$",
    "r_HP":                   r"$r_H = A_{\mathrm{HP}}/T$",
    "r_Stamina":              r"$r_\sigma = A_{\mathrm{stam}}/T$",
    "r_Damage":               r"$r_D = A_{\mathrm{dmg}}/T$",
    "M_HP":                   r"$M_H = K A_{\mathrm{HP}}/T$",
    "M_Stamina":              r"$M_\sigma = K A_{\mathrm{stam}}/T$",
    "M_Damage":               r"$M_D = K A_{\mathrm{dmg}}/T$",
}

LATEX_NAME_MAP = {
    "P": r"$\theta_0$",
    "B": r"$\theta^\star$",
    "F": r"$\theta_{\mathrm{FT}}$",
}


# ═══════════════════════════════════════════════════════════════════════════
# PARSER (log format unchanged)
# ═══════════════════════════════════════════════════════════════════════════
FIELD_PATTERNS = {
    "KILLS":            r"KILLS\s*=\s*(\d+)",
    "T_ALIVE":          r"T_ALIVE\s*=\s*(\d+)",
    "A_HP":             r"A_HP\s*=\s*(\d+)",
    "A_STAMINA":        r"A_STAMINA\s*=\s*(\d+)",
    "A_DAMAGE":         r"A_DAMAGE\s*=\s*(\d+)",
    # The following are still parsed for compatibility but not reported.
    "Total_Actions":    r"Total Actions\s*:\s*(\d+)",
    "Overall_Failures": r"Overall Failures\s*:\s*(\d+)",
    "WC_minus":         r"Wall Collisions\s*\(-\)\s*:\s*(\d+)",
    "AE_minus":         r"Adjancent Enemies\s*\(-\)\s*:\s*(\d+)",
    "FE_minus":         r"Flanking Enemies\s*\(-\)\s*:\s*(\d+)",
    "Overall_Success":  r"Overall Success\s*:\s*(\d+)",
    "WC_plus":          r"Wall Collisions\s*\(\+\)\s*:\s*(\d+)",
    "AE_plus":          r"Adjancent Enemies\s*\(\+\)\s*:\s*(\d+)",
    "FE_plus":          r"Flanking Enemies\s*\(\+\)\s*:\s*(\d+)",
}

# Fields the parser must find for an episode to be valid.
REQUIRED_FIELDS = {"KILLS", "T_ALIVE", "A_HP", "A_STAMINA", "A_DAMAGE"}


def parse_log(path: str) -> tuple[pd.DataFrame, list[tuple[int, list[str]]]]:
    text = Path(path).read_text()
    raw_blocks = re.split(r"\n={5,}\s*\n", "\n" + text)
    episodes, bad = [], []
    for i, blk in enumerate(raw_blocks):
        if "METRICS" not in blk:
            continue
        d, missing = {}, []
        for k, pat in FIELD_PATTERNS.items():
            m = re.search(pat, blk)
            if m is None:
                if k in REQUIRED_FIELDS:
                    missing.append(k)
            else:
                d[k] = int(m.group(1))
        if missing:
            bad.append((i, missing))
            continue
        episodes.append(d)
    return pd.DataFrame(episodes), bad


# ═══════════════════════════════════════════════════════════════════════════
# DERIVED METRICS
# ═══════════════════════════════════════════════════════════════════════════
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    K = df["KILLS"]
    T = df["T_ALIVE"] + EPS
    H = df["A_HP"]
    Sig = df["A_STAMINA"]
    D = df["A_DAMAGE"]

    # Simple derived
    df["S1_KillRate"]            = K / T
    df["S2_KillPerStamina"]      = K / (Sig + EPS)
    df["S3_EffectiveKillOutput"] = (K ** 2) / (T * (D + EPS))

    # Resource rates
    df["r_HP"]      = H / T
    df["r_Stamina"] = Sig / T
    df["r_Damage"]  = D / T

    # Composite: kill * resource / T
    df["M_HP"]      = K * H / T
    df["M_Stamina"] = K * Sig / T
    df["M_Damage"]  = K * D / T

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
def sign_test_exact(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(wins, losses), n, 0.5,
                                 alternative="two-sided").pvalue)


def cohens_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    return float("nan") if sd == 0 else float(np.mean(diff) / sd)


def hedges_g(diff: np.ndarray) -> float:
    n = len(diff)
    dz = cohens_dz(diff)
    if np.isnan(dz) or n < 4:
        return float("nan")
    return dz * (1 - 3 / (4 * n - 9))


def interpret_dz(dz: float) -> str:
    if dz is None or np.isnan(dz):
        return "n/a"
    a = abs(dz)
    if a < 0.2:  return "negligible"
    if a < 0.5:  return "small"
    if a < 0.8:  return "medium"
    return "large"


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = p * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj_sorted, 0.0, 1.0)
    return out.tolist()


def paired_stats(x: np.ndarray, y: np.ndarray, *,
                 alpha: float = 0.05, n_boot: int = 5000,
                 rng_seed: int = 0) -> dict | None:
    d = np.asarray(y, float) - np.asarray(x, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return None

    mean_d = float(np.mean(d))
    sd_d   = float(np.std(d, ddof=1))
    se     = sd_d / np.sqrt(n)
    t_stat, p_t = stats.ttest_rel(y, x)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lo, ci_hi = mean_d - t_crit * se, mean_d + t_crit * se

    dz = cohens_dz(d)
    g  = hedges_g(d)
    wins_y = int(np.sum(d > 0))
    wins_x = int(np.sum(d < 0))
    ties   = int(np.sum(d == 0))
    skew   = float(stats.skew(d)) if n >= 3 else float("nan")

    if np.all(d == 0):
        p_wilcox = 1.0
    else:
        try:
            _, p_wilcox = stats.wilcoxon(y, x, zero_method="wilcox")
            p_wilcox = float(p_wilcox)
        except Exception:
            p_wilcox = float("nan")

    p_sign = sign_test_exact(wins_y, wins_x)

    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    boot_lo, boot_hi = np.percentile(boot_means, [2.5, 97.5])

    p_shapiro = float(stats.shapiro(d)[1]) if n >= 3 else float("nan")

    use_nonparam = (n < 30) or (abs(skew) > 1) or (
        not np.isnan(p_shapiro) and p_shapiro < 0.05
    )
    if use_nonparam and not np.isnan(p_wilcox):
        p_primary, test_used = p_wilcox, "Wilcoxon"
    elif use_nonparam:
        p_primary, test_used = p_sign, "sign"
    else:
        p_primary, test_used = float(p_t), "paired t"

    return dict(
        n=n, mean_diff=mean_d, sd_diff=sd_d, se=se,
        t=float(t_stat), df=n - 1, p_t=float(p_t),
        ci95_low=ci_lo, ci95_high=ci_hi,
        boot_ci_low=float(boot_lo), boot_ci_high=float(boot_hi),
        cohens_dz=dz, hedges_g=g, effect_size=interpret_dz(dz),
        M2_wins=wins_y, M1_wins=wins_x, ties=ties, skew=skew,
        p_wilcoxon=p_wilcox, p_sign=p_sign, p_shapiro=p_shapiro,
        p_primary=float(p_primary), test_used=test_used,
        use_nonparam=use_nonparam,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Analysis:
    names: list[str]
    frames: dict[str, pd.DataFrame]
    pairs: list[tuple[str, str]]
    results: dict[tuple[str, str], dict[str, dict]]
    n: int
    alpha: float = 0.05

    def metric_means(self, metric: str) -> dict[str, float]:
        return {m: float(self.frames[m][metric].mean()) for m in self.names}

    def winner(self, metric: str, a_name: str, b_name: str) -> str:
        a = self.frames[a_name][metric].mean()
        b = self.frames[b_name][metric].mean()
        if np.isnan(a) or np.isnan(b):
            return "n/a"
        if metric in LOWER_IS_BETTER:
            return a_name if a < b else (b_name if b < a else "tie")
        return a_name if a > b else (b_name if b > a else "tie")


def run_analysis(model_logs: dict[str, str], *,
                 alpha: float = 0.05, n_boot: int = 5000,
                 n_first: int | None = None,
                 verbose: bool = True) -> Analysis:
    frames, integrity = {}, {}
    for name, path in model_logs.items():
        df, bad = parse_log(path)
        frames[name] = df
        integrity[name] = bad
        if verbose:
            print(f"  {name}: {len(df)} valid blocks, {len(bad)} malformed blocks")
            for i, missing in bad:
                print(f"    block #{i} missing: {missing}")

    n = min(len(df) for df in frames.values())
    if n_first is not None:
        n = min(n, n_first)
    if verbose:
        print(f"  Final paired n = {n}")

    for name in frames:
        frames[name] = add_derived(frames[name].iloc[:n].reset_index(drop=True))

    names = list(model_logs.keys())
    pairs = [(names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]

    raw = {}
    pvals, keys = [], []
    for (a, b) in pairs:
        raw[(a, b)] = {}
        for k in ALL_METRICS:
            r = paired_stats(frames[a][k].values, frames[b][k].values,
                             alpha=alpha, n_boot=n_boot)
            if r is None:
                continue
            raw[(a, b)][k] = r
            pvals.append(r["p_primary"])
            keys.append(((a, b), k))

    qvals = bh_fdr(pvals)
    for (pair, k), q in zip(keys, qvals):
        raw[pair][k]["p_fdr"] = q

    return Analysis(names=names, frames=frames, pairs=pairs,
                    results=raw, n=n, alpha=alpha)


# ═══════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════
def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001: return " ✔✔"
    if p < 0.05:  return " ✔"
    return ""


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def bar(value, max_value, width=40, fill="█", empty="░"):
    if max_value <= 0:
        return ""
    n = int(round(width * max(value, 0) / max_value))
    return fill * n + empty * (width - n)


def hdr(title):
    print()
    print("═" * 78)
    print(title)
    print("═" * 78)


# ═══════════════════════════════════════════════════════════════════════════
# CONSOLE REPORTER
# ═══════════════════════════════════════════════════════════════════════════
class ConsoleReporter:
    def __init__(self, a: Analysis):
        self.a = a

    def report(self):
        self._summary()
        self._descriptive()
        self._derived()
        self._composite()
        self._stats_table()
        self._forest()
        self._pareto()
        self._findings()
        self._scorecard()

    def _summary(self):
        a = self.a
        hdr("SECTION 0 — EXECUTIVE SUMMARY (per pair)")
        for (m1, m2) in a.pairs:
            print(f"\n--- {m1} vs {m2} ---")
            axes = [
                ("KILLS", "KILLS"),
                ("T_ALIVE", "T_ALIVE"),
                ("S1 = K/T", "S1_KillRate"),
                ("S2 = K/A_stam", "S2_KillPerStamina"),
                ("S3", "S3_EffectiveKillOutput"),
                ("r_HP", "r_HP"),
                ("r_Sigma", "r_Stamina"),
                ("r_D", "r_Damage"),
                ("M_HP", "M_HP"),
                ("M_Sigma", "M_Stamina"),
                ("M_D", "M_Damage"),
            ]
            print(f"{'Axis':<20}{m1:>14}{m2:>14}{'Winner':>10}"
                  f"{'p_prim':>10}{'p_FDR':>10}{'':>4}")
            for label, k in axes:
                x = a.frames[m1][k].mean()
                y = a.frames[m2][k].mean()
                w = a.winner(k, m1, m2)
                r = a.results[(m1, m2)][k]
                print(f"{label:<20}{x:>14.6g}{y:>14.6g}{w:>10}"
                      f"{fmt_p(r['p_primary']):>10}{fmt_p(r['p_fdr']):>10}"
                      f"{stars(r['p_fdr']):>4}")

    def _descriptive(self):
        a = self.a
        hdr("SECTION 1 — BASE METRICS")
        print("\n### Raw aggregate totals")
        print(f"{'Metric':<22}" + "".join(f"{n:>15}" for n in a.names))
        for k in ["KILLS", "T_ALIVE", "A_HP", "A_STAMINA", "A_DAMAGE"]:
            print(f"{k:<22}" + "".join(
                f"{a.frames[n][k].sum():>15.0f}" for n in a.names
            ))

        print("\n### Per-episode mean ± SD")
        print(f"{'Metric':<22}" + "".join(f"{n+' mean±sd':>28}" for n in a.names))
        for k in ["KILLS", "T_ALIVE", "A_HP", "A_STAMINA", "A_DAMAGE"]:
            row = f"{k:<22}"
            for n in a.names:
                x = a.frames[n][k].values
                row += f"{np.mean(x):>14.4g} ± {np.std(x, ddof=1):<12.4g}"
            print(row)

    def _derived(self):
        a = self.a
        hdr("SECTION 2 — DERIVED METRICS")
        print(f"{'Metric':<26}" + "".join(f"{n:>15}" for n in a.names)
              + f"{'best':>10}")
        derived = [
            ("S1_KillRate", "S1 = K/T"),
            ("S2_KillPerStamina", "S2 = K/A_stam"),
            ("S3_EffectiveKillOutput", "S3"),
            ("r_HP", "r_HP = A_HP/T"),
            ("r_Stamina", "r_Sigma = A_stam/T"),
            ("r_Damage", "r_D = A_dmg/T"),
        ]
        for k, label in derived:
            row = f"{label:<26}"
            means = {}
            for n in a.names:
                m = a.frames[n][k].mean()
                means[n] = m
                row += f"{m:>15.6g}"
            best = max(means, key=means.get)  # all higher-is-better
            row += f"{best:>10}"
            print(row)

    def _composite(self):
        a = self.a
        hdr("SECTION 3 — COMPOSITE METRICS (anti-loafing)")
        print(f"{'Metric':<26}" + "".join(f"{n:>15}" for n in a.names)
              + f"{'best':>10}")
        for k, label in [("M_HP", "M_H = K A_HP/T"),
                         ("M_Stamina", "M_Sig = K A_stam/T"),
                         ("M_Damage", "M_D = K A_dmg/T")]:
            row = f"{label:<26}"
            means = {}
            for n in a.names:
                m = a.frames[n][k].mean()
                means[n] = m
                row += f"{m:>15.6g}"
            best = max(means, key=means.get)
            row += f"{best:>10}"
            print(row)

        print("\n### Relative composite score (per model, mean across 3)")
        rel = {n: [] for n in a.names}
        for k in COMPOSITE_METRICS:
            vals = [a.frames[n][k].mean() for n in a.names]
            mx = max(vals)
            for n, v in zip(a.names, vals):
                rel[n].append(v / mx if mx > 0 else 0.0)
        print(f"{'Model':<10}{'mean rel.':>12}")
        for n in a.names:
            print(f"{n:<10}{np.mean(rel[n]):>12.4f}")

    def _stats_table(self):
        a = self.a
        hdr("SECTION 4 — STATISTICAL TESTS")
        print(f"n = {a.n}\n")
        for (m1, m2) in a.pairs:
            print(f"\n--- {m1} vs {m2} (Δ = {m2} − {m1}) ---")
            print(f"{'Metric':<26}{'Δ mean':>14}{'SE':>14}{'p_prim':>10}"
                  f"{'p_FDR':>10}{'test':>14}{'dz':>7}{'size':>14}{'wins':>10}")
            for k in ALL_METRICS:
                r = a.results[(m1, m2)].get(k)
                if r is None:
                    continue
                wins = f"{r['M1_wins']}:{r['M2_wins']}"
                print(f"{k:<26}{r['mean_diff']:>14.6g}{r['se']:>14.6g}"
                      f"{fmt_p(r['p_primary']):>10}{fmt_p(r['p_fdr']):>10}"
                      f"{r['test_used']:>14}{fmt(r['cohens_dz'],2):>7}"
                      f"{r['effect_size']:>14}{wins:>10}")

    def _forest(self):
        a = self.a
        hdr("SECTION 4.2 — FOREST PLOTS (per pair, ranked by |dz|)")
        for (m1, m2) in a.pairs:
            print(f"\n--- {m1} vs {m2} ---")
            ranked = sorted(
                a.results[(m1, m2)].items(),
                key=lambda kv: abs(kv[1]["cohens_dz"])
                if not np.isnan(kv[1]["cohens_dz"]) else 0,
                reverse=True,
            )
            for k, r in ranked:
                span = max(abs(r["ci95_low"]), abs(r["ci95_high"]), 1e-9)

                def col(v):
                    return int(round(20 + 19 * v / span))

                pl, ph, pm = col(r["ci95_low"]), col(r["ci95_high"]), col(r["mean_diff"])
                pl, ph = max(0, pl), min(40, ph)
                pm = max(0, min(40, pm))
                line = [" "] * 41
                for i in range(pl, ph + 1):
                    line[i] = "═"
                line[20] = "│"
                line[pm] = "●"
                print(f"{k:<26} |{''.join(line)}| "
                      f"Δ={r['mean_diff']:+.6g} q={fmt_p(r['p_fdr'])} {stars(r['p_fdr'])}")

    def _pareto(self):
        a = self.a
        hdr("SECTION 5 — PARETO FRONTIER (KILLS vs M_Damage)")
        print(f"{'Model':<12}{'KILLS':>14}{'M_Damage':>18}{'M_HP':>18}")
        for n in a.names:
            v = a.frames[n]["KILLS"].sum()
            md = a.frames[n]["M_Damage"].mean()
            mh = a.frames[n]["M_HP"].mean()
            print(f"{n:<12}{v:>14.0f}{md:>18.6g}{mh:>18.6g}")

        pts = {n: (a.frames[n]["KILLS"].sum(),
                   a.frames[n]["M_Damage"].mean()) for n in a.names}
        for n in a.names:
            dominated_by = [
                m for m in a.names if m != n
                and pts[m][0] >= pts[n][0] and pts[m][1] >= pts[n][1]
                and (pts[m][0] > pts[n][0] or pts[m][1] > pts[n][1])
            ]
            if dominated_by:
                print(f"→ {n} dominated by {dominated_by}")
            else:
                print(f"→ {n} is Pareto-optimal")

    def _findings(self):
        a = self.a
        hdr("SECTION 6 — KEY FINDINGS")
        for (m1, m2) in a.pairs:
            print(f"\n=== {m1} vs {m2} ===")
            ranked = sorted(
                a.results[(m1, m2)].items(),
                key=lambda kv: abs(kv[1]["cohens_dz"])
                if not np.isnan(kv[1]["cohens_dz"]) else 0,
                reverse=True,
            )
            for i, (k, r) in enumerate(ranked[:8], 1):
                w = a.winner(k, m1, m2)
                sig = stars(r["p_fdr"])
                sig_txt = "significant" if sig.strip() else "n.s."
                print(f"\n**Finding {i} — {k}: {w} wins ({sig_txt}).**")
                print(f"  Δ={r['mean_diff']:+.6g}, 95% CI "
                      f"[{r['ci95_low']:.6g},{r['ci95_high']:.6g}], "
                      f"q={fmt_p(r['p_fdr'])}, dz={fmt(r['cohens_dz'],2)} "
                      f"({r['effect_size']}), wins={r['M1_wins']}:{r['M2_wins']}.")

    def _scorecard(self):
        a = self.a
        hdr("SECTION 7 — FINAL SCORECARD (dz-based, higher-is-better)")
        score_dims = [
            ("Kills",         "KILLS"),
            ("S1 = K/T",      "S1_KillRate"),
            ("S2 = K/A_stam", "S2_KillPerStamina"),
            ("S3",            "S3_EffectiveKillOutput"),
            ("r_HP",          "r_HP"),
            ("r_Sigma",       "r_Stamina"),
            ("r_D",           "r_Damage"),
            ("M_HP",          "M_HP"),
            ("M_Sigma",       "M_Stamina"),
            ("M_D",           "M_Damage"),
        ]
        for (m1, m2) in a.pairs:
            print(f"\n--- {m1} vs {m2} ---")
            print(f"{'Dimension':<18}{m1:>10}{m2:>10}{'Winner':>10}{'dz':>8}")
            tally = {m1: 0, m2: 0}
            for label, k in score_dims:
                r = a.results[(m1, m2)][k]
                dz = r["cohens_dz"]
                if k in LOWER_IS_BETTER:
                    dz = -dz
                s1 = 50 - 50 * np.tanh(dz / 2)
                s2 = 50 + 50 * np.tanh(dz / 2)
                w = m1 if s1 > s2 else (m2 if s2 > s1 else "tie")
                if w in tally:
                    tally[w] += 1
                print(f"{label:<18}{s1:>10.1f}{s2:>10.1f}{w:>10}{fmt(dz,2):>8}")
            print(f"Axis count: {m1}={tally[m1]}, {m2}={tally[m2]}")


# ═══════════════════════════════════════════════════════════════════════════
# LATEX REPORTER
# ═══════════════════════════════════════════════════════════════════════════
class LatexReporter:
    def __init__(self, a: Analysis, out_dir: Path):
        self.a = a
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)

    def _name(self, n):
        return LATEX_NAME_MAP.get(n, n)

    def write_all(self):
        self._write_base_table()
        self._write_derived_table()
        self._write_composite_table()
        self._write_pairwise_effect_table()
        self._write_scorecard_table()
        print(f"  LaTeX tables written to {self.out.resolve()}")

    def _write_base_table(self):
        a = self.a
        path = self.out / "tab_base.tex"
        with open(path, "w") as f:
            cols = "l" + "c" * len(a.names)
            f.write(r"\begin{table}[h]" "\n")
            f.write(r"\centering" "\n")
            f.write(r"\small" "\n")
            f.write(r"\begin{tabular}{" + cols + "}\n")
            f.write(r"\toprule" "\n")
            f.write("Metric & " + " & ".join(self._name(n) for n in a.names)
                    + r" \\" "\n")
            f.write(r"\midrule" "\n")
            for k in ["T_ALIVE", "KILLS", "A_HP", "A_STAMINA", "A_DAMAGE"]:
                line = METRIC_DISPLAY.get(k, k) + " & "
                vals = []
                for n in a.names:
                    x = a.frames[n][k].values
                    vals.append(f"${np.mean(x):.2f} \\pm {np.std(x, ddof=1):.2f}$")
                line += " & ".join(vals) + r" \\" "\n"
                f.write(line)
            f.write(r"\bottomrule" "\n")
            f.write(r"\end{tabular}" "\n")
            f.write(
                rf"\caption{{Base metrics over $n={a.n}$ paired episodes. "
                rf"Mean $\pm$ SD.}}"
                "\n"
            )
            f.write(r"\label{tab:base}" "\n")
            f.write(r"\end{table}" "\n")

    def _write_derived_table(self):
        a = self.a
        path = self.out / "tab_derived.tex"
        with open(path, "w") as f:
            cols = "l" + "c" * len(a.names)
            f.write(r"\begin{table}[h]" "\n")
            f.write(r"\centering" "\n")
            f.write(r"\small" "\n")
            f.write(r"\begin{tabular}{" + cols + "}\n")
            f.write(r"\toprule" "\n")
            f.write("Metric & " + " & ".join(self._name(n) for n in a.names)
                    + r" \\" "\n")
            f.write(r"\midrule" "\n")
            rows = [
                ("S1_KillRate",            r"$S_1 = K/T$"),
                ("S2_KillPerStamina",      r"$S_2 = K/A_\sigma$"),
                ("S3_EffectiveKillOutput", r"$S_3 = K^2/(T\,A_\mathrm{dmg})$"),
                ("r_HP",                   r"$r_H = A_\mathrm{HP}/T$"),
                ("r_Stamina",              r"$r_\sigma = A_\mathrm{stam}/T$"),
                ("r_Damage",               r"$r_D = A_\mathrm{dmg}/T$"),
            ]
            for k, name in rows:
                vals = [f"${a.frames[n][k].mean():.6g}$" for n in a.names]
                f.write(name + " & " + " & ".join(vals) + r" \\" "\n")
            f.write(r"\bottomrule" "\n")
            f.write(r"\end{tabular}" "\n")
            f.write(
                rf"\caption{{Derived metrics over $n={a.n}$ paired episodes. "
                rf"All higher-is-better.}}"
                "\n"
            )
            f.write(r"\label{tab:derived}" "\n")
            f.write(r"\end{table}" "\n")

    def _write_composite_table(self):
        a = self.a
        path = self.out / "tab_composite.tex"
        with open(path, "w") as f:
            cols = "l" + "c" * len(a.names)
            f.write(r"\begin{table}[h]" "\n")
            f.write(r"\centering" "\n")
            f.write(r"\small" "\n")
            f.write(r"\begin{tabular}{" + cols + "}\n")
            f.write(r"\toprule" "\n")
            f.write("Metric & " + " & ".join(self._name(n) for n in a.names)
                    + r" \\" "\n")
            f.write(r"\midrule" "\n")
            rows = [
                ("M_HP",      r"$M_H = K\,A_\mathrm{HP}/T$"),
                ("M_Stamina", r"$M_\sigma = K\,A_\mathrm{stam}/T$"),
                ("M_Damage",  r"$M_D = K\,A_\mathrm{dmg}/T$"),
            ]
            for k, name in rows:
                vals = [f"${a.frames[n][k].mean():.6g}$" for n in a.names]
                f.write(name + " & " + " & ".join(vals) + r" \\" "\n")
            f.write(r"\bottomrule" "\n")
            f.write(r"\end{tabular}" "\n")
            f.write(
                rf"\caption{{Composite metrics over $n={a.n}$ paired episodes. "
                rf"$K^2$ in the numerator penalizes models with low kill rate; "
                rf"all higher-is-better.}}"
                "\n"
            )
            f.write(r"\label{tab:composite}" "\n")
            f.write(r"\end{table}" "\n")

    def _write_pairwise_effect_table(self):
        a = self.a
        for (m1, m2) in a.pairs:
            path = self.out / f"tab_pairwise_{m1}_vs_{m2}.tex"
            with open(path, "w") as f:
                f.write(r"\begin{table}[h]" "\n")
                f.write(r"\centering" "\n")
                f.write(r"\small" "\n")
                f.write(r"\begin{tabular}{lrrrrr}" "\n")
                f.write(r"\toprule" "\n")
                f.write(r"Metric & $\Delta$ & $q_{\mathrm{FDR}}$ & $d_z$ & size & wins \\" "\n")
                f.write(r"\midrule" "\n")
                ranked = sorted(
                    a.results[(m1, m2)].items(),
                    key=lambda kv: abs(kv[1]["cohens_dz"])
                    if not np.isnan(kv[1]["cohens_dz"]) else 0,
                    reverse=True,
                )
                for k, r in ranked:
                    name = METRIC_DISPLAY.get(k, k)
                    q = fmt_p(r["p_fdr"])
                    sz = r["effect_size"]
                    wins = f"{r['M1_wins']}:{r['M2_wins']}"
                    f.write(
                        f"{name} & ${r['mean_diff']:+.6g}$ & ${q}$ & "
                        f"${fmt(r['cohens_dz'],2)}$ & {sz} & {wins} \\\\" "\n"
                    )
                f.write(r"\bottomrule" "\n")
                f.write(r"\end{tabular}" "\n")
                f.write(
                    rf"\caption{{Paired comparison: {self._name(m1)} (M1) vs "
                    rf"{self._name(m2)} (M2), $\Delta = M2 - M1$. "
                    rf"$n={a.n}$. wins = (M1 wins):(M2 wins).}}"
                    "\n"
                )
                f.write(rf"\label{{tab:pairwise-{m1}-{m2}}}" "\n")
                f.write(r"\end{table}" "\n")

    def _write_scorecard_table(self):
        a = self.a
        path = self.out / "tab_scorecard.tex"
        score_dims = [
            ("Kills",         "KILLS"),
            ("S1 = K/T",      "S1_KillRate"),
            ("S2 = K/A_stam", "S2_KillPerStamina"),
            ("S3",            "S3_EffectiveKillOutput"),
            ("r_HP",          "r_HP"),
            ("r_Sigma",       "r_Stamina"),
            ("r_D",           "r_Damage"),
            ("M_HP",          "M_HP"),
            ("M_Sigma",       "M_Stamina"),
            ("M_D",           "M_Damage"),
        ]
        with open(path, "w") as f:
            f.write(r"\begin{table}[h]" "\n" r"\centering" "\n" r"\small" "\n")
            f.write(r"\begin{tabular}{l" + "c" * len(a.names) + r"}" "\n")
            f.write(r"\toprule" "\n")
            f.write("Dimension & " + " & ".join(self._name(n) for n in a.names) + r" \\" "\n")
            f.write(r"\midrule" "\n")
            for label, k in score_dims:
                row = [label]
                for n in a.names:
                    # normalize metric across models to 0-100
                    vals = [a.frames[m][k].mean() for m in a.names]
                    lo, hi = min(vals), max(vals)
                    s = 50.0 if hi == lo else (a.frames[n][k].mean() - lo) / (hi - lo) * 100
                    row.append(f"{s:.1f}")
                f.write(" & ".join(row) + r" \\" "\n")
            f.write(r"\bottomrule" "\n" r"\end{tabular}" "\n")
            f.write(rf"\caption{{Normalized scorecard across {len(a.names)} checkpoints. "
                    rf"Each row is min-max normalized so the best checkpoint scores 100.}}"
                    "\n")
            f.write(r"\label{tab:scorecard}" "\n" r"\end{table}" "\n")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE MAKER
# ═══════════════════════════════════════════════════════════════════════════
class FigureMaker:
    def __init__(self, a: Analysis, out_dir: Path):
        self.a = a
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
        self.colors = {n: palette[i % len(palette)] for i, n in enumerate(a.names)}

    def _save(self, fig, stem):
        fig.savefig(self.out / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(self.out / f"{stem}.png", dpi=200, bbox_inches="tight")

    def write_all(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            print("  [skip figures] matplotlib not installed")
            return
        self._fig_base()
        self._fig_derived()
        self._fig_composite()
        self._fig_radar()
        self._fig_pareto()
        if len(self.a.pairs) == 1:
            self._fig_forest_single_pair()
        else:
            self._fig_forest_multi()
        print(f"  Figures written to {self.out.resolve()}")

    def _fig_base(self):
        plt = self.plt
        a = self.a
        metrics = ["T_ALIVE", "KILLS", "A_HP", "A_STAMINA", "A_DAMAGE"]
        labels = [METRIC_DISPLAY.get(m, m) for m in metrics]
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
        for ax, k, lbl in zip(axes, metrics, labels):
            means = [a.frames[n][k].mean() for n in a.names]
            sds   = [a.frames[n][k].std(ddof=1) for n in a.names]
            colors = [self.colors[n] for n in a.names]
            x = np.arange(len(a.names))
            ax.bar(x, means, yerr=sds, color=colors, capsize=6,
                   edgecolor="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([LATEX_NAME_MAP.get(n, n) for n in a.names])
            ax.set_title(lbl)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
        fig.suptitle(f"Base metrics over $n={a.n}$ paired episodes",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        self._save(fig, "fig1_base")
        plt.close(fig)

    def _fig_derived(self):
        plt = self.plt
        a = self.a
        metrics = ["S1_KillRate", "S2_KillPerStamina",
                   "S3_EffectiveKillOutput",
                   "r_HP", "r_Stamina", "r_Damage"]
        labels = [METRIC_DISPLAY.get(m, m) for m in metrics]
        fig, axes = plt.subplots(1, 6, figsize=(20, 4.2))
        for ax, k, lbl in zip(axes, metrics, labels):
            means = [a.frames[n][k].mean() for n in a.names]
            sds   = [a.frames[n][k].std(ddof=1) for n in a.names]
            colors = [self.colors[n] for n in a.names]
            x = np.arange(len(a.names))
            ax.bar(x, means, yerr=sds, color=colors, capsize=6,
                   edgecolor="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([LATEX_NAME_MAP.get(n, n) for n in a.names])
            ax.set_title(lbl)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
        fig.suptitle(f"Derived metrics ($n={a.n}$)", fontsize=13, y=1.02)
        fig.tight_layout()
        self._save(fig, "fig2_derived")
        plt.close(fig)

    def _fig_composite(self):
        plt = self.plt
        a = self.a
        metrics = ["M_HP", "M_Stamina", "M_Damage"]
        labels = [METRIC_DISPLAY.get(m, m) for m in metrics]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, k, lbl in zip(axes, metrics, labels):
            means = [a.frames[n][k].mean() for n in a.names]
            sds   = [a.frames[n][k].std(ddof=1) for n in a.names]
            colors = [self.colors[n] for n in a.names]
            x = np.arange(len(a.names))
            ax.bar(x, means, yerr=sds, color=colors, capsize=6,
                   edgecolor="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([LATEX_NAME_MAP.get(n, n) for n in a.names])
            ax.set_title(lbl)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
        fig.suptitle(
            f"Composite metrics (anti-loafing) over $n={a.n}$ episodes",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        self._save(fig, "fig3_composite")
        plt.close(fig)

    def _fig_radar(self):
        plt = self.plt
        a = self.a
        axes_spec = [
            ("Kills",       "KILLS",               "higher"),
            ("S1 = K/T",    "S1_KillRate",         "higher"),
            ("S2 = K/A_s",  "S2_KillPerStamina",   "higher"),
            ("S3",          "S3_EffectiveKillOutput", "higher"),
            ("r_HP",        "r_HP",                "higher"),
            ("r_Sigma",     "r_Stamina",           "higher"),
            ("r_D",         "r_Damage",            "higher"),
            ("M_HP",        "M_HP",                "higher"),
            ("M_Sigma",     "M_Stamina",           "higher"),
            ("M_D",         "M_Damage",            "higher"),
        ]
        N = len(axes_spec)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        raw = {n: [a.frames[n][k].mean() for _, k, _ in axes_spec] for n in a.names}
        norm = {n: [] for n in a.names}
        for j, (_, _, direction) in enumerate(axes_spec):
            col = [raw[n][j] for n in a.names]
            lo, hi = min(col), max(col)
            for n in a.names:
                v = raw[n][j]
                if hi == lo:
                    s = 0.5
                elif direction == "higher":
                    s = (v - lo) / (hi - lo)
                else:
                    s = (hi - v) / (hi - lo)
                norm[n].append(s)

        fig, ax = plt.subplots(figsize=(7.5, 7.5),
                               subplot_kw={"projection": "polar"})
        for n in a.names:
            vals = norm[n] + norm[n][:1]
            ax.plot(angles, vals, linewidth=2,
                    label=LATEX_NAME_MAP.get(n, n), color=self.colors[n])
            ax.fill(angles, vals, alpha=0.15, color=self.colors[n])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([s[0] for s in axes_spec], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
        ax.set_title("Normalized comparison across all metrics", fontsize=11)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        self._save(fig, "fig4_radar")
        plt.close(fig)

    def _fig_pareto(self):
        plt = self.plt
        a = self.a
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for n in a.names:
            x = a.frames[n]["KILLS"].mean()
            y = a.frames[n]["M_Damage"].mean()
            ax.scatter(x, y, s=220, color=self.colors[n],
                       edgecolor="black", linewidth=0.7, zorder=3)
            ax.annotate(LATEX_NAME_MAP.get(n, n), (x, y),
                        textcoords="offset points", xytext=(8, 8), fontsize=11)
        ax.set_xlabel("Mean kills per episode")
        ax.set_ylabel(r"$M_D = K A_\mathrm{dmg}/T$")
        ax.set_title("Pareto frontier: kills vs composite damage output")
        ax.grid(alpha=0.3, linestyle=":")
        fig.tight_layout()
        self._save(fig, "fig5_pareto")
        plt.close(fig)

    def _fig_forest_single_pair(self):
        plt = self.plt
        a = self.a
        (m1, m2) = a.pairs[0]
        items = sorted(
            a.results[(m1, m2)].items(),
            key=lambda kv: abs(kv[1]["cohens_dz"])
            if not np.isnan(kv[1]["cohens_dz"]) else 0,
            reverse=True,
        )
        labels = [METRIC_DISPLAY.get(k, k) for k, _ in items]
        dzs    = [r["cohens_dz"] for _, r in items]
        cis    = [(r["ci95_low"], r["ci95_high"]) for _, r in items]
        y = np.arange(len(items))
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(items) + 1.5))
        for i, (dz, (lo, hi)) in enumerate(zip(dzs, cis)):
            color = "#d62728" if dz < 0 else "#2ca02c"
            ax.plot([lo, hi], [i, i], color=color, linewidth=2)
            ax.plot(dz, i, "o", color=color, markersize=8,
                    markeredgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(r"$\Delta$ (95% CI)")
        ax.set_title(f"{LATEX_NAME_MAP.get(m1, m1)} vs "
                     f"{LATEX_NAME_MAP.get(m2, m2)}")
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        fig.tight_layout()
        self._save(fig, "fig6_forest")
        plt.close(fig)

    def _fig_forest_multi(self):
        plt = self.plt
        a = self.a
        k = len(a.pairs)
        fig, axes = plt.subplots(1, k, figsize=(6.5 * k, 6), sharey=False)
        if k == 1:
            axes = [axes]
        for ax, (m1, m2) in zip(axes, a.pairs):
            items = sorted(
                a.results[(m1, m2)].items(),
                key=lambda kv: abs(kv[1]["cohens_dz"])
                if not np.isnan(kv[1]["cohens_dz"]) else 0,
                reverse=True,
            )
            labels = [METRIC_DISPLAY.get(kk, kk) for kk, _ in items]
            dzs    = [r["cohens_dz"] for _, r in items]
            y = np.arange(len(items))
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in dzs]
            ax.barh(y, dzs, color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel(r"Cohen's $d_z$")
            ax.set_title(f"{LATEX_NAME_MAP.get(m1, m1)} vs "
                         f"{LATEX_NAME_MAP.get(m2, m2)}")
            ax.grid(axis="x", alpha=0.3, linestyle=":")
            for yv, v in zip(y, dzs):
                ax.text(v + (0.05 if v >= 0 else -0.05), yv, f"{v:+.2f}",
                        va="center", ha="left" if v >= 0 else "right",
                        fontsize=8)
        fig.suptitle("Pairwise effect sizes", fontsize=12, y=1.02)
        fig.tight_layout()
        self._save(fig, "fig6_forest")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def _parse_models(args):
    if args.models:
        out = {}
        for spec in args.models:
            if "=" not in spec:
                raise SystemExit(f"--models entry must be name=path, got: {spec}")
            name, path = spec.split("=", 1)
            out[name.strip()] = path.strip()
        return out
    if not (args.m1 and args.m2):
        raise SystemExit("Provide --models name=path ... OR --m1 PATH --m2 PATH")
    return {args.name1: args.m1, args.name2: args.m2}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", nargs="+", default=None,
                    help="Space-separated name=path entries (e.g. P=... B=... F=...)")
    ap.add_argument("--m1", default=None)
    ap.add_argument("--m2", default=None)
    ap.add_argument("--name1", default="M1")
    ap.add_argument("--name2", default="M2")
    ap.add_argument("--n-first", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--latex", type=Path, default=None)
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--no-console", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    model_logs = _parse_models(args)

    if not args.quiet:
        print("DATA INTEGRITY")
    a = run_analysis(model_logs, alpha=args.alpha, n_boot=args.boot,
                     n_first=args.n_first, verbose=not args.quiet)

    if not args.no_console:
        ConsoleReporter(a).report()

    if args.latex is not None:
        hdr("LATEX EXPORT")
        LatexReporter(a, args.latex).write_all()

    if args.figures is not None:
        hdr("FIGURE EXPORT")
        FigureMaker(a, args.figures).write_all()

    if args.export is not None:
        args.export.mkdir(parents=True, exist_ok=True)
        rows = []
        for (m1, m2), per_metric in a.results.items():
            for metric, r in per_metric.items():
                rows.append({"pair": f"{m1}_vs_{m2}", "metric": metric, **r})
        pd.DataFrame(rows).to_csv(args.export / "pairwise_tests.csv", index=False)

        for (m1, m2) in a.pairs:
            joined = a.frames[m1].add_suffix(f"_{m1}").join(
                     a.frames[m2].add_suffix(f"_{m2}"))
            joined.to_csv(args.export / f"paired_{m1}_vs_{m2}.csv", index=False)

        with open(args.export / "summary.json", "w") as f:
            json.dump({
                "n": a.n, "models": a.names, "alpha": a.alpha,
                "metrics": ALL_METRICS, "pairs": [list(p) for p in a.pairs],
            }, f, indent=2)
        if not args.quiet:
            print(f"  CSV/JSON written to {args.export.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())