#!/usr/bin/env python3
"""Plot training diagnostics from bots/bot-ppo/backup/training_stats.csv (and optionally metrics.csv).

Usage: python3 tools/plot_training.py [training_stats.csv] [metrics.csv] [-o training.png]
Panels: reward/return, losses, entropy, approx-KL & clip-fraction, explained variance, PopArt mu/sigma,
        and (if metrics.csv is given) the five Lab-Protocol metrics per episode.
"""
import csv, sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("(plot skipped: matplotlib is not installed -> sudo apt install python3-matplotlib)")
    sys.exit(0)

def col(rows, k): return [float(r[k]) for r in rows]
def load(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def smooth(y, w=5):
    out = []
    for i in range(len(y)):
        a = max(0, i - w + 1); out.append(sum(y[a:i + 1]) / (i + 1 - a))
    return out

def main(argv):
    out = "training.png"
    if "-o" in argv:
        i = argv.index("-o"); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    stats = load(argv[0] if argv else "bots/bot-ppo/backup/training_stats.csv")
    met = load(argv[1]) if len(argv) > 1 else None
    x = col(stats, "update")
    rows = 3 if met is None else 4
    fig, ax = plt.subplots(rows, 2, figsize=(12, 3.2 * rows))
    def line(a, keys, title, sm=False):
        for k, lab in keys:
            y = col(stats, k); a.plot(x, smooth(y) if sm else y, label=lab)
        a.set_title(title); a.set_xlabel("PPO update"); a.grid(alpha=.3); a.legend(fontsize=8)
    line(ax[0][0], [("mean_reward", "mean step reward"), ("return_mean", "mean return target")], "Reward / return", True)
    line(ax[0][1], [("policy_loss", "policy"), ("value_loss", "value (normalised)")], "Losses", True)
    line(ax[1][0], [("entropy", "entropy")], "Policy entropy (max = ln 9 = 2.197)")
    line(ax[1][1], [("approx_kl", "approx KL"), ("clip_frac", "clip fraction")], "Update size", True)
    line(ax[2][0], [("explained_var", "explained variance")], "Critic quality (1 = perfect)", True)
    line(ax[2][1], [("popart_mu", "mu"), ("popart_sigma", "sigma")], "PopArt statistics")
    if met is not None:
        e = list(range(1, len(met) + 1))
        for a, ks in ((ax[3][0], ["T_alive", "N_kills"]), (ax[3][1], ["A_HP", "A_stamina", "A_damage"])):
            for k in ks: a.plot(e, smooth(col(met, k)), label=k)
            a.set_title("Lab-Protocol metrics per episode (smoothed)"); a.set_xlabel("episode"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); print("saved", out)

if __name__ == "__main__":
    main(sys.argv[1:])
