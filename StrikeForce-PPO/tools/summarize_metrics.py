#!/usr/bin/env python3
"""Summarise bots/bot-ppo/backup/metrics.csv into the report table of StrikeForce-Lab-Protocol.md §11.

Usage:  python3 tools/summarize_metrics.py [metrics.csv ...]
Each CSV is treated as one model/checkpoint (run the same number of episodes for each — Protocol §12).
Reports n, win-rate (survived with >=1 kill), and mean ± std of T_alive, N_kills, A_HP, A_stamina, A_damage.
"""
import csv, statistics as st, sys

COLS = ["T_alive", "N_kills", "A_HP", "A_stamina", "A_damage"]

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

def main(paths):
    win_kills = 5                       # the game's rule: alive and kills >= 5 * level  (level 1 -> 5)
    if "--win-kills" in paths:
        i = paths.index("--win-kills"); win_kills = int(paths[i + 1]); paths = paths[:i] + paths[i + 2:]
    if not paths:
        paths = ["bots/bot-ppo/backup/metrics.csv"]
    tables = {p: load(p) for p in paths}
    print(f"{'Metric':<12}" + "".join(f"{p[-28:]:>34}" for p in paths))
    print(f"{'episodes':<12}" + "".join(f"{len(r):>34d}" for r in tables.values()))
    def win(r):
        if r and all(x['died'] == 'NA' for x in r): return None
        return sum(1 for x in r if x['died'] == '0' and float(x['N_kills']) >= win_kills) / max(1, len(r))
    def win1(r):
        if r and all(x['died'] == 'NA' for x in r): return None
        return sum(1 for x in r if x['died'] == '0' and float(x['N_kills']) >= 1) / max(1, len(r))
    print(f"{'win >=1 kill':<12}" + "".join(f"{('n/a' if win1(r) is None else format(win1(r), '.3f')):>34}" for r in tables.values()) + "   (old definition)")
    print(f"{'win >=%d kills' % win_kills:<12}" + "".join(f"{('n/a' if win(r) is None else format(win(r), '.3f')):>34}" for r in tables.values()))
    for c in COLS:
        cells = []
        for r in tables.values():
            m, s = ms([float(x[c]) for x in r]) if r else (float('nan'), 0.0)
            cells.append(f"{m:>22.2f} ± {s:<9.2f}")
        print(f"{c:<12}" + "".join(cells))

if __name__ == "__main__":
    main(sys.argv[1:])
