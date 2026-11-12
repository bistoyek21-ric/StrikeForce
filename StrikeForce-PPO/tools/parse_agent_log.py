#!/usr/bin/env python3
"""Convert an `agent_log.log` written by the older bot-bc-fap agent into a bot-ppo-style metrics.csv
so the two agents can be compared with tools/summarize_metrics.py.

Usage: python3 tools/parse_agent_log.py agent_log.log [-o bcfap_metrics.csv] [--frames-per-step 2]

The bc-fap log counts GAME FRAMES (T_ALIVE == 2 x "Total Actions", verified on every episode), while bot-ppo
counts AGENT STEPS (one per decision).  T_alive and the three area metrics are therefore divided by
--frames-per-step (default 2) so that both agents are reported in the same unit.
A header without metrics at the end of the file (a game that never finished) is reported, not counted.
"""
import argparse, csv, re, statistics as st, sys

def parse(text):
    eps, unfinished = [], 0
    for chunk in re.split(r"(?m)^={9}\n(?=Agent parameters)", text):
        if "Agent parameters" not in chunk: continue
        m = re.search(r"KILLS = (\d+), T_ALIVE = (\d+)", chunk)
        h = re.search(r"A_HP = (\d+), A_STAMINA = (\d+), A_DAMAGE = (\d+)", chunk)
        a = re.search(r"Total Actions\s*:\s*(\d+)", chunk)
        if not (m and h):
            unfinished += 1; continue
        f = lambda r: int(re.search(r, chunk).group(1)) if re.search(r, chunk) else None
        eps.append(dict(kills=int(m[1]), t=int(m[2]), hp=int(h[1]), st=int(h[2]), dmg=int(h[3]),
                        actions=int(a[1]) if a else None,
                        fail=f(r"Overall Failures\s*:\s*(\d+)"), succ=f(r"Overall Success\s*:\s*(\d+)")))
    return eps, unfinished

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("-o", default="bcfap_metrics.csv")
    ap.add_argument("--frames-per-step", type=float, default=2.0)
    a = ap.parse_args()
    eps, unfinished = parse(open(a.log, errors="replace").read())
    if not eps: sys.exit("no finished episodes found")
    k = a.frames_per_step
    with open(a.o, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("unix_time,T_alive,N_kills,A_HP,A_stamina,A_damage,died,return_total,return_kill,return_death,return_area,return_potential".split(","))
        for e in eps:
            w.writerow([0, e["t"] / k, e["kills"], e["hp"] / k, e["st"] / k, e["dmg"] / k, "NA", "", "", "", "", ""])
    ms = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
    print(f"finished episodes: {len(eps)}   unfinished (header without metrics): {unfinished}")
    if eps[0]["actions"]:
        r = st.mean(e["t"] / e["actions"] for e in eps)
        print(f"T_ALIVE / Total Actions = {r:.4f}  -> frames per agent step (used: {k})")
    for name, key, div in (("T_alive [steps]", "t", k), ("N_kills", "kills", 1), ("A_HP [per step]", "hp", k),
                           ("A_stamina", "st", k), ("A_damage", "dmg", k)):
        m, s = ms([e[key] / div for e in eps]); print(f"{name:18s}: {m:16.1f} +- {s:14.1f}")
    print(f"avg HP per frame  : {st.mean(e['hp'] / e['t'] for e in eps):.0f}    avg stamina: {st.mean(e['st'] / e['t'] for e in eps):.0f}"
          f"    avg damage/bullet: {st.mean(e['dmg'] / e['t'] for e in eps):.0f}")
    print("wrote", a.o)

if __name__ == "__main__":
    main()
