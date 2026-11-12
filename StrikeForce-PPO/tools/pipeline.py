#!/usr/bin/env python3
"""bot-ppo pipeline helper — build, snapshot, evaluate, report.

Run from StrikeForce-client/ ... all paths are relative to the directory you run it from,
exactly like the game itself (so `cd StrikeForce-client` first).

    python3 ../tools/pipeline.py build --headless [--eval] compile the WINDOWLESS runner (no SFML needed): StrikeForce_headless[_eval]
    python3 ../tools/pipeline.py preset NAME               copy bots/bot-ppo/presets/NAME.txt to config.txt (baseline_pure_ppo | stage1_imitation | stage2_ppo_finetune)
    python3 ../tools/pipeline.py keep                      copy the whole backup/ directory to backup_KEEP_<time> (do this before experiments)
    python3 ../tools/pipeline.py build --headless --expert  compile the SCRIPTED teacher policy as a player: StrikeForce_headless_expert
    python3 ../tools/pipeline.py expert [runner options]   run the scripted teacher (reference / upper bound for imitation)
    python3 ../tools/pipeline.py train [runner options]    run N games headless in training mode, print ONE result table
    python3 ../tools/pipeline.py evaluate NAME [options]   snapshot (if needed) -> frozen headless run -> restore -> result table + plot
    python3 ../tools/pipeline.py build [--eval]            compile the windowed game (--eval adds -DFREEZE_AGENT_BLOCK)
    python3 ../tools/pipeline.py snapshot NAME             copy agent_backup/ -> backup/snapshots/NAME/
    python3 ../tools/pipeline.py eval-prepare NAME         stash the training state, stage snapshot NAME for evaluation, reset metrics.csv
    python3 ../tools/pipeline.py eval-collect NAME         save metrics.csv -> backup/results/NAME.csv and restore the training state
    python3 ../tools/pipeline.py report [NAME ...]         Lab-Protocol §11 table over backup/results/*.csv  (+ training plot)

DEFAULT SETTING (what you asked for): headless builds define DATA_GATHERING_MODE (Solo without kill cap) and every run defaults to
   --mode Solo --level 9 (override with --mode / --level; build with --no-data-gathering to switch the macro off).
Runner options (train / evaluate): --episodes N --level L --max-steps S --seed S --threads K --user NAME --quiet
   e.g.   python3 ../tools/pipeline.py train --episodes 500 --level 1 --threads 4
          python3 ../tools/pipeline.py evaluate run1 --episodes 100 --level 1

Typical cycle (windowed):
    build → play/train → snapshot A → (more training) → snapshot B →
    build --eval → for each X in A B:  eval-prepare X → play N episodes with ./StrikeForce_eval → eval-collect X → report
"""
import os, shutil, subprocess, sys, glob

BK      = "bots/bot-ppo/backup"
AGENT   = f"{BK}/agent_backup"
METRICS = f"{BK}/metrics.csv"
SNAPS   = f"{BK}/snapshots"
RESULTS = f"{BK}/results"
STASH   = f"{BK}/_stash"
HERE    = os.path.dirname(os.path.abspath(__file__))

def need_client_dir():
    if not os.path.isdir("bots/bot-ppo"):
        sys.exit("error: run this from StrikeForce-client/ (the folder that contains bots/bot-ppo)")

def build(eval_mode, headless=False, expert=False, data_gathering=True):
    flags = os.environ.get("TORCH_FLAGS", "").split()
    if headless and data_gathering and "-DDATA_GATHERING_MODE" not in flags: flags.append("-DDATA_GATHERING_MODE")
    if headless:
        out = "StrikeForce_headless_expert" if expert else ("StrikeForce_headless_eval" if eval_mode else "StrikeForce_headless")
        cmd = ["g++", "-std=c++17", "-O3", "-march=native", "main.cpp", "-o", out, "-pthread",
               "-DHIGHLY_OPTIMIZED", "-DAUTOMATIC", "-DHEADLESS_RUN", "-DHEADLESS_SET_THREADS",
               "-DHEADLESS_EVAL" if (eval_mode or expert) else "-DHEADLESS_TRAIN"] + (["-DEXPERT_POLICY"] if expert else []) + ["-ltorch", "-ltorch_cpu", "-lc10"] + flags
    else:
        out = "StrikeForce_eval" if eval_mode else "StrikeForce"
        cmd = ["g++", "-std=c++17", "main.cpp", "-o", out] + (["-DFREEZE_AGENT_BLOCK"] if eval_mode else []) + \
              ["-lsfml-graphics", "-lsfml-window", "-lsfml-system", "-ltorch", "-ltorch_cpu", "-lc10"] + flags
    print(" ".join(cmd)); sys.exit(subprocess.call(cmd))

def with_defaults(args):
    """The requested evaluation setting: Solo, level 9 (together with -DDATA_GATHERING_MODE, see build())."""
    a = list(args)
    if "--mode" not in a:  a += ["--mode", "Solo"]
    if "--level" not in a: a += ["--level", "9"]
    return a

def preset(name):
    src = f"bots/bot-ppo/presets/{name}.txt"
    if not os.path.isfile(src): sys.exit(f"unknown preset {name}; available: " + ", ".join(sorted(os.path.basename(f)[:-4] for f in glob.glob("bots/bot-ppo/presets/*.txt"))))
    import time
    dst = "bots/bot-ppo/config.txt"
    if os.path.isfile(dst): shutil.copy(dst, f"{dst}.bak_{int(time.time())}")
    shutil.copy(src, dst); print(f"config.txt <- preset {name}  (previous config kept as config.txt.bak_*)")

def keep():
    import time
    dst = f"bots/bot-ppo/backup_KEEP_{time.strftime('%Y%m%d_%H%M%S')}"
    if not os.path.isdir(BK): sys.exit("nothing to keep: no backup directory")
    shutil.copytree(BK, dst); print("copied", BK, "->", dst)

def run_headless(binary, args):
    if not os.path.isfile(binary): sys.exit(f"{binary} not found — run:  pipeline.py build --headless{' --eval' if 'eval' in binary else ''}")
    return subprocess.call([os.path.abspath(binary)] + args, stdin=subprocess.DEVNULL)

def expert(args):
    sys.exit(run_headless("StrikeForce_headless_expert", with_defaults(args)))

def train(args):
    sys.exit(run_headless("StrikeForce_headless", with_defaults(args)))

def evaluate(name, args):
    import time
    if not os.path.isfile("StrikeForce_headless_eval"):      # check BEFORE touching any state
        sys.exit("StrikeForce_headless_eval not found. Build it first:\n    python3 ../tools/pipeline.py build --headless --eval")
    if not os.path.isdir(f"{SNAPS}/{name}"): snapshot(name)
    eval_prepare(name)
    t0 = time.time() - 1
    try:
        rc = run_headless("StrikeForce_headless_eval", with_defaults(args))
    finally:
        eval_collect(name)
    # the runner writes one CSV per run (recorded by the engine, valid for any agent)
    fresh = sorted([f for f in glob.glob(f"{RESULTS}/headless_*.csv") if os.path.getmtime(f) >= t0], key=os.path.getmtime)
    if fresh:
        os.makedirs(RESULTS, exist_ok=True)
        shutil.copy(fresh[-1], f"{RESULTS}/{name}.csv")
    if rc == 0: report([name])
    sys.exit(rc)

def snapshot(name):
    if not os.path.isfile(f"{AGENT}/model.pt"): sys.exit(f"no checkpoint at {AGENT}/model.pt — train first")
    dst = f"{SNAPS}/{name}"
    if os.path.exists(dst): sys.exit(f"snapshot {name} already exists")
    shutil.copytree(AGENT, dst, ignore=shutil.ignore_patterns("checkpoints", "agent_log.log"))
    if os.path.isfile(METRICS): shutil.copy(METRICS, f"{dst}/train_metrics_at_snapshot.csv")
    print("snapshot ->", dst)

def eval_prepare(name):
    src = f"{SNAPS}/{name}"
    if not os.path.isdir(src): sys.exit(f"unknown snapshot {name}")
    if os.path.exists(STASH): sys.exit(f"{STASH} exists — run eval-collect first (or delete it if you are sure)")
    os.makedirs(STASH)
    if os.path.isdir(AGENT): shutil.move(AGENT, f"{STASH}/agent_backup")
    if os.path.isfile(METRICS): shutil.move(METRICS, f"{STASH}/metrics.csv")
    shutil.copytree(src, AGENT, ignore=shutil.ignore_patterns("train_metrics_at_snapshot.csv"))
    print(f"staged snapshot {name} (evaluate: same number of episodes for every model, then eval-collect {name})")

def eval_collect(name):
    os.makedirs(RESULTS, exist_ok=True)
    if os.path.isfile(METRICS): shutil.move(METRICS, f"{RESULTS}/{name}_agent.csv"); print("agent metrics ->", f"{RESULTS}/{name}_agent.csv")
    if os.path.isdir(AGENT): shutil.rmtree(AGENT)
    if os.path.isdir(f"{STASH}/agent_backup"): shutil.move(f"{STASH}/agent_backup", AGENT)
    if os.path.isfile(f"{STASH}/metrics.csv"): shutil.move(f"{STASH}/metrics.csv", METRICS)
    shutil.rmtree(STASH, ignore_errors=True)
    print("training state restored")

def report(names):
    files = [f"{RESULTS}/{n}.csv" for n in names] if names else sorted(glob.glob(f"{RESULTS}/*.csv"))
    missing = [f for f in files if not os.path.isfile(f)]
    if missing: sys.exit("missing result file(s): " + ", ".join(missing) + "\n(run `evaluate NAME ...` first)")
    subprocess.call([sys.executable, os.path.join(HERE, "summarize_metrics.py")] + files)
    stats = f"{BK}/training_stats.csv"
    if os.path.isfile(stats):
        subprocess.call([sys.executable, os.path.join(HERE, "plot_training.py"), stats, "-o", f"{BK}/training.png"])
    else:
        print("(no training_stats.csv yet - the agent has not done a PPO update)")

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    need_client_dir()
    c = a[0]
    if   c == "build":        build("--eval" in a, "--headless" in a, "--expert" in a, "--no-data-gathering" not in a)
    elif c == "preset":       preset(a[1])
    elif c == "keep":         keep()
    elif c == "expert":       expert(a[1:])
    elif c == "train":        train(a[1:])
    elif c == "evaluate":     evaluate(a[1], a[2:])
    elif c == "snapshot":     snapshot(a[1])
    elif c == "eval-prepare": eval_prepare(a[1])
    elif c == "eval-collect": eval_collect(a[1])
    elif c == "report":       report(a[1:])
    else: sys.exit(__doc__)
