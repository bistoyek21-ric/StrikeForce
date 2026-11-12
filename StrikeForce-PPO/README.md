# StrikeForce

**StrikeForce** is a 2D battle-arena game and an experimental environment for autonomous agents. It combines a playable C++/SFML game with several AI-agent implementations, including behavioral cloning, imitation-learning variants, and a recurrent Proximal Policy Optimization (PPO) agent.

This repository currently selects **`bot-ppo`** as the active agent through `StrikeForce-client/selected_agent.hpp` and `selected_custom.hpp`. The PPO training and evaluation workflow is supported by the scripts in `tools/`.

> **Repository note:** This README describes the code and workflow included in this repository snapshot. Some optional components (for example online/API-based training) require separate services and are not needed for local PPO experiments.

---

## Contents

- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Build and run](#build-and-run)
- [Playing the game](#playing-the-game)
- [AI agents](#ai-agents)
- [PPO training workflow](#ppo-training-workflow)
- [Configuration](#configuration)
- [Evaluation and metrics](#evaluation-and-metrics)
- [Outputs and checkpoints](#outputs-and-checkpoints)
- [Troubleshooting](#troubleshooting)
- [Research and documentation](#research-and-documentation)
- [License and attribution](#license-and-attribution)

## Highlights

- 2D arena combat with multiple game modes, maps, weapons, items, and character configurations.
- C++17 game client and optional multiplayer server.
- Modular agent interface: switch the selected agent without rewriting the game engine.
- Multiple agent implementations in `StrikeForce-client/bots/`.
- Recurrent PPO training with Generalized Advantage Estimation (GAE), clipped policy updates, and value normalization.
- Scripted teacher policy and optional imitation-learning signals in the PPO implementation.
- Headless runners for faster experiments without rendering.
- Per-episode evaluation metrics, training logs, model snapshots, and comparison reports.
- Configurable reward shaping and PPO hyperparameters through a text configuration file.

## Repository structure

```text
StrikeForce-main/
├── README.md
├── INSTALLDEP.md
├── LICENSE.md
├── docs/
│   └── REWARD_SHAPING_AND_IMPLEMENTATION.md
├── tools/
│   ├── pipeline.py
│   ├── summarize_metrics.py
│   └── plot_training.py
└── StrikeForce-client/
    ├── main.cpp
    ├── gameplay.hpp
    ├── Character.hpp
    ├── GraphicPrinter.hpp
    ├── macros.hpp
    ├── selected_agent.hpp
    ├── selected_custom.hpp
    ├── StrikeForce-Lab-Protocol.md
    ├── bots/
    │   ├── bot-0/
    │   ├── bot-1/
    │   ├── bot-1.1/
    │   ├── bot-bc/
    │   ├── bot-bc-fap/
    │   └── bot-ppo/
    │       ├── Agent.hpp
    │       ├── Custom.hpp
    │       ├── Config.hpp
    │       ├── PPOMath.hpp
    │       ├── RewardShaping.hpp
    │       ├── Modules.hpp
    │       ├── config.txt
    │       └── presets/
    ├── map/
    ├── character/
    ├── Items/
    └── accounts/
```

The active agent is determined by the two selector headers:

- `selected_agent.hpp` includes the agent implementation.
- `selected_custom.hpp` includes the game/agent integration.

In this repository snapshot, both point to `bot-ppo`.

## Requirements

### Core build requirements

| Dependency | Purpose |
|---|---|
| C++17-compatible compiler (`g++` recommended) | Compile the game and agents |
| Python 3 | Run `tools/pipeline.py` and reporting utilities |
| SFML (graphics, window, system) | Required for the windowed game |
| LibTorch C++ distribution | Required by the neural-network agents, including `bot-ppo` |
| pthreads | Used by the headless build |

The headless PPO runner does not link SFML, but it still requires LibTorch.

### Linux / Ubuntu / WSL

Install the basic build tools and SFML:

```bash
sudo apt update
sudo apt install -y build-essential python3 python3-pip libsfml-dev
```

Download and extract a **Linux LibTorch** distribution from the official PyTorch website. Choose a build compatible with your compiler, C++ ABI, and CUDA setup. A CPU LibTorch build is sufficient if you do not intend to use GPU acceleration in the C++ application.

For example, if LibTorch is extracted to `$HOME/libtorch`, expose its headers and libraries:

```bash
export LIBTORCH="$HOME/libtorch"
export CPLUS_INCLUDE_PATH="$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$LIBTORCH/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
```

Add the exports to `~/.bashrc` if you want them to persist. Open a new terminal or run `source ~/.bashrc`.

The build helper also accepts extra compiler/linker flags through `TORCH_FLAGS`:

```bash
export TORCH_FLAGS="-I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -L$LIBTORCH/lib -Wl,-rpath,$LIBTORCH/lib"
```

Use paths that match your own LibTorch installation. Do not mix Windows LibTorch binaries with a Linux/WSL compiler.

### Windows and macOS

The source can be built with a compatible C++17 toolchain, SFML SDK, and LibTorch distribution. The compiler ABI must match the prebuilt SFML and LibTorch libraries. The commands in the PPO pipeline are written for `g++`; Windows users may find it easiest to use WSL2 with the Linux dependencies above.

For additional dependency notes, see [`INSTALLDEP.md`](INSTALLDEP.md).

## Build and run

All commands below assume that the current working directory is `StrikeForce-client/`, because the build and runtime paths are relative to that directory.

```bash
cd StrikeForce-client
```

### Build the windowed game

From `StrikeForce-client/`:

```bash
g++ -std=c++17 main.cpp -o StrikeForce \
  -lsfml-graphics -lsfml-window -lsfml-system \
  -ltorch -ltorch_cpu -lc10
```

Run it:

```bash
./StrikeForce
```

If your LibTorch installation is not in a standard compiler/library path, add the appropriate include and library flags or set `TORCH_FLAGS` as described above.

### Build through the helper script

The repository includes a helper for the PPO experiment workflow. Run it from `StrikeForce-client/`:

```bash
python3 ../tools/pipeline.py build
```

This produces the windowed executable `StrikeForce`.

To build a windowed executable with the agent frozen for evaluation:

```bash
python3 ../tools/pipeline.py build --eval
```

This produces `StrikeForce_eval` and defines `FREEZE_AGENT_BLOCK`.

### Build a headless runner

Headless mode disables rendering and is intended for training/evaluation batches:

```bash
python3 ../tools/pipeline.py build --headless
```

Output: `StrikeForce_headless`.

Build the frozen evaluation runner:

```bash
python3 ../tools/pipeline.py build --headless --eval
```

Output: `StrikeForce_headless_eval`.

Build the scripted expert/teacher runner:

```bash
python3 ../tools/pipeline.py build --headless --expert
```

Output: `StrikeForce_headless_expert`.

By default, the headless build enables `DATA_GATHERING_MODE`. To build without that macro:

```bash
python3 ../tools/pipeline.py build --headless --no-data-gathering
```

The headless pipeline uses optimization flags and links LibTorch. It does not need SFML.

## Playing the game

The exact controls/help text is available in the in-game help screen. The repository's documented controls include:

| Key | Action |
|---|---|
| `x` | Shoot with the selected weapon |
| `z` | Punch |
| `[` | Place a block |
| `]` | Place a portal |
| `f`, `g`, `h`, `j` | Select consumables |
| `k`, `l`, `;`, `'` | Select throwables |
| `c`, `v`, `b`, `n`, `m`, `,`, `.`, `/` | Select weapons |
| `u` | Use selected item |
| `0` | Show help |
| `-` | Open backpack |
| `F` / `O` | Fullscreen controls |
| `3` | Toggle manual/automatic agent mode |
| `Space` | Pause rendering in automatic mode |

Controls can vary by game state and build. Use the in-game help screen as the final reference.

## AI agents

Agent implementations are kept under `StrikeForce-client/bots/`.

| Agent folder | Description |
|---|---|
| `bot-0` | Minimal/example agent |
| `bot-1`, `bot-1.1` | Earlier learning-agent implementations, including reward-learning components |
| `bot-bc` | Behavioral-cloning implementation |
| `bot-bc-fap` | Behavioral cloning with the Future Action Prediction (FAP) approach and related research code |
| `bot-ppo` | Recurrent PPO implementation used by the current selectors |

To change the active agent, update both selector headers consistently:

```cpp
// selected_agent.hpp
#include "./bots/YOUR_AGENT/Agent.hpp"

// selected_custom.hpp
#include "bots/YOUR_AGENT/Custom.hpp"
```

The selected `Agent` implementation must provide the interface expected by the corresponding `Custom.hpp` and the game engine. Do not change only one selector header.

## PPO training workflow

The current PPO workflow is managed by `tools/pipeline.py`. The script must be run from `StrikeForce-client/`.

### 1. Build the training runner

```bash
python3 ../tools/pipeline.py build --headless
```

### 2. (Optional) Run the scripted teacher

The teacher is useful as a reference policy and can provide imitation targets to the PPO agent.

```bash
python3 ../tools/pipeline.py build --headless --expert
python3 ../tools/pipeline.py expert --episodes 20 --mode Solo --level 9 --max-steps 1000
```

### 3. Train the PPO agent

```bash
python3 ../tools/pipeline.py train --episodes 80 --mode Solo --level 9 --max-steps 1000
```

A longer run might look like:

```bash
python3 ../tools/pipeline.py train --episodes 500 --mode Solo --level 9 --max-steps 1000
```

The helper defaults to `--mode Solo --level 9` when these arguments are omitted. You can override them explicitly. Supported runner options documented by the helper include:

- `--episodes N`: number of games/episodes.
- `--mode MODE`: game mode.
- `--level L`: level/map difficulty.
- `--max-steps S`: maximum steps per episode.
- `--seed S`: seed for reproducibility.
- `--threads K`: number of worker threads.
- `--user NAME`: user identifier.
- `--quiet`: reduce console output.

For example:

```bash
python3 ../tools/pipeline.py train --episodes 100 --level 1 --threads 4 --seed 42
```

### 4. Preserve a copy before another experiment

The `keep` command copies the current PPO backup directory to a timestamped directory:

```bash
python3 ../tools/pipeline.py keep
```

This is useful before changing training settings or starting a new run. It does not replace a proper external backup.

### 5. Save a named model snapshot

After training has produced `model.pt`:

```bash
python3 ../tools/pipeline.py snapshot experiment_A
```

Snapshots are stored under:

```text
bots/bot-ppo/backup/snapshots/experiment_A/
```

A snapshot name must be unique. The command refuses to overwrite an existing snapshot.

### 6. Evaluate a snapshot

Build the frozen headless evaluation runner:

```bash
python3 ../tools/pipeline.py build --headless --eval
```

Then evaluate a named snapshot:

```bash
python3 ../tools/pipeline.py evaluate experiment_A \
  --episodes 20 --mode Solo --level 9 --max-steps 1000
```

The `evaluate` command stages the selected snapshot, runs the frozen evaluation binary, collects the metrics, restores the previous training state, and generates a report when the run succeeds.

Repeat with the same evaluation conditions for every model being compared:

```bash
python3 ../tools/pipeline.py evaluate experiment_B \
  --episodes 20 --mode Solo --level 9 --max-steps 1000
```

### 7. Generate comparison reports

```bash
python3 ../tools/pipeline.py report
```

Or report selected snapshots:

```bash
python3 ../tools/pipeline.py report experiment_A experiment_B
```

The report utility reads CSV files from `bots/bot-ppo/backup/results/`. If training statistics are present, the pipeline also generates a training plot.

### Manual snapshot evaluation commands

The pipeline's `evaluate` command is the recommended path. The lower-level commands are available when a researcher needs to manage the steps manually:

```bash
python3 ../tools/pipeline.py eval-prepare experiment_A
# Run the frozen evaluation executable with the desired arguments.
./StrikeForce_headless_eval --episodes 20 --mode Solo --level 9 --max-steps 1000
python3 ../tools/pipeline.py eval-collect experiment_A
python3 ../tools/pipeline.py report experiment_A
```

`eval-prepare` temporarily stashes the current training state and stages the selected snapshot. Always run `eval-collect` afterward so the original training state is restored. Avoid interrupting this sequence; if it is interrupted, inspect `bots/bot-ppo/backup/_stash/` before deleting or moving anything.

## Configuration

### PPO configuration file

Most PPO and reward-shaping parameters can be changed without recompiling:

```text
StrikeForce-client/bots/bot-ppo/config.txt
```

The file uses `key = value` syntax. Lines beginning with `#` are comments. Unknown keys are reported in `agent_log.log`.

The main parameter groups include:

| Group | Examples | Meaning |
|---|---|---|
| PPO | `ppo.rollout_len`, `ppo.num_epochs`, `ppo.gamma`, `ppo.gae_lambda`, `ppo.lr`, `ppo.ppo_clip` | Rollout size, optimization, discounting, and policy update settings |
| Exploration / imitation | `ppo.entropy_coef`, `ppo.bc_coef`, `ppo.bc_floor`, `ppo.bc_decay_updates`, `ppo.expert_mix` | Entropy regularization and teacher-guided behavior |
| Value learning | `ppo.value_coef`, `ppo.value_norm`, `ppo.value_norm_beta` | Critic loss and value normalization |
| Stability | `ppo.max_grad_norm`, `ppo.target_kl`, `ppo.normalize_adv` | Gradient clipping, KL stopping, and advantage normalization |
| Evaluation | `ppo.deterministic_eval`, `ppo.seed` | Deterministic action selection and seed |
| Reward shaping | `shaping.kill_bonus`, `shaping.death_penalty`, `shaping.area_scale`, `shaping.dealt_potential` | Event, survival/resource, and offensive reward terms |
| Reward shaping | `shaping.approach_potential`, `shaping.aim_potential`, `shaping.view_radius` | Potential-based approach/aim signals |
| Normalization | `shaping.hp_ref`, `shaping.stamina_ref`, `shaping.damage_ref`, `shaping.norm_cap` | Reference scales and clipping |

The shipped `config.txt` documents the available keys and their defaults. Uncomment or add a setting to override it. For example:

```text
ppo.lr = 0.0003
ppo.num_epochs = 4
ppo.deterministic_eval = 1
shaping.area_scale = 0.005
```

Keep a copy of the configuration used for each experiment. A checkpoint alone is not a complete experiment record: also preserve the configuration, code revision, training seed, and run arguments.

### Presets

The helper supports the following preset names if their files are present in `bots/bot-ppo/presets/`:

```bash
python3 ../tools/pipeline.py preset baseline_pure_ppo
python3 ../tools/pipeline.py preset stage1_imitation
python3 ../tools/pipeline.py preset stage2_ppo_finetune
```

Before applying a preset, the helper backs up the existing `config.txt` to a timestamped `config.txt.bak_*` file.

### Compile-time macros

`StrikeForce-client/macros.hpp` contains optional compile-time switches. Examples include:

- `FREEZE_AGENT_BLOCK`: freeze agent learning for evaluation in the windowed build.
- `AUTOMATIC`: run the agent automatically.
- `HIGHLY_OPTIMIZED`: enable optimized execution paths.
- `DATA_GATHERING_MODE`: change the Solo/data-gathering behavior.
- `CROWDSOURCED_TRAINING`: enable the optional crowdsourced-training integration, where supported.

Some pipeline build modes define macros on the compiler command line rather than relying on `macros.hpp`. If you change compile-time settings, rebuild the relevant executable.

## Evaluation and metrics

The baseline evaluation protocol is documented in [`StrikeForce-Lab-Protocol.md`](StrikeForce-client/StrikeForce-Lab-Protocol.md). Use the same conditions for all models in a comparison: game mode, level/map, opponents, initial conditions, maximum steps, action-selection mode, and number of episodes.

The PPO implementation records the following per-episode metrics:

| Metric | Meaning |
|---|---|
| `T_alive` | Number of frames the agent remained alive |
| `N_kills` | Number of opponents eliminated |
| `A_HP` | Sum of health values over the agent's alive frames |
| `A_stamina` | Sum of stamina values over the agent's alive frames |
| `A_damage` | Sum of damage-per-bullet capability over the agent's alive frames |
| `died` | Whether the episode ended in death |

The area metrics are trajectory-based sums, not just the final value at the end of an episode. Their interpretation and recommended reporting procedure are described in the Lab Protocol.

For a fair model comparison:

1. Train and save each model/checkpoint.
2. Use the same number of evaluation episodes for each checkpoint.
3. Use the same mode, level, step limit, and other environment settings.
4. Evaluate with the frozen evaluation runner, not a training runner.
5. Keep the per-episode CSV files and report means and standard deviations, along with win rate where applicable.

Do not compare training-run results directly with frozen evaluation results as if they were the same measurement. Training runs may update the policy during play and may use teacher mixing or other training-only behavior.

## Outputs and checkpoints

The main PPO artifacts are stored under:

```text
StrikeForce-client/bots/bot-ppo/backup/
├── agent_backup/
│   ├── model.pt
│   ├── optimizer.pt
│   ├── agent_log.log
│   └── checkpoints/
├── metrics.csv
├── training_stats.csv
├── snapshots/
└── results/
```

Typical files:

| File / directory | Description |
|---|---|
| `agent_backup/model.pt` | Current policy/value model checkpoint |
| `agent_backup/optimizer.pt` | Optimizer state used to resume training |
| `agent_backup/agent_log.log` | Configuration and PPO/episode log |
| `agent_backup/checkpoints/` | Periodic model checkpoints, such as `model_u10.pt` |
| `metrics.csv` | Per-episode metrics from the PPO agent |
| `training_stats.csv` | Per-update training statistics for plots |
| `snapshots/NAME/` | Named, frozen copies of model state |
| `results/NAME.csv` | Evaluation metrics for a named model |
| `results/NAME_agent.csv` | Agent-side metrics collected during evaluation |

Back up the whole `backup/` directory when you need to preserve an experiment. For reproducibility, also save `config.txt` and record the exact command, seed, compiler, and LibTorch version.

## Troubleshooting

### `pipeline.py` says to run from `StrikeForce-client/`

The helper expects the current directory to contain `bots/bot-ppo/`. Use:

```bash
cd StrikeForce-client
python3 ../tools/pipeline.py build --headless
```

### `StrikeForce_headless` or another runner is missing

Build the corresponding executable first:

```bash
python3 ../tools/pipeline.py build --headless
python3 ../tools/pipeline.py build --headless --eval
python3 ../tools/pipeline.py build --headless --expert
```

### `torch/torch.h: No such file or directory`

The compiler cannot find the LibTorch headers. Check that the include paths point to both:

```text
LIBTORCH/include
LIBTORCH/include/torch/csrc/api/include
```

### `cannot find -ltorch`, `-ltorch_cpu`, or `-lc10`

The linker cannot find LibTorch libraries. Add `LIBTORCH/lib` to the library search path and ensure `TORCH_FLAGS` includes the correct `-L` path.

### Runtime error: shared library not found

Ensure the LibTorch `lib` directory is available to the dynamic linker:

```bash
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
```

### SFML headers or libraries are missing

SFML is needed for the windowed executable, not the headless runner. Install the development package or configure the include/library paths for your SFML SDK.

### Evaluation does not use the intended checkpoint

Use the pipeline's named snapshot workflow:

```bash
python3 ../tools/pipeline.py snapshot my_checkpoint
python3 ../tools/pipeline.py build --headless --eval
python3 ../tools/pipeline.py evaluate my_checkpoint --episodes 20
```

Check the snapshot directory and the generated `results/my_checkpoint.csv`.

### Training appears not to update

Check `agent_backup/agent_log.log` and `training_stats.csv`. A short episode may not reach the configured rollout threshold. The PPO configuration includes `ppo.min_train_steps` to allow a sufficiently large partial rollout to be used at episode termination.

## Research and documentation

- [`StrikeForce-Lab-Protocol.md`](StrikeForce-client/StrikeForce-Lab-Protocol.md): baseline metrics and model-comparison protocol.
- [`docs/REWARD_SHAPING_AND_IMPLEMENTATION.md`](docs/REWARD_SHAPING_AND_IMPLEMENTATION.md): design rationale, reward terms, implementation notes, and verification tests for PPO reward shaping.
- [`INSTALLDEP.md`](INSTALLDEP.md): additional dependency installation notes.
- `StrikeForce-client/bots/bot-bc-fap/No-BPTT-via-FAP/`: implementation and paper-related material for Future Action Prediction and related behavioral-repair experiments.

### Reward design overview

The PPO reward combines several signals:

- kill reward for eliminating an opponent;
- death penalty for terminal failure;
- a dense metric-aligned term based on health, stamina, and damage capability while alive;
- potential-based terms that provide denser feedback for offensive progress and, where configured, approach/aim behavior.

The detailed equations, assumptions, parameter rationale, and limitations are documented in `docs/REWARD_SHAPING_AND_IMPLEMENTATION.md`. Reward shaping should be treated as an experimental design choice: compare variants under the same frozen evaluation protocol rather than assuming a shaped reward automatically improves performance.

## License and attribution

See [`LICENSE.md`](LICENSE.md) for the license included with this repository. The project README and source files contain the available author and research attribution information. Preserve applicable copyright and license notices when redistributing or modifying the code.

---

**For a first PPO experiment:** install compatible dependencies, enter `StrikeForce-client/`, build the headless runner, train the agent, save a named snapshot, build the frozen evaluation runner, and evaluate the snapshot using the same settings you will use for every comparison.
