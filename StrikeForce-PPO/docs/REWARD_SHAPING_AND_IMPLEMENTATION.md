# StrikeForce — PPO with Shaped Reward

**Section 1 — Reward Shaping** · **Section 2 — Implementation**
Agent folder: `StrikeForce-client/bots/bot-ppo/` · Evaluation: `StrikeForce-Lab-Protocol.md`

---

## 0. Summary

| | |
|---|---|
| Algorithm | Recurrent PPO (clipped surrogate, GAE) — trained **only** on a hand-designed shaped reward |
| Reward | `r = r_kill + r_death + r_area + F` (sparse events + metric-aligned dense term + potential-based term) |
| Evaluation | The five Lab-Protocol metrics `T_alive, N_kills, A_HP, A_stamina, A_damage`, written to `metrics.csv` once per episode |
| Tuning | Every number lives in `Config.hpp` and can be overridden in `bots/bot-ppo/config.txt` without recompiling |
| Engine changes | 1 hook in `Character.hpp` (`deleteAgent`, guarded by `#if defined(SHAPED_REWARD)`), 2 one-line selectors |

**What "best value" means here.** The game could not be run in the environment where this was written, so no value below was
tuned on real training curves. Each default was *derived from a stated constraint* (§1.4, §2.3) and the reward-related ones were
checked numerically by the test-suite (§1.5). Treat them as a principled starting point and confirm the three most
influential ones with the ablation in §2.8.

---

# Section 1 — Reward Shaping

## 1.1 What the environment exposes

The agent's `predict(obs)` only receives the 32×31×31 observation. Two facts found while reading the code decide the design:

1. **Kills cannot be read from the observation.** `describe()` stores `get_kills() / 20` with *integer* division, so the
   channel is `0` until 20 kills. Therefore reward signals are read from the `Human` getters in `Custom.hpp` and passed to
   `Agent::observe()` (no change to the observation, so other bots/checkpoints are unaffected).
2. **`Human::get_damage()` is the cumulative damage dealt to enemies** (incremented in `gameplay.hpp` when the owner's bullet
   hits a hostile human, reset in `Human::reset()`). This is the dense offensive signal used by the potential term.

| Signal | Getter | Lab-Protocol symbol |
|---|---|---|
| HP | `get_Hp()` | `HP(t)` |
| Stamina | `get_stamina()` | `S(t)` |
| Damage per bullet currently available | `get_damage_effect()[0]` | `D(t)` (§9) |
| Cumulative damage dealt | `get_damage()` | — |
| Kills | `get_kills()` | `N_kills` |

## 1.2 The reward

For the transition `(s_t, a_t) → s_{t+1}` (signals measured at `s_{t+1}`):

```
r_t = r_kill + r_death + r_area + F
```

| Term | Formula | Purpose |
|---|---|---|
| `r_kill` | `kill_bonus · (kills_{t+1} − kills_t)` | the task objective, `N_kills` |
| `r_death` | `−death_penalty` if dead at `t+1` | terminal cost; makes `T_alive` matter beyond the area term |
| `r_area` | `area_scale · (w_hp·hp + w_st·st + w_dmg·dmg)` while alive | **dense, metric-aligned**: summed over an episode it is exactly a weighted `A_HP, A_stamina, A_damage` (§7–§9) and, being paid only while alive, it also rewards `T_alive` |
| `F` | `γ·Φ(s_{t+1}) − Φ(s_t)`, `Φ = dealt_potential · min(dealt / enemy_hp_ref, cap)`, `Φ(terminal)=0` | **potential-based** (Ng, Harada & Russell 1999): dense credit for hitting the enemy *without changing the optimal policy* |

with `hp = clip(HP/hp_ref, 0, 1.5)`, `st = clip(S/stamina_ref, 0, 1.5)`, `dmg = clip(D/damage_ref, 0, 1.5)`, and the total clipped to `±reward_clip`.

Why three kinds of term: sparse events alone give almost no learning signal in a 1-v-1 fight that lasts hundreds of frames;
the area term turns the evaluation metrics themselves into a per-frame signal (so the optimiser is aimed at what will be
measured); the potential term supplies shot-level credit and, by construction, cannot create a new "exploit" optimum.

## 1.3 Design guard-rails (all numerically verified, §1.5)

* **Camping must never beat fighting.** Each metric-aligned term pays out while the agent merely stays alive, so a
  hide-and-hoard policy earns income forever. The maximal discounted income is `area_scale / (1 − γ)` (all normalised
  signals at 1, weights summing to 1). The rule is `area_scale/(1−γ) < kill_bonus`.
* **Stamina and damage are resources that must be spent to fight.** Hence their weights are the smallest, so hoarding
  them can't outweigh the kill reward.
* **The potential term telescopes.** `Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0) = 0` for a terminal episode, so it can only speed up
  learning, never change what is optimal. `Φ` is bounded (`potential_cap`) as the theorem requires.
* **Bounded rewards** (`|r_t| ≤ 2`) keep value targets O(1).

## 1.4 Parameters and why these values

| Parameter | Default | Derivation |
|---|---:|---|
| `kill_bonus` | **1.0** | Sets the unit of the reward scale; everything else is expressed relative to it. |
| `death_penalty` | **1.0** | Symmetric with the kill in a zero-sum 1-v-1 (my death = their kill). |
| `area_scale` | **0.005** | Largest value of the sweep that keeps the survival ceiling `0.005/0.01 = 0.5` at **half** a kill, i.e. a 2× safety margin; 0.01 passes with a thin margin (ceiling 1.0 vs winner 1.15), 0.02+ loses to camping at `kill_bonus=1` (table in §1.5). |
| `w_hp / w_stamina / w_damage` | **0.5 / 0.2 / 0.3** | HP is the only quantity that ends the episode → largest; damage-per-bullet sets the time-to-kill → middle; stamina is a currency consumed by shooting → smallest (avoids reward for idling). Sum = 1 so `area_scale` is the true income ceiling. |
| `auto_ref` | **true** | **Errata (§5):** refs are now calibrated from the agent's own first-frame stats. The fixed 1000/1000/200/1000 below assumed `character/human.txt`, but the shipped account has HP 15000, damage 2500, stamina 15000. |
| `hp_ref` / `stamina_ref` | 1000 / 1000 | Only used when `auto_ref = false`. |
| `damage_ref` | 200 | Default damage per bullet is 100 (minimum 20); strong weapons are ≈ 175–225, so a typical agent sits near 0.5 and a well-armed one near 1. |
| `enemy_hp_ref` | 1000 | Default HP of an opponent → 1000 damage dealt ≙ one opponent's worth of health. |
| `dealt_potential` | **0.5** | Dealing a full opponent's HP is worth half a kill; the other half arrives with the kill itself, so offence is rewarded steadily but the kill stays the dominant event (kill + potential ≈ 1.5 < `reward_clip`). |
| `potential_cap` | 3.0 | Bounds `Φ ≤ 1.5` (≈ three opponents) — required for the invariance theorem. |
| `norm_cap` | 1.5 | Items can raise HP/stamina above defaults; capped so power-ups can't dominate. |
| `reward_clip` | 2.0 | Above the largest legitimate step (kill + potential ≈ 1.5). |
| `terminal_potential_zero` | true | Required for invariance at terminal states. |

## 1.5 Verification (`tests/test_reward_shaping.cpp`, compiled with g++ 13, all passing)

* **Metrics:** hand-computed episode gives exactly `T_alive = 3`, `A_HP = 2700`, `A_stamina = 2970`, `A_damage = 280`.
* **Policy invariance:** over 200 random episodes `max |Σ γ^t F_t| = 5.4·10⁻⁷` (float round-off).
* **Ordering** (discounted return): winner `0.988` > camper (1000 frames, never hit) `0.500` > loser `−0.979`.
  A perfect camper surviving 20 000 frames earns `0.500`, exactly the analytical ceiling.
* **Sensitivity sweep** (γ = 0.99; "camping" = a never-hit camper out-earns a winner):

| `area_scale` | `kill_bonus` | camper(∞) | winner | loser | winner > camper |
|---:|---:|---:|---:|---:|:--:|
| 0.001 | 1.0 | 0.100 | 0.859 | −1.085 | yes |
| **0.005** | **1.0** | **0.500** | **0.988** | **−0.979** | **yes (default)** |
| 0.010 | 0.5 | 1.000 | 0.873 | −0.848 | **NO** |
| 0.010 | 1.0 | 1.000 | 1.149 | −0.848 | yes (thin) |
| 0.020 | 1.0 | 2.000 | 1.472 | −0.584 | **NO** |
| 0.050 | 2.0 | 5.000 | 2.975 | 0.206 | **NO** |

Run it yourself: `g++ -std=c++17 -O2 tests/test_reward_shaping.cpp -o /tmp/t && /tmp/t` (full 15-row table is printed).

## 1.6 Known limitations of the reward

* The synthetic "winner/loser/camper" trajectories are hand-built; they validate the *incentive ordering*, not learning speed.
* `r_area` includes `T_alive` implicitly; if you want a pure kill-seeking agent, set `shaping.area_scale = 0`.
* Friendly fire / zombie damage does not enter `get_damage()` (only hostile humans do) — zombies give no offensive credit.

---

# Section 2 — Implementation

## 2.1 File map

```
StrikeForce-client/
  bots/bot-ppo/
    Config.hpp          all hyper-parameters (PPOConfig, ShapingConfig) + config.txt loader      [pure C++, unit-tested]
    RewardShaping.hpp   RewardShaper: reward, five lab metrics, metrics.csv writer               [pure C++, unit-tested]
    Agent.hpp           recurrent PPO agent                                                       [needs libtorch]
    Custom.hpp          observation builder (from bot-1) + reward-signal hand-off                 [needs libtorch]
    Modules.hpp         Backbone / AgentModel (copied unchanged from bot-1)
    AgentClient.hpp     crowdsourcing client (copied unchanged from bot-1)
    config.txt          run-time overrides (all keys listed, commented, with defaults)
  Character.hpp         + 3 lines in Human::deleteAgent()  (guarded by SHAPED_REWARD)
  selected_agent.hpp    -> bots/bot-ppo/Agent.hpp
  selected_custom.hpp   -> bots/bot-ppo/Custom.hpp
tests/test_reward_shaping.cpp      unit tests + sensitivity sweep (see also §3.5: PPOMath.hpp, test_ppo_math.cpp, pipeline.py)
tools/summarize_metrics.py         Lab-Protocol §11 report table from metrics.csv
```

## 2.2 Data flow (per frame)

```
bot(player)                       Custom.hpp
  ├─ obs = describe(31×31 window)
  ├─ agent->observe(hp, stamina, D, dealt, kills)   reward of the PREVIOUS action  → rewards_[t-1]
  └─ a_t = agent->predict(obs)                      (if rollout full: PPO update first)
gameplay: agent->update(a_t, is_human)              stores a_t
...
Human::deleteAgent()  ──► agent->on_terminal(...)   final reward, metrics row, last PPO update
```

Rewards are attached one frame late (the reward of `a_t` needs the state *after* it), so a rollout is complete when `T`
rewards have arrived; the bootstrap value `V(s_T)` is computed on the very next `predict`.

## 2.3 PPO and why these values

| Hyper-parameter | Default | Rationale |
|---|---:|---|
| `rollout_len` T | **512** | Same as bot-1; the full ordered rollout is replayed through the CNN+GRU, so cost grows linearly with T. |
| `min_train_steps` | 64 | The agent object lives for one game; short fights (death before 512 frames) would never train, so the final partial rollout is used if ≥ 64 steps. |
| `num_epochs` | 4 | One full-batch gradient step per epoch ⇒ ≤ 4 steps per 512 frames: conservative. Raise to 8–10 if learning is too slow; KL early-stop protects you. |
| `gamma` | 0.99 | Effective horizon ≈ 100 frames ≥ the ~60-frame engagement in §1.5, so a hit and the kill it enables are in one horizon. |
| `gae_lambda` | 0.95 | Standard bias/variance trade-off for GAE. |
| `lr` | 3·10⁻⁴ | Adam default for PPO (Schulman et al. 2017; Andrychowicz et al. 2021). |
| `ppo_clip` | 0.2 | Standard. |
| `entropy_coef` | 0.01 | Standard for ~10 discrete actions; keeps the policy from collapsing onto idle `+`. |
| `value_coef` | 0.5 | Standard. |
| `max_grad_norm` | 0.5 | Standard; protects the GRU from exploding gradients. |
| `target_kl` | 0.03 | Epoch loop stops when approx-KL > 0.045 — a safety net for full-batch updates. |
| `normalize_adv` | true | Advantages are standardised **after** the critic target `A_raw + V` is formed (bot-1 formed returns from normalised advantages, which is wrong). |

Other implementation choices:
* **Single Adam optimiser**, one loss `L = L_clip + c_v·L_V − c_e·H`, losses *averaged* over T.
* **No observation re-normalisation**: bot-1 divided by the std of per-frame channel *means*, which can blow sparse channels up;
  `Custom.hpp` already bounds inputs below 10.
* **Human takeover** (`flip`): frames with `is_human = true` are dropped from the rollout (off-policy).
* **Termination:** death ⇒ bootstrap value 0; quit/win ⇒ truncation, bootstrapped with the last value estimate.
* **Auxiliary heads** of `AgentModel` are present but unused (kept so checkpoints stay compatible with bot-1).

## 2.4 Changes to existing engine files

```diff
 // Character.hpp  — Human::deleteAgent()
 			if(!active_agent)
 				return;
+#if defined(SHAPED_REWARD)
+			agent->on_terminal(get_Hp(), get_stamina(), get_damage_effect()[0], get_damage(), get_kills());
+#endif
 			delete agent;
```
`SHAPED_REWARD` is defined by `bots/bot-ppo/Agent.hpp`, so other bots compile exactly as before.
`selected_agent.hpp` / `selected_custom.hpp` now include `bot-ppo` instead of `bot-bc-fap`.

## 2.5 Build and run

```bash
cd StrikeForce-client
g++ -std=c++17 main.cpp -o StrikeForce \
    -lsfml-graphics -lsfml-window -lsfml-system -ltorch -ltorch_cpu -lc10
./StrikeForce
```
Run from `StrikeForce-client/` (paths such as `bots/bot-ppo/backup` are relative).
For evaluation (no learning, no checkpoint overwrite) enable `#define FREEZE_AGENT_BLOCK` in `macros.hpp`.

## 2.6 Outputs

| File | Content |
|---|---|
| `bots/bot-ppo/backup/agent_backup/agent_log.log` | config line, one line per PPO update (mean reward, losses, entropy, KL, epochs run), one `EPISODE` line per game |
| `bots/bot-ppo/backup/metrics.csv` | per episode: `T_alive, N_kills, A_HP, A_stamina, A_damage, died` + undiscounted return per reward component |
| `bots/bot-ppo/backup/agent_backup/model.pt, optimizer.pt` | checkpoint |

Frame definition: `T_alive` counts frames in which the agent was alive and `bot()` ran; `A_*` are the plain sums of Lab-Protocol §7–§9.

## 2.7 Evaluating with the Lab Protocol

1. Train; keep checkpoints of each model to compare.
2. For each model: copy its checkpoint into `agent_backup/`, enable `FREEZE_AGENT_BLOCK`, play the **same number of episodes**
   under the same map/opponent/`deterministic_eval` setting (§12), clearing or renaming `metrics.csv` between models.
3. `python3 tools/summarize_metrics.py model_A.csv model_B.csv` prints the §11 table (episodes, win-rate, mean ± std of the five metrics).

## 2.8 Recommended ablation (confirm the defaults)

Run each line for the same number of updates and compare the lab metrics; change one key at a time in `config.txt`:

| Question | Keys to vary | Expect |
|---|---|---|
| Is the area term helping or inducing camping? | `shaping.area_scale = 0 / 0.005 / 0.01` | at 0.01, watch `T_alive↑` with `N_kills↓` |
| Is the potential term worth it? | `shaping.dealt_potential = 0 / 0.5 / 1.0` | faster rise in `A_damage`/kills early on |
| Update aggressiveness | `ppo.num_epochs = 4 / 8`, `ppo.lr = 1e-4 / 3e-4` | look at `approx_kl` in the log; should stay ≲ 0.03 |
| Horizon | `ppo.gamma = 0.97 / 0.99 / 0.995` | longer horizon helps survival, may slow credit assignment |

## 2.9 Verification status (superseded by §3.6 — read that table)

| Component | Status |
|---|---|
| `Config.hpp`, `RewardShaping.hpp` | compiled (`-Wall -Wextra`, clean) and unit-tested, 10 checks pass |
| `Agent.hpp` | **type/syntax-checked against a stub of the libtorch calls it uses, but not compiled against real libtorch and never run** (libtorch/SFML were unavailable) |
| `Custom.hpp`, `Character.hpp` hook | **not compiled** (need SFML/libtorch + full build); changes are small and mirror existing call patterns |
| Learning behaviour | **untested** — no training run was possible |

First thing to do on your machine: build, play one episode, check that `metrics.csv` gets a row and `agent_log.log` shows a `PPO#1` line.

## 2.10 Bugs found in the existing code (not fixed there, avoided here)

* `bots/bot-1/Agent.hpp` declares `flip_` twice in one member list (`bool … flip_` and `float … flip_ = false`) — it does not compile; its `Custom.hpp` also constructs `Agent(false, true)`, which maps to `T = 1`.
* `bots/bot-1` observation normalisation divides by the std of per-frame channel means (see §2.3).
* `bots/bot-1` computes critic returns from *normalised* advantages.
* `describe()` stores `get_kills() / 20` as integer division (§1.1).


---

# Section 3 — Value Head (critic) and Training Pipeline

## 3.1 What was already there, and what changed

The inherited `AgentModel` already contained a value head (`ResB → Linear(H,1)` on the shared feature `z`), so nothing was
"missing". For *this* reward it had four weaknesses, each fixed below:

| Weakness of the old critic | Why it matters here | Fix |
|---|---|---|
| Unbounded output regressed on raw returns | Return scale depends on `kill_bonus`, `area_scale`, `γ`; changing a reward weight silently changes the critic's difficulty and the value-loss gradient size | **PopArt value normalisation** (§3.2) |
| Default (random) initialisation of both heads | Large random logits at step 0 → low-entropy, noisy start | **Orthogonal init**: policy output gain 0.01 (≈ uniform policy), value hidden gain √2, value output gain 1 |
| One thin layer after the trunk | Limited capacity to separate "about to win" from "about to die" | Head is now `ResB → Linear(H,H/2) → ReLU → Linear(H/2,1)` |
| No visibility into critic quality | Impossible to tell if PPO fails because of the critic | **Explained variance**, value loss, clip-fraction, KL logged every update |

## 3.2 PopArt, precisely

The critic outputs `v` and the raw-scale value is `V = μ + σ·v`.
After each rollout, with returns `R_t = A_t + V_t` (computed on the raw scale):

1. Update statistics: `μ ← (1−β)μ + β·mean(R)`, `ν ← (1−β)ν + β·mean(R²)`, `σ ← sqrt(max(ν − μ², σ_min²))`.
   The first update uses `β = 1` (adopt the data statistics), later ones `β = value_norm_beta = 0.1` per update
   (≈ the per-sample rate 3·10⁻⁴ of the PopArt paper accumulated over 512 samples).
2. **Output-preserving rescale** of the last critic layer, so no prediction changes:
   `W' = W·σ/σ'`, `b' = (σ·b + μ − μ')/σ'`  ⇒  `μ' + σ'·(W'x + b') = μ + σ·(Wx + b)`.
3. Regress the head on `(R_t − μ')/σ'` with MSE (unit-scale targets).
4. GAE always uses raw-scale values (`μ + σ·v`), so reward weights keep their meaning.

`σ_min = 0.01` prevents the targets from exploding when a rollout has almost constant returns.
Statistics are persisted in `agent_backup/train_state.txt` so resumed training continues seamlessly.
Disable with `ppo.value_norm = 0` (the head then regresses raw returns, as before).

**Not included (deliberately):** value-loss clipping (Andrychowicz et al. 2021 found it unhelpful) and a separate critic
backbone (doubles the cost of the 512-step replay). Both are easy to add if explained variance stays low.

## 3.3 How to read the critic diagnostics

| Metric (in `training_stats.csv`) | Healthy | Problem → action |
|---|---|---|
| `explained_var` | rises towards 0.5–0.9 | stays ≤ 0 for many updates → critic not learning: lower `ppo.lr`, raise `ppo.value_coef` to 1.0, or increase `num_epochs` |
| `value_loss` (normalised units) | falls, ~0.1–1 | stays ≫ 1 → returns are mostly unpredictable noise; check reward scale |
| `approx_kl` | ≲ 0.03 | consistently > 0.045 → epochs stop early; lower `ppo.lr` |
| `clip_frac` | 0.05–0.3 | ≈ 0 → updates too timid (raise `lr`/`num_epochs`); > 0.5 → too aggressive |
| `entropy` | starts ≈ ln 9 = 2.197, declines slowly | collapses to < 0.3 within a few updates → raise `entropy_coef` |
| `popart_sigma` | settles | keeps growing → reward is non-stationary or exploding |

## 3.4 The pipeline

```
 ┌────────┐  ┌─────────────────────────────────────────────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐
 │ build  │→ │ TRAIN (game loop)                               │→ │ snapshot │→ │ EVALUATE  │→ │ report │
 │ pipeline│  │ observe → predict → update → … → PPO update     │  │  NAME    │  │ frozen    │  │ table +│
 │ .py     │  │ writes: agent_log.log, training_stats.csv,      │  │          │  │ N episodes│  │ plots  │
 │ build   │  │ metrics.csv, model.pt, checkpoints/model_u<k>.pt│  │          │  │ per model │  │        │
 └────────┘  └─────────────────────────────────────────────────┘  └──────────┘  └───────────┘  └────────┘
```

**Inside one PPO update** (`Agent::train`, in this order — the order matters):

1. `gae()` on raw-scale rewards and values → advantages `A`, critic targets `R = A + V`.
2. Diagnostics: explained variance of the *pre-update* critic.
3. PopArt: update `(μ,σ)`, rescale last critic layer, normalise `R`.
4. Standardise `A` (after `R` was formed — standardising first would corrupt the critic target).
5. `num_epochs` × { replay rollout from zero GRU memory → clipped surrogate + `value_coef`·MSE − `entropy_coef`·H →
   backward → clip grad-norm → Adam step → stop if approx-KL > 1.5·`target_kl` }.
6. Log line, `training_stats.csv` row, checkpoint every `save_every_updates`, clear buffers.

**Script commands** (run from `StrikeForce-client/`):

| Command | Does |
|---|---|
| `python3 ../tools/pipeline.py build [--eval]` | compiles `StrikeForce` or `StrikeForce_eval` (`-DFREEZE_AGENT_BLOCK`); extra flags via `$TORCH_FLAGS` |
| `... snapshot A` | copies `agent_backup/` → `backup/snapshots/A/` (without logs/periodic checkpoints) |
| `... eval-prepare A` | stashes the live training state, stages snapshot `A`, resets `metrics.csv` |
| `... eval-collect A` | moves `metrics.csv` → `backup/results/A.csv`, **restores the training state** |
| `... report [A B ...]` | Lab-Protocol §11 table (episodes, win-rate, mean ± std of the five metrics) + `training.png` |
| `python3 ../tools/plot_training.py` | training curves only |

The game window must still be played/run by you (it is an SFML GUI); everything around it is scripted.
For a fair comparison play the **same number of episodes** for every snapshot (Lab-Protocol §12).

## 3.5 New / changed files

| File | Change |
|---|---|
| `bots/bot-ppo/PPOMath.hpp` | **new** — `gae`, `explained_variance`, `ValueNorm`, PopArt reference formula (pure C++) |
| `bots/bot-ppo/Modules.hpp` | critic head redesigned, `init_heads()`, `popart_rescale()`, `value_out()` (no longer identical to bot-1) |
| `bots/bot-ppo/Agent.hpp` | PopArt, diagnostics, `training_stats.csv`, `train_state.txt`, periodic checkpoints |
| `bots/bot-ppo/Config.hpp`, `config.txt` | + `value_norm`, `value_norm_beta`, `value_sigma_min`, `save_every_updates`, `stats_csv` |
| `tests/test_ppo_math.cpp` | **new** — 10 checks |
| `tools/pipeline.py`, `tools/plot_training.py` | **new** |

## 3.6 Verification status (superseded by §4.7)

| Component | Status |
|---|---|
| `Config.hpp`, `RewardShaping.hpp`, `PPOMath.hpp` | compiled `-Wall -Wextra`; **20 unit checks pass** (10 reward + 10 math: GAE for λ=0 and λ=1, explained variance, PopArt preserves outputs to 3.6·10⁻¹⁵ over 500 random layers) |
| `tools/*.py` | functionally tested on fake CSV/checkpoint files: snapshot → eval-prepare → eval-collect restores the training state, report and plot are produced |
| `Agent.hpp` | syntax/type-checked against a **stub** of the libtorch calls it uses; **never compiled against real libtorch, never run** |
| `Modules.hpp` head changes | **not compiled.** Uses `Sequential::at<LinearImpl>(i)`, `torch::nn::init::orthogonal_`, in-place `mul_/add_/div_` on parameters under `NoGradGuard` — standard libtorch API, but unverified here |
| `Custom.hpp`, `Character.hpp` hook, `pipeline.py build` | not compiled/run (need SFML + libtorch) |
| Learning behaviour | untested — no training run was possible |

**Most likely compile-time surprises** (if any): the `Sequential::at<>` template call in `Modules.hpp`, and
`.to(torch::kFloat32)` after the comparison in the clip-fraction line. Both have one-line fixes; send the compiler message.


---

# Section 4 — Windowless, highly optimised runner (no graphics, only the result)

## 4.1 Why and what

The SFML window does not open on your machine, and a window is not needed to train or evaluate. The game already has two
switches for this — `HIGHLY_OPTIMIZED` (no rendering) and `AUTOMATIC` (skips the prompts) — but it still needed SFML to link,
a login, the menu and keyboard polling. The new **headless runner** removes all of that:

```
./StrikeForce_headless --episodes 500 --level 1 --threads 4
```
plays 500 games back-to-back with no window, no menu, no keyboard and no frame limiter, and prints **one result table**
(`survival rate`, mean ± std of `T_alive, N_kills, A_HP, A_stamina, A_damage`, shaped return, throughput), which is also saved to
`bots/bot-ppo/backup/results/headless_<time>.txt`. **SFML is not required at all** to build it.

## 4.2 Build and run (3 commands)

```bash
cd StrikeForce-client
export TORCH_FLAGS="-I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -L$LIBTORCH/lib -Wl,-rpath,$LIBTORCH/lib"   # your libtorch
python3 ../tools/pipeline.py build --headless            # -> StrikeForce_headless       (training)
python3 ../tools/pipeline.py build --headless --eval     # -> StrikeForce_headless_eval  (frozen weights)

python3 ../tools/pipeline.py train    --episodes 500 --level 1 --threads 4    # train, prints the result table
python3 ../tools/pipeline.py evaluate run1 --episodes 100 --level 1           # snapshot -> frozen run -> table + training.png
```
Direct build line (what `pipeline.py` runs): `g++ -std=c++17 -O3 -march=native main.cpp -o StrikeForce_headless -pthread
-DHIGHLY_OPTIMIZED -DAUTOMATIC -DHEADLESS_RUN -DHEADLESS_SET_THREADS -DHEADLESS_TRAIN $TORCH_FLAGS -ltorch -ltorch_cpu -lc10`
(`-DHEADLESS_EVAL` instead of `-DHEADLESS_TRAIN` for the frozen evaluation binary; these flags override any `FREEZE_AGENT_BLOCK` left in `macros.hpp`). Run it from `StrikeForce-client/`. The account `1` shipped in `accounts/game/` is
used for the loadout (`--user NAME` to change).

Runner options: `--episodes N` `--level L` `--max-steps S` (cap per game in **agent steps**, default 3000; a capped game counts as
"survived") `--seed S` (game *e* uses map seed `S+e`) `--threads K` `--user NAME` `--out DIR` `--quiet`.

**Fair comparison for free:** every model is evaluated on the *same* `S..S+N−1` map seeds, so differences between models are not
map luck (Lab-Protocol §12).

## 4.3 What was optimised (all behind `HEADLESS_RUN`, other builds unchanged)

| Change | Where | Effect |
|---|---|---|
| SFML headers/`sf::sleep` no longer needed when `HIGHLY_OPTIMIZED` | `GraphicPrinter.hpp`, `basic.hpp` | builds and runs with no display and no SFML |
| `render_it()` / `find_recom()` return immediately | `gameplay.hpp` | removes a thread spawn and a 30×100 map + `Human` copy **per frame**, and ~27 000 loop iterations per call |
| `cls()` (a `system("clear")` call) is a no-op | `basic.hpp` | no shell spawn, no escape codes in the output |
| No keyboard polling | `gameplay.hpp::my_command` | the agent chooses every action; no `kbhit()` on a closed stdin |
| Game text silenced (`cout` → null buffer) and no account/ranking file writes after each game | `headless_run`, `play()` | only the result table is printed; no disk I/O per game |
| Observation builder rewritten (one preallocated buffer, empty-cell fast path) | `bots/bot-ppo/Custom.hpp` | **bit-identical observations**, see §4.4 |
| Model + optimizer + critic statistics kept in memory for the whole batch, written every 20 games and at exit | `Agent.hpp` | no checkpoint load/save between games (previously twice per game) |
| `-O3 -march=native`, torch threads set from `--threads` | build flags | |
| Deterministic episodes | `headless_tb` seed | reproducible runs |

## 4.4 Measured speed-up and proof that nothing changed

Measured on this machine with the **real game loop** and a random-action stub in place of libtorch (5 games × 1 500 agent steps,
identical seeds):

| Version | agent steps / s |
|---|---:|
| windowless, original observation builder (`-O2`) | 1 453 |
| + optimised observation builder (`-O2`) | 2 221 |
| + empty-cell fast path (`-O2`) | 2 805 |
| + `-O3 -march=native` | **3 396** |

= **2.3× faster game engine**. A checksum over every observation of all 5 games is identical (`879317.849…, 1754325.674…, …`) for
the original and the optimised builder, and all game outcomes (kills, A_HP, A_damage) are identical, so the optimisation did not
change what the agent sees. These numbers exclude the neural network: with the real agent, the CNN+GRU forward pass (and the
4×-replayed PPO update) will dominate, so use `--threads` and tune `ppo.num_epochs` for speed.

## 4.5 Limitations

* **Solo mode only.** `Timer` mode ends on wall-clock time, which is meaningless at full speed; `Squad` was not exercised.
* The cap `--max-steps` truncates a game; truncated games are reported as *survived* and bootstrap the critic (as designed in §2.3).
* Every game starts from the same account loadout (the post-game `me = hum[ind]` copy is skipped on purpose, so games are independent).
* A windowed build (`pipeline.py build`) is unchanged and still needs SFML.

## 4.6 New / changed files

| File | Change |
|---|---|
| `StrikeForce-client/HeadlessSummary.hpp` | **new** — option parser, null stream, result table (pure C++) |
| `StrikeForce-client/main.cpp` | `headless_main()` behind `HEADLESS_RUN` |
| `StrikeForce-client/gameplay.hpp`, `basic.hpp`, `GraphicPrinter.hpp` | the `HEADLESS_RUN` / `HIGHLY_OPTIMIZED` guards of §4.3 |
| `bots/bot-ppo/Custom.hpp`, `Agent.hpp`, `Config.hpp`, `config.txt` | faster observations; process-wide model; `save_every_episodes` |
| `tools/pipeline.py` | `build --headless`, `train`, `evaluate` |
| `tests/test_headless_summary.cpp`, `tests/stub_agent/Agent.hpp` | **new** |

## 4.7 Verification status (current — supersedes §2.9 and §3.6)

| Component | Status |
|---|---|
| Real game loop, headless, random-action stub agent | **compiled (no SFML, no libtorch) and run end-to-end**: 20 games, no crash, deterministic, result table printed; `pipeline.py train` and `evaluate` (snapshot → frozen run → restore → report) verified on it |
| Observation optimisation | bit-identical to the original (checksum, §4.4) |
| `Config.hpp`, `RewardShaping.hpp`, `PPOMath.hpp`, `HeadlessSummary.hpp` | compiled `-Wall -Wextra`; **30 unit checks pass** (10 reward, 10 PPO math, 10 headless) |
| `Agent.hpp` (libtorch) | syntax/type-checked against a **stub** of the libtorch calls it uses; **never compiled against real libtorch, never run** |
| `Modules.hpp` head changes | **not compiled** (`Sequential::at<LinearImpl>`, `orthogonal_`, in-place ops on parameters) |
| Learning behaviour, real training speed | **untested** |

First run on your machine: `build --headless`, then `train --episodes 20 --max-steps 500`; you should get a result table and
`bots/bot-ppo/backup/training_stats.csv` with a few rows. If the compiler complains, send the first error message.


---

# Section 5 — Analysis of `agent_log.log` and errata

## 5.1 What the uploaded file is (and is not)

It is **not a weights file** and not from the PPO bot. It is the text log of **64 finished games of the older `bot-bc-fap`
agent running in evaluation mode** (`Inference mode : 1`, 3 382 220 parameters, model read from
`bots/bot-bc-fap/backup/model.pt`). No learning happened in it, and nothing in it can be "optimised" — the weights live in
`model.pt`. What it *does* provide is a measured baseline and some hard facts about the game, parsed with
`tools/parse_agent_log.py` into `bcfap_baseline_metrics.csv` (same columns as bot-ppo's `metrics.csv`, converted from game
frames to agent steps because `T_ALIVE` is exactly 2.0004 × `Total Actions` in every game).

| Baseline: bot-bc-fap, 64 games | mean ± std |
|---|---:|
| `T_alive` [agent steps] | 5 496 ± 2 960 (min 1 798, max 18 113) |
| `N_kills` | 188.5 ± 88.9 (min 39, max 451) |
| `A_HP` [per step] | 4.94·10⁷ ± 2.97·10⁷ |
| `A_stamina` | 5.75·10⁷ ± 2.05·10⁷ |
| `A_damage` | 1.19·10⁷ ± 0.79·10⁷ |
| average HP / stamina / damage-per-bullet per frame | 8 876 / 11 089 / 2 075 |

The spread is huge (coefficient of variation ≈ 50 %), so the apparent rise between the first and last fifth of the file
(`T_alive` 8.2k → 13.5k frames) is **noise, not learning** — the model never changed. To compare PPO against this baseline use
the same level/mode and at least as many games, and raise the cap (`--max-steps 20000`), otherwise truncation lowers `T_alive`.

## 5.2 Why it "took very long and got stuck"

* **Very long games, no cap.** Games last 3 600–36 200 frames and have 39–451 kills (zombies count as kills). Nothing in the old
  code ends a game except death.
* **Real-time pacing.** When the game renders, each frame is padded to 40 ms (25 fps; `lim` in `gameplay.hpp`). 64 games ×
  ≈ 11 000 frames ≈ 7 hours of wall-clock at 25 fps — this is by design, not a hang, unless you toggled silent mode with Space.
* **A genuinely unfinished game.** The file contains 65 game headers but 64 result blocks: the last game printed its header and
  never its result — it was killed or hung. The log cannot say which; a game with no cap and a window that stops responding
  (your display problem) is the most likely cause.
* The headless runner removes the pacing and the window and adds `--max-steps`, so a game can no longer run unbounded.

## 5.3 Bug found because of this log (fixed)

The log shows average HP ≈ 8 900, stamina ≈ 11 100 and damage ≈ 2 100 — not the 1000 / 1000 / 100 that I had assumed from
`character/human.txt`. The shipped account `1` really has **HP 15 000, damage 2 500, stamina 15 000**. My reward normalised with
fixed references (1000/1000/200), so on this account every normalised signal sat at the clip value 1.5 permanently: the metric-aligned
`r_area` term became a **constant** (no gradient, camper income 0.75 instead of 0.5) and the damage potential was mis-scaled.

Fix: `shaping.auto_ref` (default **on**) calibrates `hp_ref`, `stamina_ref`, `damage_ref = 2·D₀` and `enemy_hp_ref` from the
agent's own first-frame stats in every game, so the reward is identical for any account. Test: a trajectory with 15× larger
HP/stamina and 25× larger damage now gives exactly the same return (0.9876 vs 0.9876); with fixed references it does not.
`enemy_hp_ref` assumes opponents have the agent's HP; if enemies are weaker, set `shaping.auto_ref = 0` and tune the four
`*_ref` values (§1.4 table).

## 5.4 What you can do next with the log / weights

1. Compare PPO against the baseline: `python3 tools/summarize_metrics.py bcfap_baseline_metrics.csv bots/bot-ppo/backup/results/run1.csv`.
2. If you want the *weights* inspected or converted, send `bots/bot-bc-fap/backup/model.pt` (and the game level/mode used for this log).
3. Run evaluations with the headless runner and a cap, e.g. `pipeline.py evaluate run1 --episodes 64 --level 9 --max-steps 20000`
   (level 9 is what the project's `AUTOMATIC` mode uses; replace by the level you used for the log).


---

# Section 6 — Troubleshooting: "episodes=0" and how long to train

## 6.1 The `episodes=0` result (first real run) — cause and fix

Symptom: `./StrikeForce_headless --episodes 2 ...` printed `episodes=0` and all zeros after 16 s.
**Cause:** the first version of the runner built its table from `bots/bot-ppo/backup/metrics.csv`, a file that only the
`bot-ppo` agent writes. The 16 s show the games *were* played, but if the build still used another agent
(`selected_agent.hpp` / `selected_custom.hpp` pointing to `bot-bc-fap`, which is what the original repository ships), nothing wrote
that file, so the table was empty. (The same would happen if the hook in `Character.hpp` was not applied.)

**Fix (this version):**
* The **engine itself records every game** (`headless_tick()` / `headless_finish()` in `gameplay.hpp`), so the table is filled for
  *any* agent — you can also evaluate `bot-bc-fap` with identical metric definitions.
* The table prints `agent : …` and says explicitly when the build is not `bot-ppo`; the compiler prints
  `HEADLESS_RUN: the selected agent is not bots/bot-ppo` in that case.
* If no game finishes, the runner prints `WARNING: no game finished`.
* Every run also saves `results/headless_<time>.csv` (same columns as `metrics.csv`) and `.txt`.

Check which agent you built: `grep -n include selected_agent.hpp selected_custom.hpp | tail -2` — both must mention `bots/bot-ppo`.
For PPO training you additionally need the `Character.hpp` hook (`grep -n SHAPED_REWARD Character.hpp`).

## 6.2 How long to train

Measured by you: 2 games × 200 steps = 400 agent steps in 16.5 s ≈ **40 ms per step** with the network forward pass.
PPO training costs, per collected step, one forward pass plus `num_epochs` × (forward + backward ≈ 3 forwards), so roughly
**13× the inference cost with 4 epochs, 7× with 2 epochs** (the KL early-stop can shorten it):

| `ppo.num_epochs` | seconds per training step (estimate) | 10 000 steps | 20 000 steps |
|---:|---:|---:|---:|
| 4 (default) | ≈ 0.5 | ≈ 1.4 h | ≈ 2.8 h |
| 2 (recommended for CPU) | ≈ 0.3 | ≈ 0.8 h | ≈ 1.6 h |

These are extrapolations from one measurement — run the 3-game pilot first and replace 40 ms by what you see.
**Expectation:** ~20 000 steps (≈ 40 updates) is enough to see the learning signal (explained variance rising above 0, entropy
slowly falling, `T_alive`/`N_kills` moving), not a strong agent; PPO from scratch normally needs hundreds of thousands of steps.

## 6.3 Recommended commands

```bash
cd StrikeForce-client
# 0. once: faster updates on CPU
echo "ppo.num_epochs = 2" >> bots/bot-ppo/config.txt

# 1. pilot: 3 games, measures real speed (read "wall time" in the table)
python3 ../tools/pipeline.py train --episodes 3 --max-steps 300 --threads 4

# 2. baseline of the (almost untrained) agent
python3 ../tools/pipeline.py evaluate pilot --episodes 10 --max-steps 500

# 3. main run: 40 games x 500 steps = 20 000 steps = 40 PPO updates
python3 ../tools/pipeline.py train --episodes 40 --max-steps 500 --threads 4

# 4. evaluate again (same seeds, same cap) and compare
python3 ../tools/pipeline.py evaluate run1 --episodes 10 --max-steps 500
python3 ../tools/pipeline.py report pilot run1
```
Use `--threads` = number of physical cores. Run step 3 again to continue training (it resumes from the checkpoint); look at
`bots/bot-ppo/backup/training.png` (`pipeline.py report`) — if `explained_var` stays ≤ 0 or entropy collapses, see §3.3.


## 6.4 Second real run: three findings (all fixed)

Your WSL log showed (a) a result table in the **old** format (no `agent :` line, still a `shaped return` row) — the binary was
built from the previous source tree, so the engine-side recording of §6.1 was not in it; (b) `train --episodes 40 --max-steps 500`
finishing 20 000 steps in 254 s ≈ **13 ms per step, i.e. inference speed — the agent did not learn at all**; (c) `evaluate` aborting
because the eval binary had not been built, and `report` crashing on a missing file / missing matplotlib.

| Finding | Likely cause | Fix in this version |
|---|---|---|
| no learning during `train` | the agent ran **frozen**: a `#define FREEZE_AGENT_BLOCK` left in `macros.hpp` (you had evaluation-mode runs before) makes `training_ = false` | the training binary is built with `-DHEADLESS_TRAIN`, which **undefines** `FREEZE_AGENT_BLOCK`; the eval binary with `-DHEADLESS_EVAL`; the result table now prints `agent : bot-ppo ... | TRAINING, PPO updates so far: N` or `FROZEN` |
| empty table | table was read from an agent-written CSV | table now comes from the engine (§6.1) |
| `evaluate` / `report` errors | eval binary missing; no result file; no matplotlib | `evaluate` checks the binary **before** touching any state; `report` names the missing file; the plot is skipped (with a hint) when matplotlib is absent |

**How to tell you are on the new build:** the table starts with `agent : bot-ppo (PPO + shaped reward) | TRAINING, PPO updates so far: …`.
After a `train` run the number of PPO updates must be > 0 and `bots/bot-ppo/backup/training_stats.csv` must exist.
If it says `FROZEN`, `macros.hpp` is overriding you — rebuild with the new `pipeline.py` (it passes `-DHEADLESS_TRAIN`).

## 6.5 WSL tips

* **Copy the project into the Linux filesystem** (`cp -r /mnt/c/Users/.../StrikeForce-PPO ~/StrikeForce && cd ~/StrikeForce/StrikeForce-main/StrikeForce-client`).
  `/mnt/c` is a network-style mount: every map load, log line and checkpoint write is several times slower there.
* `sudo apt install python3-matplotlib` for the training plot (optional).
* A CUDA build of libtorch (`+cu117`) runs on CPU only here; it works, but the CPU build is smaller and loads faster.
* Remove stale snapshots from the failed runs before re-using the names:
  `rm -rf bots/bot-ppo/backup/snapshots/pilot bots/bot-ppo/backup/snapshots/run1 bots/bot-ppo/backup/_stash`.
* Order matters: build **both** binaries first (`build --headless` and `build --headless --eval`), then `train`, then `evaluate`.


---

# Section 7 — Why kills were low, and what was changed (from `PPO_DIAGNOSTICS_FOR_CLAUDE.txt`)

## 7.1 What the diagnostics show

| Observation (41 training games × 500 steps, 30 evaluation games) | Meaning |
|---|---|
| 10 kills in 41 training games (0.24/game); 2 kills in 30 evaluation games (0.07/game), 2 deaths | essentially random-agent level (random stub on the same game: 0.5–1.2 kills per 1000 steps) |
| `A_HP` ≈ 99 % and `A_stamina` 100 % of the maximum in almost every game | the agent is **passive**: it is never hit because it never fights; the four non-kill metrics are already saturated |
| `return_total` = 2.12 in 31 of 41 games | that is exactly the survival income `500 × 0.005 × 0.85`; nothing else happened, so those rollouts contain **no learning signal** |
| `return_std` 0.01–0.03 in no-event rollouts, 0.14 when a kill happened | advantage standardisation divided by ~0.02 → noise amplified ×50: **entropy fell 2.19 → 0.55 in 35 updates**, `clip_frac` up to 0.98 — the policy chased noise |
| `explained_var` −1 … −9 | an artefact: EV is a ratio to Var(return), which is ≈ 0 in no-event rollouts; only the −0.0…−0.03 values (games with kills) are meaningful, and those say "critic ≈ constant" |
| `approx_kl` ≤ 0.034, 4 epochs every time | updates were not too large; the problem is the signal, not the step size |
| **Solo level 1 ends after 5 kills** (`level*5 <= kills` in `check_end`) | `N_kills` can never exceed 5 there; the bc-fap log (up to 451 kills) cannot be Solo |
| bc-fap log: max `T_ALIVE` 36 226 < 37 500 = 25 min × 60 s × 25 fps | the bc-fap games were most probably **Timer mode, level 5** (kills uncapped); its rate is **34 kills per 1000 agent steps** (188.5 kills / 5 496 steps) |

**New measurement (random agent, real game loop, this version):** a hostile (zombie or enemy human) is visible in the 31×31
window in **94 %** of the steps at mean distance 6.8 cells, and in the line of fire in about **5 %** of the steps. So enemies are
*not* hard to find — the agent simply never learned to **turn towards them and shoot**. That decides the fix.

## 7.2 Changes

| # | Change | Why | Where |
|---|---|---|---|
| 1 | **Aim potential** `Φ_aim = 0.15` when a hostile is in the line of fire (**errata §8.1: corrected to "adjacent, in the facing direction" — the agent cannot shoot, only punch**) | dense signal for the missing skill (turn, then shoot); base rate 5 % ⇒ frequent feedback | `Custom.hpp`, `RewardShaping.hpp` |
| 2 | **Approach potential** `Φ_app = 0.10·(15−d)/15` for the nearest hostile at Manhattan distance *d* | gradient towards enemies | same |
| | Both are **potentials** (`F = γΦ' − Φ`, `Φ(terminal)=0`): policy-invariant; unit test: `max |Σγᵗ F| = 1.3·10⁻⁷` over 200 random episodes; bounded by 0.25 | cannot create a "farm the shaping" optimum | `tests/test_reward_shaping.cpp` |
| 3 | **Advantage floor** `ppo.adv_std_floor = 0.1`: advantages ÷ max(std, 0.1) | 0.1 ≈ the std of a rollout that contains a kill (observed 0.14); weaker rollouts are no longer inflated to unit variance | `Agent.hpp` |
| 4 | `ppo.entropy_coef` 0.01 → **0.02** | sparse-reward exploration; entropy had collapsed | `Config.hpp` |
| 5 | **Win-screen hang fixed** | at 5 kills in Solo the game waited for a Space key — the windowless run would have **frozen forever** the first time the agent won. Every "press space" wait is now skipped in headless builds | `gameplay.hpp` (`SF_WAIT_SPACE`) |
| 6 | **Timer mode in headless runs** (`--mode Timer`): length counted in frames (level × 3750 agent steps) instead of wall-clock | uncapped kills, long games, same mode as the bc-fap baseline; the original code ended Timer games immediately at full speed | `gameplay.hpp` (`timer_up`) |
| 7 | Result table prints **kills / 1000 steps** and the bc-fap reference (34) | comparable across modes and caps | `HeadlessSummary.hpp` |

Not changed (deliberately): `kill_bonus` 1.0, `death_penalty` 1.0, `area_scale` 0.005 — the survival income is a constant offset
and does not disturb PPO's advantages; the metric-aligned term is kept so the other four lab metrics stay saturated.

## 7.3 Recommended runs

Train in **Timer** mode (uncapped kills, enemies accumulate over time) and evaluate in the same mode:
```bash
# rebuild first (new sources): python3 ../tools/pipeline.py build --headless ; python3 ../tools/pipeline.py build --headless --eval
rm -rf bots/bot-ppo/backup                      # start clean: the old checkpoint learned from the noisy signal
python3 ../tools/pipeline.py train --episodes 3 --mode Timer --level 1 --max-steps 1000 --threads 4     # pilot / speed
python3 ../tools/pipeline.py train --episodes 40 --mode Timer --level 1 --max-steps 1000 --threads 4    # ~40 000 steps
python3 ../tools/pipeline.py evaluate run2 --episodes 10 --mode Timer --level 1 --max-steps 1000
```
Success looks like: `kills / 1000 steps` clearly above the random level (~1) and rising run over run (bc-fap: 34),
`entropy` falling slowly (not below ~1.0 in the first 40 updates), `explained_var` ≥ 0 in games with kills.
If kills are still ≈ 0 after ~40 000 steps, raise `shaping.aim_potential` to 0.3 and train longer; if the agent starts
dying a lot, lower `shaping.approach_potential` to 0.05.

## 7.4 Verification status

| Component | Status |
|---|---|
| Targeting signals, Timer mode, win-screen fix | **compiled and run on the real game loop** (random-action stub): Timer game ended by itself at the expected length; an instant-win Solo game returned instead of waiting for a key; signal statistics as quoted in §7.1 |
| Reward / potential maths | 16 unit checks (incl. telescoping of the new terms), all pass |
| Effect on real learning (kills ↑) | **not measured** — needs your training run; the changes target the three causes found in the diagnostics, they do not guarantee a result |
| `Agent.hpp` / `Modules.hpp` | as before: syntax-checked against a libtorch stub; your WSL build compiled them |


---

# Section 8 — Result of your run, root cause, and the teacher (scripted policy) that fixes it

## 8.1 Your run (43 000 training steps, 86 PPO updates, Timer level 1, 1000-step games)

| | kills / 1000 steps | win rate (alive & ≥ 5 kills) |
|---|---:|---:|
| random actions (measured on the real game loop) | ≈ 1.2 | 0 |
| **your trained PPO** (training 1.35, frozen evaluation 1.50) | **1.4** | 0.60 under the old, looser rule (alive & ≥ 1 kill); ≈ 0 under the game's rule |
| bot-bc-fap (from the log, most probably Timer L5) | 34 | — |

PPO learned nothing about kills in 43 000 steps. `T_alive` = 1000 and HP ≈ 93 % in every game: it survives by staying passive. Reading the movement and
combat code showed why the skill is hard to find **and why my previous shaping was aiming at the wrong thing**:

1. **The agent cannot shoot.** The action set is `"+xzqeawsd"`. `x` shoots only when a weapon is selected (`backpack.vec` 1 or 2), but
   `vec = -1` at the start and **no key in the set selects a weapon**. Measured: forcing `x` on aligned enemies gives 0 damage dealt and 0 kills.
   The only damage is the melee punch `z`, weapon range 1 (an enemy must stand in the **adjacent cell in the facing direction**).
   → *Errata:* the "aim potential" of §7 rewarded lining up a gun that cannot fire. It now rewards **an enemy adjacent in the facing direction**.
2. **Moving does not turn.** `a/s/d/w` move, `q/e` rotate, so killing needs: walk to an enemy → turn to face it → `z`. Three separate skills.
3. **Walls.** A scripted policy that walked straight at the nearest enemy had **76 % of its moves blocked** (7 237 of 9 547) and made 10 kills / 1000 steps.
   Random exploration has no chance.

## 8.2 The teacher

`Custom.hpp` now contains a scripted policy that uses the engine state (the same information the observation encodes):
(1) enemy in the facing cell → `z`; (2) enemy adjacent elsewhere → `q`/`e`; (3) otherwise **breadth-first search over the visible 31×31 grid**
with the game's own walkability rule (`? ^ v . X *`) to the nearest cell that touches an enemy, take the first step; (4) nothing reachable → wander.

Measured on the real game loop (`pipeline.py expert`, 10 games, same seeds as the other numbers):

| Scripted teacher | kills / 1000 steps | survival | win rate |
|---|---:|---:|---:|
| Timer level 1, 1000 steps | **40.6** | 0.90 | **0.90** |
| Solo level 1 (5 kills ends the game) | 25.0 (5 kills in ≈ 200 steps) | 1.00 | **1.00** |

That is above the bc-fap reference (34) and about 30× your PPO. Build it with `pipeline.py build --headless --expert`.

## 8.3 How PPO now learns from the teacher (DAgger + decaying imitation)

* Every visited state is labelled with the teacher's action (`predict(obs, expert_idx)`).
* **Loss** = PPO + `λ_bc · CE(policy, teacher action)`, `λ_bc = bc_coef · max(bc_floor, 1 − update/bc_decay_updates)` (1.0 → 0.05 over 150 updates).
* During training the agent **executes** the teacher's action with probability `expert_mix · (1 − update/bc_decay_updates)` (0.5 → 0); those steps are **excluded from the PPO policy
  loss** (not on-policy) but their rewards still train the critic. Advantage statistics, KL and clip-fraction use on-policy steps only.
* The teacher fades out, so PPO is free to improve on it; the floor keeps a small anchor. Evaluation (frozen) never uses the teacher.
* Logged per update in `training_stats.csv`: `bc_loss`, `p_teacher` (mean probability the policy gives the teacher's action — should rise towards 1), `lambda_bc`, `expert_frac`.
* Config keys: `ppo.bc_coef`, `ppo.bc_floor`, `ppo.bc_decay_updates`, `ppo.expert_mix` (set `ppo.expert_mix = 0` for pure labels without executing the teacher).

## 8.4 Commands

```bash
cd StrikeForce-client            # Linux filesystem copy recommended (§6.5)
python3 ../tools/pipeline.py build --headless
python3 ../tools/pipeline.py build --headless --eval
python3 ../tools/pipeline.py build --headless --expert
rm -rf bots/bot-ppo/backup                       # the old checkpoint learned from noise

python3 ../tools/pipeline.py expert --episodes 20 --mode Timer --level 1 --max-steps 1000        # reference: ~40 kills / 1000 steps
python3 ../tools/pipeline.py train  --episodes 80 --mode Timer --level 1 --max-steps 1000 --threads 4
echo "ppo.deterministic_eval = 1" >> bots/bot-ppo/config.txt        # frozen evaluation plays the best action, not a sample
python3 ../tools/pipeline.py evaluate teach1 --episodes 20 --mode Timer --level 1 --max-steps 1000
python3 ../tools/pipeline.py report teach1
```
Time: your run measured ≈ 50 agent-steps/s in training and ≈ 357 in evaluation, so 80 games × 1000 steps ≈ 27 minutes (≈ 2 PPO updates per game ⇒ ≈ 160 updates, i.e. the
whole 150-update teacher schedule).

Read the table: `kills / 1000 steps` and `win rate` should rise far above 1.4 / 0 (teacher: 40.6 / 0.90). In `training_stats.csv` watch `p_teacher` rise towards ≈ 1 and `bc_loss` fall.
If `p_teacher` stays low, lower the learning pressure from PPO (`ppo.lr = 0.0001`) or raise `ppo.bc_coef` to 2.

## 8.5 Verification status

| Component | Status |
|---|---|
| Teacher policy, targeting, win rule, Timer mode | **compiled and run on the real game loop** (random-action stub agent + `-DEXPERT_POLICY`): numbers in §8.2 |
| Teacher labels reach the agent every step | verified (label histogram: 0 steps without a label in 4 000) |
| Schedule / reward maths | 41 unit checks pass (reward 16, PPO math 13, headless 12) |
| **Imitation + PPO training code (`Agent.hpp`)** | **syntax/type-checked against a libtorch stub only — never compiled against real libtorch or run.** The loss lines use the same tensor operations as the existing PPO code. If your compile fails, send the first error |
| Whether the learned agent reaches teacher-level kills | **not measured** — depends on your run; the teacher's 40.6 is an upper reference for imitation, not a promise |


---

# Section 9 — v3: the requested setting (DATA_GATHERING_MODE, Solo, level 9), corrections, and a safer training order

## 9.0 Corrections to earlier statements in this document

| Where | Wrong statement | Correct statement (evidence) |
|---|---|---|
| §5.2, §7.1, §8.1 | the bc-fap log's last game "was killed or hung"; the log is "most probably Timer level 5" | `DATA_GATHERING_MODE` ends the process at game 65 on purpose (`gameplay.hpp` `SEED == 64 → exit(0)`) and **disables the Solo "5 × level kills" win** — the log is consistent with **Solo, level 9, DATA_GATHERING_MODE** |
| §8.4 / §7.3 | `rm -rf bots/bot-ppo/backup` | destroys `snapshots/` (the original PPO checkpoint). Use `pipeline.py keep` first and never delete `backup/` |
| §8.4 | enable `ppo.deterministic_eval = 1` together with the teacher run | two simultaneous changes; the offline experiment below shows sampling beats argmax for an imitator (7.1 vs 3.7 kills / 1000 steps) — leave it **0** |
| `pipeline.py expert` | printed the PPO banner (`FROZEN … updates 0`) | the teacher run is now labelled `SCRIPTED TEACHER (no neural network …)`. Your "Experiment C" was this run |
| win rate | 0.60 (alive & ≥ 1 kill) compared with 0.00 (alive & ≥ 5·level kills) | the table now prints **both** definitions on separate lines |

## 9.1 What v3 changes

* **Setting you asked for is the default.** `pipeline.py build --headless …` defines `-DDATA_GATHERING_MODE`; `train` / `evaluate` / `expert` default to
  `--mode Solo --level 9` (`--no-data-gathering` and `--mode/--level` override). Under the macro `--seed S` is honoured (game *e* uses seed `S+e`) and the
  64-game process exit is disabled for headless runs, so any number of games works and a long training run can no longer be killed before `flush()`.
* **Reporting fixes** (confirmed problems of the diagnostic report): honest agent banner for the teacher build, `model.pt` size / `deterministic_eval` / `probe_teacher` in the frozen banner,
  executed-action histogram, two win-rate lines, per-class `p_teacher` (punch / turn / move / idle) in every PPO log line, frozen-run probe (top action == teacher %, punch recall).
* **`ppo.policy_coef`** (default 1 = unchanged behaviour) and three presets (`pipeline.py preset NAME`):
  `baseline_pure_ppo` (teacher off = the original PPO, to restore a reliable baseline), `stage1_imitation` (policy loss off, pure DAgger), `stage2_ppo_finetune`.
* **Not changed:** network, observation, rewards, GAE, defaults of every other hyper-parameter. **Not added:** class-weighted imitation (the experiment below says it does not help).

## 9.2 What I measured (real game loop; stand-in agents, NOT your network)

Offline imitator: 166 compact features of the real observation (9×9 hostile map, 9×9 wall map, facing) → MLP 64 units, trained with plain cross-entropy on 51 000 teacher-labelled
states (teacher executed 50 %, random 50 %, 60 games; punch = 4.6 % of labels), evaluated on held-out games and then **closed-loop in the game** (new seeds).

| Held-out accuracy | plain CE | class-weighted CE |
|---|---:|---:|
| top-1 accuracy | 0.53 | 0.49 |
| **punch recall** | **0.68** | 0.73 |

| Closed loop, Timer L1, 1000 steps, seeds 101–120 | kills / 1000 | survival |
|---|---:|---:|
| random | 1.22 | 0.95 |
| imitator, sampling | **7.10** | 1.00 |
| imitator, argmax | 3.70 | 1.00 |
| imitator, class-weighted, sampling / argmax | 5.20 / 2.36 | 1.00 / 0.90 |
| scripted teacher | 42.90 | 1.00 |

| **DATA_GATHERING_MODE, Solo, level 9**, seeds 1–20, cap 3000 steps | kills / 1000 | alive at cap | mean kills / game |
|---|---:|---:|---:|
| random | 1.14 | 0.50 | 3.0 |
| imitator, sampling | **8.23** | 0.60 | 22.5 |
| imitator, argmax | 4.00 | 0.50 | 9.8 |
| scripted teacher | **46.19** | 0.05 | 102.5 |

Reading: (1) a trivially simple, properly trained imitator is already **6–7× better than random and does not die**; (2) **sampling beats argmax**; (3) **class weighting does not help**
(so "the rare punch class is ignored" is *not* supported — punch recall is 0.68 with plain CE); (4) your frozen policy (0.36 kills / 1000, survival 0.40) is **worse than this proxy**,
so the problem is in how the real network was trained (optimisation / the PPO term fighting the imitation term / too little imitation), not in the idea of imitating the teacher.

## 9.3 Safer training order (one change at a time; each stage has a pass criterion)

```bash
cd StrikeForce-client
python3 ../tools/pipeline.py keep                                   # preserve everything first
python3 ../tools/pipeline.py build --headless
python3 ../tools/pipeline.py build --headless --eval
python3 ../tools/pipeline.py build --headless --expert
python3 ../tools/pipeline.py expert --episodes 20 --max-steps 3000                 # reference (Solo L9, DATA mode): ~46 kills / 1000 steps

# Stage 0 — reliable baseline (teacher off = original PPO)
python3 ../tools/pipeline.py preset baseline_pure_ppo
python3 ../tools/pipeline.py train --episodes 40 --max-steps 1000
python3 ../tools/pipeline.py evaluate base0 --episodes 20 --max-steps 3000

# Stage 1 — imitation only (fresh start: move bots/bot-ppo/backup/agent_backup aside, do not delete)
python3 ../tools/pipeline.py preset stage1_imitation
python3 ../tools/pipeline.py train --episodes 60 --max-steps 1000
python3 ../tools/pipeline.py evaluate stage1 --episodes 20 --max-steps 3000
```
**Pass criterion for stage 1:** frozen, stochastic evaluation ≥ ~8 kills / 1000 steps in this setting (the offline imitator gets 8.2) and, in the frozen banner,
`top action == teacher` well above chance and punch recall well above 0.3. **If stage 1 fails, the bug is in the imitation path of the real network (`Agent.hpp` train()), not in PPO** — send the
`PPO#` log lines (they now contain `p_teacher` per label class) and `training_stats.csv`. Only after stage 1 passes: `preset stage2_ppo_finetune` and PPO fine-tuning (stage 2).

## 9.4 Verification status

| Item | Status |
|---|---|
| runner / pipeline / banner / DATA mode / presets / >64 games | compiled and run end-to-end on the real game loop with stand-in agents |
| offline imitation numbers (§9.2) | measured (numpy MLP + the real game), but a **proxy**: not your libtorch network |
| `Agent.hpp` / `Modules.hpp` (libtorch) | syntax-checked against a stub only — **never compiled or run here** |
| whether stage 1 / stage 2 reach the pass criteria with the real network | **not measured** — needs your machine |
