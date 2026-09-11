# StrikeForce — Baseline Evaluation and Model Comparison Protocol

## 1. Purpose

This document defines a baseline evaluation protocol for comparing agent models in the StrikeForce environment.

The protocol is intended to provide a common and reproducible starting point for experiments conducted in the environment. Researchers may add, remove, or modify metrics according to their research objective, but the evaluation described below should be regarded as the basic comparison setup.

The same evaluation conditions should be applied to all compared models.

---

## 2. Models Under Evaluation

An experiment may compare any number of models, checkpoints, architectures, or training procedures.

Examples include:

* a baseline checkpoint;
* a fine-tuned checkpoint;
* a repaired or edited checkpoint;
* two different architectures;
* different training algorithms;
* different versions of the same agent.

Each model should be evaluated independently under the same environment and evaluation conditions.

---

## 3. Baseline Metrics

The baseline metric set is:

$$
T_{\mathrm{alive}},
\qquad
N_{\mathrm{kills}},
\qquad
A_{\mathrm{HP}},
\qquad
A_{\mathrm{stamina}},
\qquad
A_{\mathrm{damage}}.
$$

The first two metrics measure survival duration and successful eliminations.

The remaining three are trajectory-based metrics computed from the area under the corresponding state or capability curve during the agent's lifetime.

---

## 4. Head-to-Head Evaluation Scenario

A standard evaluation scenario is a direct one-versus-one encounter.

Two agents are placed in an enclosed environment or designated combat area where they are unable to leave the encounter.

Each agent attempts to eliminate the other.

A successful elimination constitutes a win for the surviving agent.

The basic outcome is therefore:

$$
\mathrm{Win}_A =
\begin{cases}
1, & \text{if agent } A \text{ eliminates agent } B,\\
0, & \text{otherwise}.
\end{cases}
$$

The same definition applies to agent $B$.

A head-to-head experiment may be repeated over multiple episodes, with the initial conditions controlled or randomized according to the purpose of the experiment.

When comparing two models, each model should be evaluated under equivalent initial conditions whenever practical.

---

## 5. Survival Duration

### $T_{\mathrm{alive}}$ — Alive Frames

$T_{\mathrm{alive}}$ is the number of frames for which an agent remains alive during an episode.

$$
T_{\mathrm{alive}}
=
\text{number of frames during which the agent is alive}.
$$

A larger value means that the agent survived for a longer period.

For an evaluation set containing multiple episodes, the metric may be reported as the mean, median, standard deviation, confidence interval, or full per-episode distribution.

---

## 6. Kill Count

### $N_{\mathrm{kills}}$ — Number of Kills

$N_{\mathrm{kills}}$ is the total number of opponents eliminated by the agent during an episode.

$$
N_{\mathrm{kills}}
=
\text{number of successful eliminations}.
$$

For the standard one-versus-one evaluation, this will typically be either zero or one per episode.

For other StrikeForce scenarios containing multiple opponents, the same metric naturally generalizes to the total number of eliminations.

---

## 7. Health Persistence

### $A_{\mathrm{HP}}$ — Area Under the HP Curve

Let $HP(t)$ denote the agent's health at time $t$.

The accumulated health maintained during the agent's lifetime is defined as:

$$
A_{\mathrm{HP}}
=
\int_{0}^{T_{\mathrm{alive}}} HP(t)\,dt.
$$

For a frame-based environment, the integral can be approximated by:

$$
A_{\mathrm{HP}}
\approx
\sum_{t=1}^{T_{\mathrm{alive}}} HP_t.
$$

This metric captures both the agent's health level and the duration for which that health is maintained.

Consequently, $A_{\mathrm{HP}}$ contains information that cannot be represented by final HP alone.

For example, two agents may finish an episode with the same HP while having experienced very different health trajectories. Their $A_{\mathrm{HP}}$ values can distinguish these cases.

A larger $A_{\mathrm{HP}}$ indicates that the agent maintained more accumulated health throughout its lifetime.

---

## 8. Stamina Persistence

### $A_{\mathrm{stamina}}$ — Area Under the Stamina Curve

Let $S(t)$ denote the agent's stamina at time $t$.

The accumulated stamina throughout the agent's lifetime is defined as:

$$
A_{\mathrm{stamina}}
=
\int_{0}^{T_{\mathrm{alive}}} S(t)\,dt.
$$

For a frame-based implementation:

$$
A_{\mathrm{stamina}}
\approx
\sum_{t=1}^{T_{\mathrm{alive}}} S_t.
$$

This metric measures how much usable stamina the agent maintains over the course of its survival.

It is therefore different from measuring only initial or final stamina.

A larger $A_{\mathrm{stamina}}$ indicates that the agent maintained a higher level of stamina over a larger portion of its lifetime.

---

## 9. Damage Capability

### $A_{\mathrm{damage}}$ — Area Under the Damage-per-Bullet Curve

StrikeForce includes a damage-per-bullet quantity representing the amount of damage the agent can inflict on an opponent with a shot at a particular point in time.

Let

$$
D(t)
$$

denote the agent's damage-per-bullet value at time $t$.

This capability is dynamic during an episode. Being hit may reduce the agent's damage-per-bullet capability, while certain actions or conditions can partially restore it.

The accumulated offensive capability is therefore defined as:

$$
A_{\mathrm{damage}}
=
\int_{0}^{T_{\mathrm{alive}}} D(t)\,dt.
$$

For a frame-based implementation:

$$
A_{\mathrm{damage}}
\approx
\sum_{t=1}^{T_{\mathrm{alive}}} D_t.
$$

Thus, $A_{\mathrm{damage}}$ measures sustained offensive capability over the agent's lifetime rather than the damage-per-bullet value at a single instant.

A larger value indicates that the agent maintained a greater effective damage-per-bullet capability for a larger portion of its lifetime.

---

## 10. Interpretation of the Baseline Metrics

Together, the five baseline metrics characterize several complementary properties of an agent:

| Metric                 | Primary interpretation                         |
| ---------------------- | ---------------------------------------------- |
| $T_{\mathrm{alive}}$   | Survival duration                              |
| $N_{\mathrm{kills}}$   | Successful eliminations                        |
| $A_{\mathrm{HP}}$      | Accumulated health maintained during survival  |
| $A_{\mathrm{stamina}}$ | Accumulated stamina maintained during survival |
| $A_{\mathrm{damage}}$  | Accumulated damage-per-bullet capability       |

The trajectory-based metrics are intentionally defined as areas under curves. They capture the entire lifetime of the agent rather than only an instantaneous or final state.

---

## 11. Recommended Head-to-Head Report

For a basic one-versus-one comparison, the following information should be reported for each model:

| Metric                 | Model A | Model B |
| ---------------------- | ------: | ------: |
| Wins                   |       — |       — |
| $T_{\mathrm{alive}}$   |       — |       — |
| $N_{\mathrm{kills}}$   |       — |       — |
| $A_{\mathrm{HP}}$      |       — |       — |
| $A_{\mathrm{stamina}}$ |       — |       — |
| $A_{\mathrm{damage}}$  |       — |       — |

When multiple episodes are used, the reported values should preferably include an aggregation such as mean and standard deviation, together with the number of evaluation episodes.

For the head-to-head scenario, win rate can additionally be reported:

$$
\mathrm{WinRate}
=
\frac{N_{\mathrm{wins}}}{N_{\mathrm{episodes}}}.
$$

---

## 12. Evaluation Conditions

For a meaningful model comparison, the following should remain consistent across the compared models whenever possible:

* map or enclosed combat area;
* number and type of opponents;
* initial resource conditions;
* episode termination conditions;
* simulation settings;
* action-selection procedure;
* inference-time modifications;
* number of evaluation episodes.

When any of these conditions are intentionally changed, the experiment should state the change explicitly.

---

## 13. Extending the Baseline

The five metrics defined in this document are a baseline rather than an exhaustive benchmark.

Researchers may add other metrics when they are useful for a particular experiment, such as:

* additional combat statistics;
* resource consumption;
* action frequencies;
* movement statistics;
* task-specific behavioral measurements;
* model-specific or research-specific metrics.

The baseline metrics should not prevent more specialized evaluations. Their purpose is to provide a common minimum set that makes results from different models and experiments easier to compare.

---

## 14. Recommended Baseline Procedure

A standard evaluation can therefore follow the following structure:

1. Select the models or checkpoints to compare.
2. Place the agents in the standard one-versus-one enclosed combat scenario.
3. Run the same number of evaluation episodes for every model.
4. Record $T_{\mathrm{alive}}$, $N_{\mathrm{kills}}$, $A_{\mathrm{HP}}$, $A_{\mathrm{stamina}}$, and $A_{\mathrm{damage}}$.
5. Report the resulting statistics for each model.
6. Add any additional metrics required by the particular research question.

This protocol is intended to be sufficiently general that any model developed for StrikeForce can be evaluated with the same basic procedure while retaining the freedom to introduce more specialized measurements.
