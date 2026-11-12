/*
MIT License (c) 2025 bistoyek21 R.I.C.

Config.hpp — bot-ppo
Single source of truth for every tunable number of the PPO agent and of the
reward-shaping module.  Pure C++17 (no libtorch), so it can be unit-tested
on its own (see tests/).

Every default below is justified in docs/REWARD_SHAPING_AND_IMPLEMENTATION.md.
Any value can be overridden at run time (no recompilation) by writing
`key = value` lines into  bots/bot-ppo/config.txt  — this is what makes
parameter sweeps cheap.
*/
#pragma once
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace sf_ppo {

// ─────────────────────────────────────────────────────────────
//  Reward shaping  (Section 1 of the documentation)
// ─────────────────────────────────────────────────────────────
struct ShapingConfig {
    // ── Normalisation references: map raw game quantities to ~[0,1] ──
    // AUTO-CALIBRATION (default ON): the account's stats are NOT the 1000/100/1000 of character/human.txt
    // (the shipped account '1' has HP 15000, damage 2500, stamina 15000 — see docs §5).  With auto_ref the four
    // reference values below are taken from the agent's own stats in the first frame of every game:
    //     hp_ref = HP0,  stamina_ref = S0,  damage_ref = 2*D0,  enemy_hp_ref = HP0
    // so the normalised signals are ~1 / 1 / 0.5 for ANY account and the reward weights keep their meaning.
    bool  auto_ref      = true;
    float hp_ref        = 1000.f;  // used only when auto_ref = false
    float stamina_ref   = 1000.f;  // default Human stamina (1000)
    float damage_ref    = 200.f;   // damage-per-bullet that counts as "1.0" (default 100, strong weapons ~175-225)
    float enemy_hp_ref  = 1000.f;  // HP of one opponent -> 1000 damage dealt == one full kill's worth of damage
    float norm_cap      = 1.5f;    // normalised HP/stamina/damage are clipped to [0, norm_cap] (items can exceed the default)

    // ── (a) Sparse event rewards ──
    float kill_bonus    = 1.0f;    // per opponent eliminated      (N_kills)
    float death_penalty = 1.0f;    // paid once when the agent dies

    // ── (b) Dense metric-aligned "area" reward (Lab-Protocol §7-§9) ──
    //  r_area(t) = area_scale * ( w_hp*HP_t + w_stamina*S_t + w_damage*D_t )
    //  Summed over an episode this is exactly a weighted sum of A_HP, A_stamina,
    //  A_damage (and, because it is only paid while alive, it also rewards T_alive).
    float area_scale    = 0.005f;  // ceiling of the whole survival income = area_scale/(1-gamma) = 0.5 < kill_bonus
    float w_hp          = 0.5f;
    float w_stamina     = 0.2f;
    float w_damage      = 0.3f;    // w_hp + w_stamina + w_damage == 1

    // ── (c) Potential-based shaping on cumulative damage dealt (Ng et al. 1999) ──
    //  Phi(s) = dealt_potential * min(damage_dealt / enemy_hp_ref, potential_cap)
    //  F_t    = gamma*Phi(s_{t+1}) - Phi(s_t),   Phi(terminal) = 0
    //  -> dense offensive credit assignment WITHOUT changing the optimal policy.
    float dealt_potential = 0.5f;
    float potential_cap   = 3.0f;  // bound Phi (Ng's theorem needs a bounded potential)

    // ── (d) Potential-based TARGETING terms (added after the first diagnostics: kills were rare because the agent
    //        stayed passive — A_HP 99 % of max, identical returns in 31 of 41 games).  Both are potentials, so they
    //        add dense guidance towards enemies without changing the optimal policy (Ng et al. 1999).
    //  Phi_approach = approach_potential * (R - d) / R   for the nearest hostile at Manhattan distance 1 <= d <= R
    //  Phi_aim      = aim_potential                      if a hostile is in the line of fire (facing direction, <= 10 cells, no wall)
    float approach_potential = 0.10f;  // total value of walking from the edge of the view to the enemy
    float aim_potential      = 0.15f;  // value of "an enemy is in front of me" (~ two hits' worth of damage potential)
    float view_radius        = 15.f;   // R: half of the 31x31 observation window
    bool  terminal_potential_zero = true;

    // ── Safety ──
    float reward_clip   = 2.0f;    // |r_t| <= reward_clip
};

// ─────────────────────────────────────────────────────────────
//  PPO  (Section 2 of the documentation)
// ─────────────────────────────────────────────────────────────
struct PPOConfig {
    int   rollout_len     = 512;     // T: transitions per PPO update (full-sequence BPTT-free replay, see docs)
    int   min_train_steps = 64;      // train on a partial rollout at episode end only if at least this long
    int   num_epochs      = 4;       // passes over each rollout
    float gamma           = 0.99f;   // discount (effective horizon ~100 frames)
    float gae_lambda      = 0.95f;   // GAE lambda
    float lr              = 3e-4f;   // Adam learning rate
    float ppo_clip        = 0.2f;    // epsilon of the clipped surrogate
    float entropy_coef    = 0.02f;   // exploration bonus (9 discrete actions); was 0.01: entropy fell 2.19 -> 0.55 in 35 updates
    // ── teacher guidance (DAgger-style): a scripted policy labels every visited state (Custom.hpp "expert") ──
    //  loss  = PPO + lambda_bc * CE(policy, teacher action)           lambda_bc = bc_coef * max(bc_floor, 1 - update/bc_decay_updates)
    //  and during training the agent EXECUTES the teacher's action with probability expert_mix * max(0, 1 - update/bc_decay_updates)
    //  (those steps are excluded from the PPO policy loss because they are not on-policy).
    float policy_coef      = 1.0f;   // weight of the PPO policy loss (0 = pure imitation / DAgger stage, see presets/stage1_imitation.txt)
    float bc_coef          = 1.0f;   // CE is ~ln 9 = 2.2 at the start, PPO losses ~0.01-0.1 -> imitation dominates early, on purpose
    float bc_floor         = 0.05f;  // never fully forget the teacher
    int   bc_decay_updates = 150;    // updates over which both the weight and the mixing probability fall to their floor / 0
    float expert_mix       = 0.5f;   // initial probability of executing the teacher's action (0 = pure DAgger labels only)
    float adv_std_floor   = 0.10f;   // advantages are divided by max(std, floor): rollouts without signal are NOT amplified to unit variance
    float value_coef      = 0.5f;    // critic loss weight
    float max_grad_norm   = 0.5f;    // global gradient clipping
    float target_kl       = 0.03f;   // early-stop an update when approx-KL > 1.5*target_kl
    bool  normalize_adv   = true;    // per-rollout advantage standardisation
    bool  probe_teacher      = false;// DIAGNOSTIC (frozen runs only): the teacher DRIVES, the policy is only compared with it -> identical states for every checkpoint
    bool  deterministic_eval = false;// argmax actions when training == false (Lab-Protocol §12: keep identical across models)
    unsigned seed         = 0;       // 0 -> random_device

    // ── critic / value head ──
    bool  value_norm       = true;   // PopArt-style value normalisation (output-preserving), see docs §3.2
    float value_norm_beta  = 0.1f;   // EMA rate of the return statistics per PPO update (first update uses beta = 1)
    float value_sigma_min  = 0.01f;  // floor of the return std (rewards here are O(0.01..1))

    // ── pipeline ──
    int   save_every_episodes = 20;  // headless batch runs: write model/optimizer to disk every k episodes (and at exit)
    int   save_every_updates = 10;   // also keep checkpoints/model_u<k>.pt every k PPO updates (0 = never)
    std::string stats_csv  = "bots/bot-ppo/backup/training_stats.csv"; // one row per PPO update
    std::string metrics_csv = "bots/bot-ppo/backup/metrics.csv"; // one row per episode: the five lab-protocol metrics
};

// ─────────────────────────────────────────────────────────────
//  key = value override loader
// ─────────────────────────────────────────────────────────────
inline std::map<std::string, std::string> read_kv(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line.erase(h);
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        };
        std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        if (!k.empty() && !v.empty()) kv[k] = v;
    }
    return kv;
}

// Applies overrides; returns the list of keys that were not recognised.
inline std::string apply_overrides(const std::map<std::string, std::string>& kv,
                                   ShapingConfig& s, PPOConfig& p) {
    std::string unknown;
    for (const auto& [k, v] : kv) {
        bool ok = true;
        auto f = [&](float& x) { x = std::stof(v); };
        auto i = [&](int& x)   { x = std::stoi(v); };
        auto b = [&](bool& x)  { x = (v == "1" || v == "true"); };
        if      (k == "shaping.hp_ref") f(s.hp_ref);
        else if (k == "shaping.stamina_ref") f(s.stamina_ref);
        else if (k == "shaping.damage_ref") f(s.damage_ref);
        else if (k == "shaping.enemy_hp_ref") f(s.enemy_hp_ref);
        else if (k == "shaping.norm_cap") f(s.norm_cap);
        else if (k == "shaping.kill_bonus") f(s.kill_bonus);
        else if (k == "shaping.death_penalty") f(s.death_penalty);
        else if (k == "shaping.area_scale") f(s.area_scale);
        else if (k == "shaping.w_hp") f(s.w_hp);
        else if (k == "shaping.w_stamina") f(s.w_stamina);
        else if (k == "shaping.w_damage") f(s.w_damage);
        else if (k == "shaping.dealt_potential") f(s.dealt_potential);
        else if (k == "shaping.potential_cap") f(s.potential_cap);
        else if (k == "shaping.approach_potential") f(s.approach_potential);
        else if (k == "shaping.aim_potential") f(s.aim_potential);
        else if (k == "shaping.view_radius") f(s.view_radius);
        else if (k == "shaping.terminal_potential_zero") b(s.terminal_potential_zero);
        else if (k == "shaping.reward_clip") f(s.reward_clip);
        else if (k == "ppo.rollout_len") i(p.rollout_len);
        else if (k == "ppo.min_train_steps") i(p.min_train_steps);
        else if (k == "ppo.num_epochs") i(p.num_epochs);
        else if (k == "ppo.gamma") f(p.gamma);
        else if (k == "ppo.gae_lambda") f(p.gae_lambda);
        else if (k == "ppo.lr") f(p.lr);
        else if (k == "ppo.ppo_clip") f(p.ppo_clip);
        else if (k == "ppo.entropy_coef") f(p.entropy_coef);
        else if (k == "ppo.adv_std_floor") f(p.adv_std_floor);
        else if (k == "ppo.policy_coef") f(p.policy_coef);
        else if (k == "ppo.bc_coef") f(p.bc_coef);
        else if (k == "ppo.bc_floor") f(p.bc_floor);
        else if (k == "ppo.bc_decay_updates") i(p.bc_decay_updates);
        else if (k == "ppo.expert_mix") f(p.expert_mix);
        else if (k == "ppo.value_coef") f(p.value_coef);
        else if (k == "ppo.max_grad_norm") f(p.max_grad_norm);
        else if (k == "ppo.target_kl") f(p.target_kl);
        else if (k == "ppo.normalize_adv") b(p.normalize_adv);
        else if (k == "ppo.deterministic_eval") b(p.deterministic_eval);
        else if (k == "ppo.probe_teacher") b(p.probe_teacher);
        else if (k == "ppo.seed") p.seed = (unsigned)std::stoul(v);
        else if (k == "ppo.metrics_csv") p.metrics_csv = v;
        else if (k == "ppo.value_norm") b(p.value_norm);
        else if (k == "ppo.value_norm_beta") f(p.value_norm_beta);
        else if (k == "ppo.value_sigma_min") f(p.value_sigma_min);
        else if (k == "ppo.save_every_updates") i(p.save_every_updates);
        else if (k == "ppo.save_every_episodes") i(p.save_every_episodes);
        else if (k == "ppo.stats_csv") p.stats_csv = v;
        else ok = false;
        if (!ok) unknown += k + " ";
    }
    return unknown;
}

} // namespace sf_ppo
