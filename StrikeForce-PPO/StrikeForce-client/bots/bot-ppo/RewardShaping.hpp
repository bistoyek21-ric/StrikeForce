/*
MIT License (c) 2025 bistoyek21 R.I.C.

RewardShaping.hpp — bot-ppo
Pure C++17 (no libtorch).  Turns raw per-frame game signals into the PPO
reward, and — as a by-product — records the five baseline metrics of
StrikeForce-Lab-Protocol.md:

    T_alive, N_kills, A_HP, A_stamina, A_damage

Reward for the transition (s_t, a_t) -> s_{t+1}:

    r_t = r_kill + r_death + r_area + F

    r_kill  = kill_bonus * (kills_{t+1} - kills_t)
    r_death = -death_penalty                     if agent died at t+1
    r_area  = area_scale * (w_hp*hp + w_st*st + w_dmg*dmg)   if alive at t+1
    F       = gamma*Phi(s_{t+1}) - Phi(s_t),   Phi = dealt_potential*min(dealt/enemy_hp_ref, cap) + approach + aim

Full derivation of the default values: docs/REWARD_SHAPING_AND_IMPLEMENTATION.md
*/
#pragma once
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include "Config.hpp"

namespace sf_ppo {

// Raw integers exactly as the game exposes them (Human getters).
struct FrameSignals {
    int hp               = 0;  // Human::get_Hp()
    int stamina          = 0;  // Human::get_stamina()
    int damage_per_bullet= 0;  // Human::get_damage_effect()[0]   == D(t) of Lab-Protocol §9
    int damage_dealt     = 0;  // Human::get_damage()             cumulative damage dealt to enemies
    int kills            = 0;  // Human::get_kills()
    int target_dist      = -1; // Manhattan distance to the nearest hostile in the view window, -1 = none visible
    int aligned          = 0;  // 1 = a hostile is in the line of fire
};

struct StepReward {
    float total = 0.f, kill = 0.f, death = 0.f, area = 0.f, potential = 0.f;
};

// The five baseline metrics (+ bookkeeping), Lab-Protocol §3.
struct EpisodeMetrics {
    long   T_alive   = 0;
    int    N_kills   = 0;
    double A_HP      = 0;
    double A_stamina = 0;
    double A_damage  = 0;
    bool   died      = false;
    // undiscounted sums of each reward component (diagnostics)
    double ret_total = 0, ret_kill = 0, ret_death = 0, ret_area = 0, ret_potential = 0;
};

class RewardShaper {
public:
    explicit RewardShaper(ShapingConfig cfg = {}, float gamma = 0.99f) : cfg_(cfg), gamma_(gamma) {}

    void reset() { started_ = false; m_ = EpisodeMetrics{}; }

    // Reference values from the agent's own stats in the first frame (see Config.hpp, auto_ref).
    void calibrate(const FrameSignals& s0) {
        if (s0.hp > 0)               { cfg_.hp_ref = (float)s0.hp; cfg_.enemy_hp_ref = (float)s0.hp; }
        if (s0.stamina > 0)            cfg_.stamina_ref = (float)s0.stamina;
        if (s0.damage_per_bullet > 0)  cfg_.damage_ref  = 2.f * s0.damage_per_bullet;
    }

    // Call once per frame with the signals of the CURRENT frame.
    // Returns the reward of the transition prev-frame -> this-frame
    // (all zeros on the very first call).
    StepReward observe(const FrameSignals& cur) {
        StepReward r;
        if (!started_ && cfg_.auto_ref) calibrate(cur);
        if (started_) r = transition(prev_, cur, /*dead=*/cur.hp <= 0, /*terminal=*/false);
        if (cur.hp > 0) accumulate(cur);          // metrics: only frames in which the agent is alive
        prev_ = cur; started_ = true;
        record(r);
        return r;
    }

    // Call once when the episode ends (agent removed from the game).
    // `last` = final signals (hp <= 0 when died).  Returns the reward of the final transition.
    StepReward finish(const FrameSignals& last, bool died) {
        StepReward r;
        FrameSignals last_ = last;
        last_.target_dist = prev_.target_dist; last_.aligned = prev_.aligned;   // not available at deletion time
        if (started_) r = transition(prev_, last_, died, /*terminal=*/died);
        m_.died = died;
        m_.N_kills = std::max(m_.N_kills, last.kills);
        prev_ = last_;
        record(r);
        return r;
    }

    const EpisodeMetrics& metrics() const { return m_; }
    const ShapingConfig&  config()  const { return cfg_; }

    // Pure function — exposed for unit tests.
    float potential(int damage_dealt) const {
        float x = std::min(std::max(damage_dealt, 0) / cfg_.enemy_hp_ref, cfg_.potential_cap);
        return cfg_.dealt_potential * x;
    }

    // Full potential of a frame: damage dealt + approach + aim (all bounded, all functions of the state).
    float potential(const FrameSignals& s) const {
        float phi = potential(s.damage_dealt);
        if (s.target_dist >= 1 && s.target_dist <= cfg_.view_radius)
            phi += cfg_.approach_potential * (cfg_.view_radius - s.target_dist) / cfg_.view_radius;
        if (s.aligned) phi += cfg_.aim_potential;
        return phi;
    }

    // One CSV row per episode.
    static void append_csv(const std::string& path, const EpisodeMetrics& m) {
        if (path.empty()) return;
        bool fresh = !std::ifstream(path).good();
        std::ofstream f(path, std::ios::app);
        if (!f) return;
        if (fresh)
            f << "unix_time,T_alive,N_kills,A_HP,A_stamina,A_damage,died,"
                 "return_total,return_kill,return_death,return_area,return_potential\n";
        f << std::time(nullptr) << ',' << m.T_alive << ',' << m.N_kills << ',' << m.A_HP << ','
          << m.A_stamina << ',' << m.A_damage << ',' << (m.died ? 1 : 0) << ','
          << m.ret_total << ',' << m.ret_kill << ',' << m.ret_death << ',' << m.ret_area << ','
          << m.ret_potential << '\n';
    }

private:
    ShapingConfig cfg_;
    float gamma_;
    bool started_ = false;
    FrameSignals prev_;
    EpisodeMetrics m_;

    float norm(int raw, float ref) const {
        return std::min(std::max(raw / ref, 0.f), cfg_.norm_cap);
    }

    void accumulate(const FrameSignals& s) {
        ++m_.T_alive;
        m_.A_HP      += s.hp;
        m_.A_stamina += s.stamina;
        m_.A_damage  += s.damage_per_bullet;
        m_.N_kills    = std::max(m_.N_kills, s.kills);
    }

    StepReward transition(const FrameSignals& p, const FrameSignals& c, bool dead, bool terminal) const {
        StepReward r;
        r.kill  = cfg_.kill_bonus * (float)std::max(0, c.kills - p.kills);
        r.death = dead ? -cfg_.death_penalty : 0.f;
        r.area  = dead ? 0.f
                       : cfg_.area_scale * (cfg_.w_hp      * norm(c.hp, cfg_.hp_ref) +
                                            cfg_.w_stamina * norm(c.stamina, cfg_.stamina_ref) +
                                            cfg_.w_damage  * norm(c.damage_per_bullet, cfg_.damage_ref));
        float phi_next = (terminal && cfg_.terminal_potential_zero) ? 0.f : potential(c);
        r.potential = gamma_ * phi_next - potential(p);
        r.total = std::max(-cfg_.reward_clip,
                  std::min(cfg_.reward_clip, r.kill + r.death + r.area + r.potential));
        return r;
    }

    void record(const StepReward& r) {
        m_.ret_total += r.total; m_.ret_kill += r.kill; m_.ret_death += r.death;
        m_.ret_area += r.area;   m_.ret_potential += r.potential;
    }
};

} // namespace sf_ppo
