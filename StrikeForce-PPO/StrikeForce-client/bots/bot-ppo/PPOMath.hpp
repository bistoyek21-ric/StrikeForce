/*
MIT License (c) 2025 bistoyek21 R.I.C.

PPOMath.hpp — bot-ppo
The numerical (non-tensor) parts of the PPO pipeline, in plain C++17 so they are unit-tested
(tests/test_ppo_math.cpp) independently of libtorch:
    gae()                 Generalised Advantage Estimation + critic targets
    explained_variance()  critic diagnostic
    ValueNorm             PopArt statistics (mu, sigma) with output-preserving rescale
*/
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace sf_ppo {

// delta_t = r_t + gamma*V_{t+1} - V_t ;  A_t = delta_t + gamma*lambda*A_{t+1} ;  target_t = A_t + V_t
// V_T (bootstrap) = last_value (0 for a true terminal state).  `values` are on the raw reward scale.
inline void gae(const std::vector<float>& rewards, const std::vector<float>& values, float last_value,
                float gamma, float lambda, std::vector<float>& adv, std::vector<float>& ret) {
    const int T = (int)rewards.size();
    adv.assign(T, 0.f); ret.assign(T, 0.f);
    float run = 0.f;
    for (int t = T - 1; t >= 0; --t) {
        float next_v = (t == T - 1) ? last_value : values[t + 1];
        float delta  = rewards[t] + gamma * next_v - values[t];
        run = delta + gamma * lambda * run;
        adv[t] = run;
        ret[t] = run + values[t];
    }
}

// 1 - Var(ret - V) / Var(ret)    (1 = perfect, 0 = constant predictor, < 0 = worse than constant)
inline float explained_variance(const std::vector<float>& v, const std::vector<float>& ret) {
    const int n = (int)ret.size();
    if (n == 0) return 0.f;
    double mr = 0, me = 0, vr = 0, ve = 0;
    for (float r : ret) mr += r;
    mr /= n;
    for (int i = 0; i < n; ++i) me += ret[i] - v[i];
    me /= n;
    for (int i = 0; i < n; ++i) {
        vr += (ret[i] - mr) * (ret[i] - mr);
        ve += (ret[i] - v[i] - me) * (ret[i] - v[i] - me);
    }
    return vr < 1e-12 ? 0.f : (float)(1.0 - ve / vr);
}

// Teacher schedule: 1 at the first PPO update, falling linearly to 0 after `decay_updates` updates.
// Weight of the imitation loss = bc_coef * max(bc_floor, schedule);  P(execute teacher action) = expert_mix * schedule.
inline float teacher_schedule(long n_updates, int decay_updates) {
    return decay_updates <= 0 ? 0.f : std::max(0.f, 1.f - (float)n_updates / (float)decay_updates);
}

// Running return statistics:  V_raw = mu + sigma * V_head
struct ValueNorm {
    double mu = 0.0, sigma = 1.0, nu = 1.0;   // nu = running E[R^2]

    // Fold a batch of returns into the statistics. `first` => adopt the batch statistics (beta = 1).
    // Returns the previous (mu, sigma) so the caller can rescale the critic's last layer.
    void update(const std::vector<float>& ret, double beta, double sigma_min, bool first,
                double& old_mu, double& old_sigma) {
        double m1 = 0, m2 = 0;
        for (float r : ret) { m1 += r; m2 += (double)r * r; }
        m1 /= ret.size(); m2 /= ret.size();
        const double b = first ? 1.0 : beta;
        old_mu = mu; old_sigma = sigma;
        mu    = (1.0 - b) * mu + b * m1;
        nu    = (1.0 - b) * nu + b * m2;
        sigma = std::sqrt(std::max(nu - mu * mu, sigma_min * sigma_min));
    }
    float denorm(float v)    const { return (float)(mu + sigma * v); }
    float normalize(float r) const { return (float)((r - mu) / sigma); }
};

// PopArt on a linear output layer  y = w.x + b  (reference implementation used by the unit test; the
// libtorch version in Modules.hpp applies the identical formula to the critic's last Linear layer).
inline void popart_rescale(std::vector<double>& w, double& b,
                           double mu, double sigma, double mu2, double sigma2) {
    for (double& x : w) x *= sigma / sigma2;
    b = (sigma * b + mu - mu2) / sigma2;
}

} // namespace sf_ppo
