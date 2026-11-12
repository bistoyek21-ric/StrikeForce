// Unit tests + parameter sanity sweep for bots/bot-ppo/RewardShaping.hpp
// Build: g++ -std=c++17 -O2 test_reward_shaping.cpp -o test_reward_shaping && ./test_reward_shaping
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>
#include "../StrikeForce-client/bots/bot-ppo/RewardShaping.hpp"
using namespace sf_ppo;

static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { ++g_fail; std::printf("  FAIL: %s\n", msg); } else std::printf("  ok:   %s\n", msg); } while (0)

struct Traj { std::vector<FrameSignals> f; bool died; };

// discounted return of a trajectory under (cfg, gamma)
static double disc_return(const Traj& t, const ShapingConfig& cfg, float gamma, EpisodeMetrics* out = nullptr) {
    RewardShaper s2(cfg, gamma); double G = 0, g = 1;
    s2.observe(t.f[0]);
    for (size_t i = 1; i + 1 < t.f.size(); ++i) { G += g * s2.observe(t.f[i]).total; g *= gamma; }
    G += g * s2.finish(t.f.back(), t.died).total;
    if (out) *out = s2.metrics();
    return G;
}

// camper: never fights, never gets hit, survives n frames (best case for the area reward)
static Traj camper(int n) { Traj t; t.died = false; for (int i = 0; i < n; ++i) t.f.push_back({1000, 1000, 200, 0, 0}); return t; }
// fighter that wins: deals 1000 damage over 60 frames, gets a kill, ends alive with 500 hp
static Traj winner() {
    Traj t; t.died = false;
    for (int i = 0; i <= 60; ++i) t.f.push_back({1000 - i * 8, 1000 - i * 5, 100, i * 1000 / 60, 0});
    t.f.back().kills = 1; t.f.back().damage_dealt = 1000;
    return t;
}
// fighter that loses: deals 500 damage then dies
static Traj loser() {
    Traj t; t.died = true;
    for (int i = 0; i <= 60; ++i) t.f.push_back({1000 - i * 17, 1000 - i * 5, 100, i * 500 / 60, 0});
    t.f.back().hp = 0;
    return t;
}

int main() {
    ShapingConfig cfg; const float gamma = 0.99f;

    std::puts("[1] metrics follow Lab-Protocol §5-§9");
    {
        RewardShaper s(cfg, gamma);
        s.observe({1000, 1000, 100, 0, 0});
        s.observe({900, 990, 100, 0, 0});
        s.observe({800, 980, 80, 150, 0});
        s.finish({0, 970, 80, 150, 0}, true);
        const auto& m = s.metrics();
        CHECK(m.T_alive == 3, "T_alive counts frames alive (3)");
        CHECK(m.A_HP == 2700 && m.A_stamina == 2970 && m.A_damage == 280, "A_HP=2700, A_stamina=2970, A_damage=280");
        CHECK(m.died && m.N_kills == 0, "died flag set, N_kills = 0");
    }

    std::puts("[2] potential-based term telescopes (policy invariance, Ng et al. 1999)");
    {
        std::mt19937 g(1); double worst = 0;
        for (int trial = 0; trial < 200; ++trial) {
            ShapingConfig c = cfg; c.kill_bonus = c.death_penalty = c.area_scale = 0;   // isolate F
            RewardShaper s(c, gamma);
            int dealt = 0, n = 20 + g() % 200; double G = 0, d = 1;
            s.observe({1000, 1000, 100, 0, 0});
            for (int i = 1; i < n; ++i) { dealt += (g() % 3 == 0) ? (int)(g() % 120) : 0;
                G += d * s.observe({1000, 1000, 100, dealt, 0}).potential; d *= gamma; }
            G += d * s.finish({0, 1000, 100, dealt, 0}, true).potential;   // terminal Phi = 0
            worst = std::max(worst, std::fabs(G + 0.0));                    // -Phi(s0) = 0
        }
        std::printf("      max |sum gamma^t F_t| over 200 random episodes = %.2e\n", worst);
        CHECK(worst < 1e-4, "discounted sum of F equals -Phi(s0)=0 (up to float error) for terminal episodes");
    }

    std::puts("[3] the survival income can never beat a kill");
    {
        double ceiling = cfg.area_scale / (1 - gamma);
        double camp = disc_return(camper(20000), cfg, gamma);
        double win  = disc_return(winner(), cfg, gamma);
        std::printf("      camper(20000 frames, never hit) = %.3f   ceiling = %.3f   winner = %.3f\n", camp, ceiling, win);
        CHECK(camp <= ceiling + 1e-3, "camper return <= area_scale/(1-gamma)");
        CHECK(ceiling < cfg.kill_bonus, "ceiling < kill_bonus");
        CHECK(win > camp, "winner > perfect camper");
    }

    std::puts("[4] ordering: winner > camper-for-1000-frames > loser");
    {
        double w = disc_return(winner(), cfg, gamma), c = disc_return(camper(1000), cfg, gamma), l = disc_return(loser(), cfg, gamma);
        std::printf("      winner=%.3f  camper=%.3f  loser=%.3f\n", w, c, l);
        CHECK(w > c && c > l, "ordering holds");
    }

    std::puts("[5] bounded: |r_t| <= reward_clip");
    {
        ShapingConfig c = cfg; RewardShaper s(c, gamma);
        s.observe({1000, 1000, 100, 0, 0});
        auto r = s.observe({1000, 1000, 100, 100000, 50});   // absurd jump
        CHECK(std::fabs(r.total) <= c.reward_clip + 1e-6, "clipped");
    }

    std::puts("[5b] reward is invariant to the account's stat scale (auto_ref)  — shipped account: HP 15000, dmg 2500, stamina 15000");
    {
        auto scaled = [](const Traj& t, double k, double kd) { Traj o = t;
            for (auto& f : o.f) { f.hp = (int)(f.hp * k); f.stamina = (int)(f.stamina * k); f.damage_per_bullet = (int)(f.damage_per_bullet * kd); f.damage_dealt = (int)(f.damage_dealt * k); }
            return o; };
        ShapingConfig a; a.auto_ref = true;
        double base = disc_return(winner(), a, gamma), big = disc_return(scaled(winner(), 15.0, 25.0), a, gamma);
        std::printf("      winner return: default stats %.4f   x15 HP/stamina, x25 damage %.4f\n", base, big);
        CHECK(std::fabs(base - big) < 0.02, "auto_ref: same return for 15x larger stats");
        ShapingConfig fx; fx.auto_ref = false;
        double fixed_big = disc_return(scaled(camper(2000), 15.0, 25.0), fx, gamma), fixed_small = disc_return(camper(2000), fx, gamma);
        std::printf("      camper return with FIXED refs: default stats %.4f, big stats %.4f (saturated at norm_cap)\n", fixed_small, fixed_big);
        CHECK(fixed_big > 1.4 * fixed_small, "fixed refs (old behaviour) inflate / saturate the area term on a strong account");
    }

    std::puts("[5c] targeting potentials: telescoping, bounds, direction");
    {
        ShapingConfig c; RewardShaper s(c, gamma);
        c.kill_bonus = c.death_penalty = c.area_scale = 0; c.dealt_potential = 0;
        RewardShaper t(c, gamma);
        std::mt19937 g(5); double worst = 0;
        for (int trial = 0; trial < 200; ++trial) {
            RewardShaper u(c, gamma); double G = 0, d = 1;
            u.observe({1000, 1000, 100, 0, 0, -1, 0});
            int n = 30 + g() % 100;
            for (int i = 1; i < n; ++i) { int td = (g() % 3 == 0) ? -1 : 1 + (int)(g() % 15); G += d * u.observe({1000, 1000, 100, 0, 0, td, (int)(g() % 2)}).potential; d *= gamma; }
            G += d * u.finish({0, 1000, 100, 0, 0}, true).potential;       // terminal Phi = 0
            worst = std::max(worst, std::fabs(G));
        }
        std::printf("      max |sum gamma^t F_t| (approach+aim) over 200 random episodes = %.2e\n", worst);
        CHECK(worst < 1e-4, "approach/aim potentials telescope to 0 (policy invariant)");
        RewardShaper v(c, gamma); v.observe({1000, 1000, 100, 0, 0, 12, 0});
        double closer = v.observe({1000, 1000, 100, 0, 0, 3, 0}).potential;
        RewardShaper w2(c, gamma); w2.observe({1000, 1000, 100, 0, 0, 3, 0});
        double farther = w2.observe({1000, 1000, 100, 0, 0, 12, 0}).potential;
        CHECK(closer > 0 && farther < 0, "walking towards an enemy is rewarded, away is penalised");
        RewardShaper a2(c, gamma); a2.observe({1000, 1000, 100, 0, 0, 5, 0});
        CHECK(a2.observe({1000, 1000, 100, 0, 0, 5, 1}).potential > 0.1, "lining up a target gives about aim_potential");
        CHECK(RewardShaper(c, gamma).potential(FrameSignals{1000, 1000, 100, 100000, 0, 1, 1}) <= c.dealt_potential * c.potential_cap + c.approach_potential + c.aim_potential + 1e-6, "total potential is bounded");
    }

    std::puts("[6] config overrides");
    {
        { std::ofstream f("/tmp/sf_cfg_test.txt"); f << "# c\nshaping.kill_bonus = 2.5 # x\nppo.gamma=0.97\nbogus.key = 1\n"; }
        ShapingConfig s; PPOConfig p;
        auto unk = apply_overrides(read_kv("/tmp/sf_cfg_test.txt"), s, p);
        CHECK(s.kill_bonus == 2.5f && p.gamma == 0.97f && unk == "bogus.key ", "overrides applied, unknown key reported");
    }

    std::puts("\n[sweep] sensitivity of the ordering to the two most influential numbers");
    std::puts("  area_scale  kill_bonus | camper(inf)  winner  loser | winner>camper  camper>loser");
    for (float a : {0.001f, 0.005f, 0.01f, 0.02f, 0.05f})
        for (float k : {0.5f, 1.0f, 2.0f}) {
            ShapingConfig c = cfg; c.area_scale = a; c.kill_bonus = k;
            double ca = disc_return(camper(20000), c, gamma), w = disc_return(winner(), c, gamma), l = disc_return(loser(), c, gamma);
            std::printf("  %9.3f  %10.1f | %11.3f %7.3f %6.3f | %-13s %s%s\n", a, k, ca, w, l,
                        w > ca ? "yes" : "NO (camping)", ca > l ? "yes" : "no",
                        (a == 0.005f && k == 1.0f) ? "   <== default" : "");
        }

    std::printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
