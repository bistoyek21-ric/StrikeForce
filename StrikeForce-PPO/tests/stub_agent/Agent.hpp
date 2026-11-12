// Test double of bots/bot-ppo/Agent.hpp WITHOUT libtorch: uniform-random policy, same interface,
// same RewardShaper + metrics.csv.  Lets the real game loop + headless runner be compiled and run anywhere.
#pragma once
#define SHAPED_REWARD
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "../../StrikeForce-client/basic.hpp"
#include "../../StrikeForce-client/bots/bot-ppo/RewardShaping.hpp"

static std::mt19937 g_rng(12345); static double g_sum = 0; static long g_hist[10] = {0}; static int g_maxdealt = 0; static long g_steps = 0, g_vis = 0, g_al = 0; static double g_dsum = 0;
const std::string bot_code = "stub", backup_path = "bots/bot-ppo/backup";

class Agent {
public:
    explicit Agent(bool = true) : shaper_(cfg_, 0.99f), rng_(0) {
        std::filesystem::create_directories("bots/bot-ppo/backup");
    }
    void observe(int hp, int st, int d, int dealt, int kills, int td = -1, int al = 0) { shaper_.observe({hp, st, d, dealt, kills, td, al}); ++g_steps; if (dealt > g_maxdealt) g_maxdealt = dealt; if (td >= 0) { ++g_vis; g_dsum += td; } g_al += al; }
    int  predict(const std::vector<float>& o, int ex = -1) { ++g_hist[ex < 0 || ex > 8 ? 9 : ex]; for (size_t i = 0; i < o.size(); i += 13) g_sum += o[i] * (double)(i % 7 + 1); return (int)(g_rng() % 9); }
    void update(int, bool) {}
    void on_terminal(int hp, int st, int d, int dealt, int kills) {
        shaper_.finish({hp, st, d, dealt, kills}, hp <= 0);
        sf_ppo::RewardShaper::append_csv(ppo_.metrics_csv, shaper_.metrics());
        std::ofstream("bots/bot-ppo/backup/obs_checksum.txt", std::ios::app) << std::setprecision(17) << g_sum << "\n";
        std::ofstream f_("bots/bot-ppo/backup/target_stats.txt"); f_ << "steps " << g_steps << " visible " << g_vis << " aligned " << g_al << " maxdealt " << g_maxdealt << " expert_hist(+xzqeawsd,none):"; for (int i = 0; i < 10; ++i) f_ << " " << g_hist[i]; f_ << " mean_dist " << (g_vis ? g_dsum / g_vis : 0) << "\n";
    }
    bool is_manual() { return false; }
    bool flip() { return false; }
    bool in_training() { return false; }
    static void flush() {}
    static std::string status() { return "stub agent (random actions)"; }
private:
    sf_ppo::ShapingConfig cfg_; sf_ppo::PPOConfig ppo_;
    sf_ppo::RewardShaper shaper_; std::mt19937 rng_;
};
