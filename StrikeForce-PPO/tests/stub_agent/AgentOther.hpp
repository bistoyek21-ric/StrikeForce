// Test double of an UNRELATED agent (like bot-bc-fap): no shaped-reward hook, writes no metrics.csv.
// Used to prove that the headless runner reports results for ANY agent.
#pragma once
#include <random>
#include <vector>
#include <string>
#include "../../StrikeForce-client/basic.hpp"
const std::string bot_code = "other", backup_path = "bots/other/backup";
class Agent {
public:
    explicit Agent(bool = true) : rng_(7) {}
    int  predict(const std::vector<float>&) { return (int)(rng_() % 9); }
    void update(int, bool) {}
    bool is_manual() { return false; }
    bool flip() { return false; }
    bool in_training() { return false; }
private: std::mt19937 rng_;
};
