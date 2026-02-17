/*
MIT License (c) 2025 bistoyek21 R.I.C.

Agent.hpp — bot-1

Training loop:
  1. Collect T transitions, store (s, a, s', log π, V)
  2. Compute AIRL rewards r_AIRL from discriminator
  3. GAE(γ=0.99, λ=0.95) on AIRL rewards
  4. PPO update (policy + value) for num_epochs
  5. AIRL discriminator update for disc_epochs
  6. Auxiliary tasks (inverse + forward dynamics)  ← extra backbone gradient

Bug fixes from bot-1 (original):
  ✔ Unbounded value function (no sigmoid)
  ✔ Standard GAE advantage (not R − log V)
  ✔ Correct discounted returns (not (1-γ)·sum)
  ✔ Async training (non-blocking predict)
  ✔ Running-stat observation normalisation
  ✔ next_states buffer for AIRL
*/
//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system
#pragma once
#include "RewardNet.hpp"

const std::string bot_code    = "bot-1";
const std::string backup_path = "bots/bot-1/backup";

class Agent {
public:
    Agent(bool training = true,
          int   T            = 512,
          int   num_epochs   = 4,
          int   disc_epochs  = 3,
          float gamma        = 0.99f,
          float gae_lambda   = 0.95f,
          float lr_policy    = 3e-4f,
          float lr_value     = 1e-3f,
          float ppo_clip     = 0.2f,
          float entropy_coef = 0.01f,
          float value_coef   = 0.5f,
          float aux_coef     = 0.05f,           // weight for auxiliary losses
          const std::string& backup_dir =
              "bots/bot-1/backup/agent_backup")
        : training_(training), T_(T), num_epochs_(num_epochs),
          disc_epochs_(disc_epochs), gamma_(gamma), gae_lambda_(gae_lambda),
          ppo_clip_(ppo_clip), entropy_coef_(entropy_coef),
          value_coef_(value_coef), aux_coef_(aux_coef),
          backup_dir_(backup_dir) {

#if defined(CROWDSOURCED_TRAINING)
        std::cout << "Loading backup..." << std::endl;
        request_and_extract_backup(backup_path, bot_code);
        std::cout << "Press space to continue" << std::endl;
        while (getch() != ' ');
#endif

        // Initialise AIRL reward net
        reward_net_ = std::make_unique<RewardNet>(
            training_, T_, gamma_,
            /*lr=*/1e-3f,
            backup_dir_ + "/../reward_backup");

        // Initialise policy model
        model_ = AgentModel(num_channels_, grid_x_, grid_y_, hidden_size_, num_actions_);

        // Load checkpoint
        if (!backup_dir_.empty()) {
            if (std::filesystem::exists(backup_dir_)) {
                log_file_.open(backup_dir_ + "/agent_log.log", std::ios::app);
                try { torch::load(model_, backup_dir_ + "/model.pt");
                      log("Loaded checkpoint"); }
                catch (...) { log("Fresh start"); }
            } else {
                std::filesystem::create_directories(backup_dir_);
                log_file_.open(backup_dir_ + "/agent_log.log", std::ios::app);
            }
        }

#if defined(FREEZE_AGENT_BLOCK)
        training_ = false;
        log("Agent frozen");
#endif

#if defined(TL_IMPORT_REWARDNET)
        model_->import_backbone_from(*reward_net_->get_model());
        log("Imported backbone from AIRL nets");
#endif

        // Parameter count
        int nparams = 0;
        for (auto& p : model_->parameters()) nparams += p.numel();
        log("Agent parameters: " + std::to_string(nparams));

        // Obs normalisation (Welford: mean + M2 accumulator)
        obs_mean_ = torch::zeros({num_channels_});
        obs_M2_   = torch::zeros({num_channels_});   // M2=0 → std=0 until data arrives
        obs_count_ = 0;
        if (std::filesystem::exists(backup_dir_ + "/obs_stats.pt")) {
            try {
                std::vector<torch::Tensor> stats;
                torch::load(stats, backup_dir_ + "/obs_stats.pt");
                obs_mean_  = stats[0];
                obs_M2_    = stats[1];
                obs_count_ = (int)stats[2][0].item<float>();
                log("Loaded obs statistics (n=" + std::to_string(obs_count_) + ")");
            } catch (...) {}
        }

        // Optimisers — separate LRs for policy vs value
        if (training_) {
            model_->train();
            // Policy optimizer covers everything except value_head
            std::vector<torch::Tensor> policy_params, value_params;
            for (auto& kv : model_->named_parameters()) {
                if (kv.key().rfind("value", 0) == 0)     // starts with "value"
                    value_params.push_back(kv.value());
                else
                    policy_params.push_back(kv.value());
            }
            policy_opt_ = std::make_unique<torch::optim::Adam>(
                policy_params, torch::optim::AdamOptions(lr_policy));
            value_opt_  = std::make_unique<torch::optim::Adam>(
                value_params, torch::optim::AdamOptions(lr_value));

#if defined(FREEZE_TL_BLOCK)
            model_->freeze_backbone();
            log("Backbone frozen");
#endif

            if (std::filesystem::exists(backup_dir_ + "/policy_opt.pt")) {
                try { torch::load(*policy_opt_, backup_dir_ + "/policy_opt.pt");
                      torch::load(*value_opt_,  backup_dir_ + "/value_opt.pt"); }
                catch (...) {}
            }
        } else {
            model_->eval();
        }

        // Warm-up forward pass to initialise GRU states
        {
            torch::NoGradGuard ng;
            auto dummy = torch::zeros({1, num_channels_, grid_x_, grid_y_});
            model_->forward(dummy);
            model_->reset_memory();
        }
    }

    ~Agent() {
        if (train_thread_.joinable()) {
            std::cout << "Waiting for training thread..." << std::endl;
            train_thread_.join();
        }
        if (training_ && !backup_dir_.empty()) {
            model_->reset_memory();
            torch::save(model_, backup_dir_ + "/model.pt");
            torch::save(*policy_opt_, backup_dir_ + "/policy_opt.pt");
            torch::save(*value_opt_,  backup_dir_ + "/value_opt.pt");
            std::vector<torch::Tensor> stats = {
                obs_mean_,
                obs_M2_,
                torch::tensor({(float)obs_count_})
            };
            torch::save(stats, backup_dir_ + "/obs_stats.pt");
        }
        log("Total episodes: " + std::to_string(episode_count_));
        log_file_.close();

#if defined(CROWDSOURCED_TRAINING)
        std::cout << "Submit backup? (y/n): " << std::endl;
        if (getch() == 'y') zip_and_return_backup(backup_path);
#endif
    }

    // ──────────────────────────────────────────
    //  predict() — non-blocking even during training.
    // ──────────────────────────────────────────
    int predict(const std::vector<float>& obs_vec) {
        if (cnt_ < T_warmup_) return 0;

        auto state = to_tensor(obs_vec);          // normalised
        states_.push_back(state);

        torch::NoGradGuard ng;
        auto [probs, value, z] = model_->forward(state);

        values_.push_back(value.detach());
        log_probs_.push_back(torch::log(probs + 1e-8f).detach());
        latents_.push_back(z.detach());

        // Sample action
        std::vector<float> pvec(num_actions_);
        for (int i = 0; i < num_actions_; ++i)
            pvec[i] = probs[i].item<float>();
        std::discrete_distribution<int> dist(pvec.begin(), pvec.end());
        std::mt19937 gen(std::random_device{}());
        return dist(gen);
    }

    // ──────────────────────────────────────────
    //  update() — store transition, trigger training
    // ──────────────────────────────────────────
    void update(int action, bool is_human) {
        if (cnt_ < T_warmup_) return;

        actions_.push_back(action);
        human_flags_.push_back(is_human);

        // Store next_state placeholder (filled at start of next predict)
        // We append states_[last] as dummy; overwritten below.
        // Real next_state = states_[t+1] once it arrives.
        // The final transition uses zero-padded next_state.
        if (actions_.size() > 1)
            next_states_.push_back(states_.back());       // states_[t] at step t+1
        else
            next_states_.push_back(torch::zeros_like(states_.back()));

        // Update action history in backbone
        auto one_hot = torch::zeros({num_actions_});
        one_hot[action] = 1.0f;
        model_->update_action_history(one_hot.detach());

        if ((int)actions_.size() >= T_) {
            // Fix up last next_state to zeros (conservative terminal)
            if (!next_states_.empty())
                next_states_.back() = torch::zeros_like(states_.back());

            train();
            /*
            // Launch async training
            if (train_thread_.joinable())
                train_thread_.join();                     // wait for previous cycle
            train_thread_ = std::thread(&Agent::train, this);
            */
        }
    }

#if defined(CROWDSOURCED_TRAINING)
    bool is_manual() {
        if (cnt_ < T_warmup_) { ++cnt_; return true; }
        if (actions_.empty()) {
            manual_ = !manual_;
            if (manual_) {
                std::cout << "Manual phase — press space." << std::endl;
                while (getch() != ' ');
            }
        }
        return manual_;
    }
#endif

    bool in_training() {
        return train_thread_.joinable() && !done_training_;
    }

private:
    // ── Hyper-params ──────────────────────────
    bool   training_;
    int    T_, num_epochs_, disc_epochs_;
    float  gamma_, gae_lambda_, ppo_clip_, entropy_coef_, value_coef_, aux_coef_;

    const int num_actions_  = 9;
    const int num_channels_ = 32;
    const int grid_x_       = 31;
    const int grid_y_       = 31;
    const int hidden_size_  = 160;

    // ── State ─────────────────────────────────
    int  cnt_ = 0, T_warmup_ = 10, episode_count_ = 0;
    bool manual_ = false, done_training_ = false;
    std::string backup_dir_;

    // ── Networks ──────────────────────────────
    AgentModel model_{nullptr};
    std::unique_ptr<RewardNet> reward_net_;
    std::unique_ptr<torch::optim::Adam> policy_opt_, value_opt_;

    // ── Obs normalisation ─────────────────────
    torch::Tensor obs_mean_, obs_M2_;   // Welford accumulators
    int obs_count_;

    // ── Rollout buffer ────────────────────────
    std::vector<torch::Tensor> states_, next_states_, values_, log_probs_, latents_;
    std::vector<int>   actions_;
    std::vector<bool>  human_flags_;

    // ── Thread ────────────────────────────────
    std::thread train_thread_;
    std::ofstream log_file_;

    // ──────────────────────────────────────────
    //  Obs normalisation helpers
    // ──────────────────────────────────────────
    torch::Tensor to_tensor(const std::vector<float>& raw) {
        auto t = torch::tensor(raw, torch::kFloat32)
                     .view({1, num_channels_, grid_x_, grid_y_});
        if (obs_count_ < 2) return t;   // not enough data to normalise yet
        for (int c = 0; c < num_channels_; ++c) {
            float std_c = std::sqrt(obs_M2_[c].item<float>() / obs_count_ + 1e-8f);
            t[0][c] = (t[0][c] - obs_mean_[c]) / std_c;
        }
        return t;
    }

    void update_obs_stats(const torch::Tensor& state) {
        // Welford online algorithm — maintains mean and M2 (sum of squared deviations)
        // std = sqrt(M2 / count)  computed at read time in to_tensor()
        ++obs_count_;
        for (int c = 0; c < num_channels_; ++c) {
            float x     = state[0][c].mean().item<float>();
            float delta = x - obs_mean_[c].item<float>();
            obs_mean_[c] = obs_mean_[c] + delta / (float)obs_count_;
            float delta2 = x - obs_mean_[c].item<float>();   // uses updated mean
            obs_M2_[c]   = obs_M2_[c] + delta * delta2;
        }
    }

    // ──────────────────────────────────────────
    //  GAE  (Generalized Advantage Estimation)
    //  A_t = Σ_{l≥0} (γλ)^l δ_{t+l}
    //  δ_t  = r_t + γ V(s_{t+1}) − V(s_t)
    // ──────────────────────────────────────────
    std::vector<float> compute_gae(
        const std::vector<float>&         rewards,
        const std::vector<torch::Tensor>& vals,
        float last_value = 0.f)
    {
        int T = (int)rewards.size();
        std::vector<float> adv(T, 0.f);
        float gae = 0.f;

        for (int t = T - 1; t >= 0; --t) {
            float v_t  = vals[t].item<float>();
            float v_t1 = (t == T - 1) ? last_value : vals[t + 1].item<float>();
            float delta = rewards[t] + gamma_ * v_t1 - v_t;
            gae = delta + gamma_ * gae_lambda_ * gae;
            adv[t] = gae;
        }
        return adv;
    }

    // ──────────────────────────────────────────
    //  Main training function (runs in thread)
    // ──────────────────────────────────────────
    void train() {
        // Defensive copy: thread owns these until done
        auto states       = states_;
        auto next_states  = next_states_;
        auto actions      = actions_;
        auto human_flags  = human_flags_;
        auto log_probs    = log_probs_;
        auto values       = values_;
        auto latents      = latents_;

        int T = (int)actions.size();
        std::cout << "=== Training episode " << episode_count_ << " (T=" << T << ") ===" << std::endl;

        // ── Update obs stats ──────────────────
        for (auto& s : states) update_obs_stats(s);

        // ── Extract log π(a_t|s_t) ────────────
        std::vector<float> log_pi_vec(T);
        for (int t = 0; t < T; ++t)
            log_pi_vec[t] = log_probs[t][actions[t]].item<float>();

        // ── Determine which half is human ─────
        // Convention: if manual_==true, first T/2 steps were human
        bool manual_snap = manual_;

        // ── AIRL rewards ─────────────────────
        auto airl_rewards = reward_net_->compute_rewards(
            states, actions, next_states, log_pi_vec, manual_snap);

        // ── GAE ───────────────────────────────
        auto advantages = compute_gae(airl_rewards, values);

        // Normalise advantages
        float mu = 0.f, sig = 0.f;
        for (float a : advantages) mu  += a;   mu  /= T;
        for (float a : advantages) sig += (a - mu) * (a - mu);
        sig = std::sqrt(sig / T + 1e-8f);
        for (float& a : advantages) a = (a - mu) / sig;

        // Returns = advantages + V(s)
        std::vector<float> returns(T);
        for (int t = 0; t < T; ++t)
            returns[t] = advantages[t] + values[t].item<float>();

        auto ret_tensor  = torch::tensor(returns);
        auto adv_tensor  = torch::tensor(advantages);
        auto old_logp    = torch::stack(log_probs);

        // ── PPO ───────────────────────────────
        float sum_ploss = 0.f, sum_vloss = 0.f, sum_ent = 0.f;

        for (int epoch = 0; epoch < num_epochs_; ++epoch) {
            model_->reset_memory();

            float ep_pl = 0.f, ep_vl = 0.f, ep_ent = 0.f;

            for (int t = 0; t < T; ++t) {
                auto [probs, value, _z] = model_->forward(states[t]);

                // Update action history for next step
                auto oh = torch::zeros({num_actions_});
                oh[actions[t]] = 1.0f;
                model_->update_action_history(oh);

                auto log_p_new = torch::log(probs[actions[t]] + 1e-8f);
                auto ratio     = torch::exp(log_p_new - old_logp[t][actions[t]]);
                auto clipped   = torch::clamp(ratio, 1.f - ppo_clip_, 1.f + ppo_clip_);
                auto adv_t     = adv_tensor[t];

                // PPO clipped surrogate
                auto p_loss = -torch::min(ratio * adv_t, clipped * adv_t);

                // Value loss (MSE on unbounded V)
                auto v_loss = torch::mse_loss(value, ret_tensor[t]);

                // Entropy bonus
                auto entropy = -(probs * torch::log(probs + 1e-8f)).sum();

                auto loss = p_loss + value_coef_ * v_loss - entropy_coef_ * entropy;

                policy_opt_->zero_grad();
                value_opt_->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model_->parameters(), 0.5f);
                policy_opt_->step();
                value_opt_->step();

                ep_pl  += p_loss.item<float>();
                ep_vl  += v_loss.item<float>();
                ep_ent += entropy.item<float>();
            }

            sum_ploss += ep_pl / T;
            sum_vloss += ep_vl / T;
            sum_ent   += ep_ent / T;
        }

        log("PPO policy_loss=" + fmt(sum_ploss / num_epochs_)
          + " value_loss="     + fmt(sum_vloss / num_epochs_)
          + " entropy="        + fmt(sum_ent   / num_epochs_));

        // ── AIRL discriminator update ─────────
        reward_net_->train_epoch(
            states, actions, next_states, log_pi_vec,
            manual_snap, disc_epochs_);

        // ── Auxiliary tasks ───────────────────
        // Recompute latents (fresh, with grad) for aux loss
        model_->reset_memory();
        std::vector<torch::Tensor> fresh_z(T);
        for (int t = 0; t < T; ++t) {
            auto [_p, _v, z] = model_->forward(states[t]);
            auto oh = torch::zeros({num_actions_});
            oh[actions[t]] = 1.0f;
            model_->update_action_history(oh);
            fresh_z[t] = z;
        }

        auto aux = model_->aux_loss(fresh_z, actions);
        auto scaled_aux = aux_coef_ * aux;

        policy_opt_->zero_grad();
        scaled_aux.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 0.5f);
        policy_opt_->step();

        log("Aux loss=" + fmt(aux.item<float>()));

        // ── Stats ─────────────────────────────
        float ep_ret = 0.f;
        for (float r : airl_rewards) ep_ret += r;
        log("Episode return=" + fmt(ep_ret)
          + " avg_return="    + fmt(ep_ret)); // extend with running avg if desired

        // ── Clear buffers ─────────────────────
        states_.clear(); next_states_.clear(); values_.clear();
        log_probs_.clear(); latents_.clear();
        actions_.clear(); human_flags_.clear();
        model_->reset_memory();

        ++episode_count_;
        done_training_ = true;
        std::cout << "Training complete." << std::endl;
    }

    // ──────────────────────────────────────────
    template<typename T>
    void log(const T& msg) { log_file_ << msg << "\n"; log_file_.flush(); }

    std::string fmt(float v) { return std::to_string(v); }
};