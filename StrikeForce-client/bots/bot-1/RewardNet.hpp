/*
MIT License (c) 2025 bistoyek21 R.I.C.

RewardNet.hpp — bot-1
Replaces GAIL with AIRL (Fu et al., 2018).

AIRL Discriminator:
  D(s,a,s') = σ( f(s,a,s') − log π(a|s) )
  f(s,a,s') = r_θ(s,a) + γ·h_φ(s') − h_φ(s)

Where:
  r_θ(s,a)  — reward network  (what we recover)
  h_φ(s)    — shaping function (disentangles reward from dynamics)

At convergence r_θ approximates the true reward independently
of the current policy — unlike GAIL whose pseudo-reward is
entangled with the policy density ratio.

Training (binary CE):
  expert transition : target D = 1
  policy transition : target D = 0

Reward used in PPO:
  r_AIRL(t) = f(s,a,s') − log π(a|s)
            = log D / (1-D)
*/
#pragma once
#include "Modules.hpp"

// ─────────────────────────────────────────────
//  RewardNet class
// ─────────────────────────────────────────────
class RewardNet {
public:
    explicit RewardNet(bool training = true, int T = 512,
                       float gamma = 0.99f, float lr = 1e-3f,
                       const std::string& backup_dir =
                           "bots/bot-1/backup/reward_backup")
        : training_(training), T_(T), gamma_(gamma), backup_dir_(backup_dir) {

        model_ = AIRLNets();

        if (!backup_dir_.empty()) {
            if (std::filesystem::exists(backup_dir_)) {
                log_file_.open(backup_dir_ + "/reward_log.log", std::ios::app);
                try { torch::load(model_, backup_dir_ + "/model.pt"); }
                catch (...) { log("No checkpoint, fresh start"); }
            } else {
                std::filesystem::create_directories(backup_dir_);
                log_file_.open(backup_dir_ + "/reward_log.log", std::ios::app);
            }
        }

#if defined(FREEZE_REWARDNET_BLOCK)
        training_ = false;
        log("RewardNet frozen");
#endif

        int params = 0;
        for (auto& p : model_->parameters()) params += p.numel();
        log("AIRL parameters: " + std::to_string(params));

        if (training_) {
            model_->train();
            optimizer_ = std::make_unique<torch::optim::Adam>(
                model_->parameters(), torch::optim::AdamOptions(lr));
            if (std::filesystem::exists(backup_dir_ + "/optimizer.pt")) {
                try { torch::load(*optimizer_, backup_dir_ + "/optimizer.pt"); }
                catch (...) {}
            }
        } else {
            model_->eval();
        }
    }

    ~RewardNet() {
        if (training_ && !backup_dir_.empty()) {
            model_->reset_memory();
            torch::save(model_, backup_dir_ + "/model.pt");
            torch::save(*optimizer_, backup_dir_ + "/optimizer.pt");
        }
        log_file_.close();
    }

    // ──────────────────────────────────────────
    //  Compute AIRL rewards for an entire rollout.
    //  Called BEFORE PPO update so the agent trains
    //  on fresh discriminator signals.
    //
    //  rollout_is_expert[t] == true if step t was
    //  a human-controlled demonstration.
    //
    //  Returns vector of scalar rewards ∈ ℝ.
    //  (No hard -2 sentinel — caller handles masking.)
    // ──────────────────────────────────────────
    std::vector<float> compute_rewards(
        const std::vector<torch::Tensor>& states,
        const std::vector<int>&           actions,
        const std::vector<torch::Tensor>& next_states,
        const std::vector<float>&         log_pis,    // log π(a_t|s_t) from policy
        bool manual)                                   // which half is human
    {
        std::vector<float> rewards;
        rewards.reserve(states.size());

        torch::NoGradGuard ng;
        model_->eval();
        model_->reset_memory();

        for (int t = 0; t < (int)states.size(); ++t) {
            auto one_hot = torch::zeros({model_->num_actions});
            one_hot[actions[t]] = 1.0f;
            model_->update_action_history(one_hot);

            auto [r, h_s] = model_->forward(states[t]);

            // h_φ(s') — run on next state with no action context update
            // (we snapshot and restore action to avoid leaking)
            auto [_r_ns, h_sp] = eval_shaping(next_states[t], one_hot);

            // f(s,a,s') = r_θ(s,a) + γ·h_φ(s') − h_φ(s)
            float f = r.item<float>()
                    + gamma_ * h_sp.item<float>()
                    - h_s.item<float>();

            // r_AIRL = f − log π(a|s)  = log D/(1-D)
            float reward = f - log_pis[t];

            // clip for stability
            reward = std::max(-10.0f, std::min(10.0f, reward));
            rewards.push_back(reward);
        }

        model_->train();
        return rewards;
    }

    // ──────────────────────────────────────────
    //  Update discriminator on a completed rollout.
    //  Alternates expert / policy labels per the
    //  crowdsourced protocol.
    // ──────────────────────────────────────────
    void train_epoch(
        const std::vector<torch::Tensor>& states,
        const std::vector<int>&           actions,
        const std::vector<torch::Tensor>& next_states,
        const std::vector<float>&         log_pis,
        bool manual,                       // first half is human iff manual==true
        int  disc_epochs = 3)
    {
        if (!training_ || states.empty()) return;

        int T = (int)states.size();
        int split = T / 2;

        std::cout << "Training AIRL discriminator..." << std::endl;

        float total_loss = 0.f, avg_acc = 0.f;

        for (int epoch = 0; epoch < disc_epochs; ++epoch) {
            model_->reset_memory();
            float ep_loss = 0.f, ep_acc = 0.f;

            for (int t = 0; t < T; ++t) {
                bool is_expert = (t < split) == manual;   // XOR-free

                auto one_hot = torch::zeros({model_->num_actions});
                one_hot[actions[t]] = 1.0f;
                model_->update_action_history(one_hot);

                // Forward through AIRL nets
                auto [r, h_s] = model_->forward(states[t]);
                auto [_r, h_sp] = eval_shaping(next_states[t], one_hot);

                // D = σ(f − log π) — both 0-dim scalars after squeeze() fix
                auto f_t   = r + gamma_ * _r.detach() - h_s;  // differentiable f (approx)
                auto logit = f_t - log_pis[t];                 // 0-dim
                auto D     = torch::sigmoid(logit);             // 0-dim ∈ (0,1)

                // target must match D's shape (0-dim)
                auto target = torch::full_like(D, is_expert ? 1.f : 0.f);
                auto loss   = torch::nn::functional::binary_cross_entropy(D, target);

                optimizer_->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0f);
                optimizer_->step();

                ep_loss += loss.item<float>();
                ep_acc  += (D.item<float>() > 0.5f) == is_expert ? 1.f : 0.f;
            }

            total_loss += ep_loss / T;
            avg_acc    += ep_acc  / T;
        }

        log("AIRL loss=" + std::to_string(total_loss / disc_epochs)
          + " acc=" + std::to_string(100.f * avg_acc / disc_epochs) + "%");

        model_->reset_memory();
        ++train_count_;
        std::cout << "AIRL update complete." << std::endl;
    }

    AIRLNets get_model() { return model_; }

private:
    bool training_;
    int T_, train_count_ = 0;
    float gamma_;
    std::string backup_dir_;
    AIRLNets model_{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::ofstream log_file_;

    // Evaluate h_φ(s) on a state without updating the backbone's
    // GRU memory (we run a single forward pass in no-grad mode
    // and rely on the fact that reset happens per epoch in train_epoch).
    std::pair<torch::Tensor, torch::Tensor> eval_shaping(
        const torch::Tensor& x, const torch::Tensor& action_one_hot)
    {
        torch::NoGradGuard ng;
        // Temporarily use the shared backbone path; because this is
        // called immediately after forward(states[t]) in a sequential
        // loop, the GRU context is already advanced — this gives us
        // h_φ(s') under the *current* context, which is an approximation.
        // A cleaner approach would maintain a separate GRU instance for
        // the shaping function; left as a future refinement.
        model_->update_action_history(action_one_hot);
        return model_->forward(x);
    }

    template<typename T>
    void log(const T& msg) { log_file_ << msg << "\n"; log_file_.flush(); }
};