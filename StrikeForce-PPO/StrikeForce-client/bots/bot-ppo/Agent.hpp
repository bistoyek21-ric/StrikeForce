/*
MIT License (c) 2025 bistoyek21 R.I.C.

Agent.hpp — bot-ppo
Recurrent PPO agent trained ONLY on the shaped reward of RewardShaping.hpp
(no discriminator, no human demonstrations needed).

Per frame the game calls, in this order:
    bot()  ->  agent->observe(hp, stamina, dmg/bullet, dmg dealt, kills)   // reward of the PREVIOUS action
               agent->predict(obs)                                          // choose action a_t
    ...    ->  agent->update(a_t, is_human)                                 // store the action
and when the agent leaves the game (death / end of game):
    Human::deleteAgent() -> agent->on_terminal(...)                         // final reward, last PPO update, metrics row

Algorithm (see docs/REWARD_SHAPING_AND_IMPLEMENTATION.md §2):
    1. collect T transitions (s, a, log pi_old, V, r_shaped)
    2. GAE(gamma, lambda); returns = A_raw + V   (computed BEFORE advantage normalisation)
    3. num_epochs of clipped-surrogate PPO over the full ordered rollout (GRU memory replayed from zero)
       with value loss, entropy bonus, global grad-clipping and approx-KL early stopping

Critic (value head):  separate residual head on the shared backbone feature z, orthogonal init, and
PopArt-style value normalisation (output-preserving rescale of the last layer) so that the critic always
regresses unit-scale targets whatever the reward scale / reward weights are.

Pipeline artefacts (all under bots/bot-ppo/backup/):
    training_stats.csv   one row per PPO update (losses, KL, clip-fraction, explained variance, PopArt mu/sigma)
    metrics.csv          one row per episode (five Lab-Protocol metrics)
    agent_backup/        model.pt, optimizer.pt, train_state.txt, checkpoints/model_u<k>.pt
*/
#pragma once
#define SHAPED_REWARD          // enables the Human::deleteAgent() -> Agent::on_terminal() hook in Character.hpp

#include <random>
#include <memory>
#include "Modules.hpp"
#include "../../macros.hpp"

// Headless builds decide learning vs. evaluation by their OWN flag, so a stray `#define FREEZE_AGENT_BLOCK`
// left in macros.hpp can no longer silently turn a training run into an evaluation run.
#if defined(HEADLESS_TRAIN) && defined(FREEZE_AGENT_BLOCK)
#undef FREEZE_AGENT_BLOCK
#endif
#if defined(HEADLESS_EVAL) && !defined(FREEZE_AGENT_BLOCK)
#define FREEZE_AGENT_BLOCK
#endif

#include "RewardShaping.hpp"
#include "PPOMath.hpp"

const std::string bot_code    = "bot-ppo";
const std::string backup_path = "bots/bot-ppo/backup";

class Agent {
public:
    explicit Agent(bool training = true,
                   const std::string& backup_dir = "bots/bot-ppo/backup/agent_backup")
        : training_(training), backup_dir_(backup_dir) {

#if defined(CROWDSOURCED_TRAINING)
        std::cout << "Loading backup..." << std::endl;
        request_and_extract_backup(backup_path, bot_code);
        std::cout << "Press space to continue" << std::endl;
        while (getch() != ' ');
#endif
        // ── hyper-parameters: defaults from Config.hpp, optional overrides from config.txt ──
        std::string unknown = sf_ppo::apply_overrides(
            sf_ppo::read_kv("bots/bot-ppo/config.txt"), shaping_cfg_, cfg_);

        std::filesystem::create_directories(backup_dir_ + "/checkpoints");
        log_file_.open(backup_dir_ + "/agent_log.log", std::ios::app);
        if (!unknown.empty()) log("WARNING unknown config keys: " + unknown);
        log("config: T=" + std::to_string(cfg_.rollout_len) + " epochs=" + std::to_string(cfg_.num_epochs) +
            " gamma=" + fmt(cfg_.gamma) + " lambda=" + fmt(cfg_.gae_lambda) + " lr=" + fmt(cfg_.lr) +
            " clip=" + fmt(cfg_.ppo_clip) + " ent=" + fmt(cfg_.entropy_coef) +
            " | kill=" + fmt(shaping_cfg_.kill_bonus) + " death=" + fmt(shaping_cfg_.death_penalty) +
            " area=" + fmt(shaping_cfg_.area_scale) + " dealtPhi=" + fmt(shaping_cfg_.dealt_potential));

        shaper_ = sf_ppo::RewardShaper(shaping_cfg_, cfg_.gamma);
        rng_.seed(cfg_.seed ? cfg_.seed : std::random_device{}());

        // The network, optimizer and critic statistics live in ONE process-wide instance: a batch of
        // hundreds of games (headless runner) never reloads / rewrites the checkpoint between games.
        if (!s_init_) {
            s_model_ = AgentModel(num_channels_, grid_x_, grid_y_, hidden_size_, num_actions_);
            try { torch::load(s_model_, backup_dir_ + "/model.pt"); log("Loaded checkpoint"); }
            catch (...) { log("Fresh start"); }
#if defined(FREEZE_AGENT_BLOCK)
            training_ = false;
            log("Agent frozen (evaluation mode)");
#endif
            int nparams = 0;
            for (auto& p : s_model_->parameters()) nparams += (int)p.numel();
            log("Agent parameters: " + std::to_string(nparams));
            if (training_) {
                s_model_->train();
                s_opt_ = std::make_unique<torch::optim::Adam>(
                    s_model_->parameters(), torch::optim::AdamOptions(cfg_.lr));
                try { torch::load(*s_opt_, backup_dir_ + "/optimizer.pt"); } catch (...) {}
            } else {
                s_model_->eval();
            }
            s_training_ = training_;
            s_cfg_ = cfg_;
            s_backup_dir_ = backup_dir_;
            load_train_state();
            s_init_ = true;
        }
        training_ = s_training_;
        model_ = s_model_;
        optimizer_ = s_opt_.get();
        model_->reset_memory();                         // every game starts with an empty GRU memory
    }

    // One line for the headless result table: is the agent really learning?
    static std::string status() {
        if (!s_init_) return "agent never constructed (no game was started)";
        std::string out = std::string(s_training_ ? "TRAINING" : "FROZEN (evaluation - no learning)") +
               ", PPO updates so far: " + std::to_string(s_updates_) + ", training steps so far: " + std::to_string(s_steps_);
        if (!s_training_) {
            std::error_code ec; const std::string mp = s_backup_dir_ + "/model.pt";
            out += "\n        model.pt: " + std::to_string(std::filesystem::exists(mp, ec) ? (long long)std::filesystem::file_size(mp, ec) : -1LL) +
                   " bytes | deterministic_eval=" + std::to_string((int)s_cfg_.deterministic_eval) +
                   " | probe_teacher=" + std::to_string((int)s_cfg_.probe_teacher);
            if (s_pr_n_ > 0) {
                out += "\n        probe vs teacher labels: top action == teacher in " + std::to_string(100.0 * s_pr_match_ / s_pr_n_).substr(0, 5) +
                       " % of " + std::to_string(s_pr_n_) + " steps";
                if (s_pr_z_ > 0)
                    out += "; punch recall " + std::to_string(100.0 * s_pr_zhit_ / s_pr_z_).substr(0, 5) + " % of " + std::to_string(s_pr_z_) +
                           " punch-labelled steps, mean p(z) there = " + std::to_string(s_pr_pz_ / s_pr_z_).substr(0, 5);
                out += "\n        policy top-action histogram (+xzqeawsd):";
                for (int i = 0; i < 9; ++i) out += " " + std::to_string(s_pr_top_[i]);
                out += "\n        teacher label histogram     (+xzqeawsd):";
                for (int i = 0; i < 9; ++i) out += " " + std::to_string(s_pr_lab_[i]);
            }
        }
        return out;
    }

    // Write checkpoint + critic statistics to disk (idempotent; no-op in evaluation mode).
    static void flush() {
        if (!s_init_ || !s_training_) return;
        torch::save(s_model_, s_backup_dir_ + "/model.pt");
        torch::save(*s_opt_, s_backup_dir_ + "/optimizer.pt");
        std::ofstream f(s_backup_dir_ + "/train_state.txt");
        f.precision(17);
        f << s_vn_.mu << ' ' << s_vn_.sigma << ' ' << s_vn_.nu << ' ' << s_updates_ << ' ' << s_steps_ << '\n';
        s_episodes_since_save_ = 0;
    }

    ~Agent() {
        if (training_) {
            model_->reset_memory();
#if defined(HEADLESS_RUN)
            if (++s_episodes_since_save_ >= cfg_.save_every_episodes) flush();   // headless_main() flushes once more at exit
#else
            flush();
#endif
        }
        log_file_.close();
#if defined(CROWDSOURCED_TRAINING)
        std::cout << "Submit backup? (y/n):" << std::endl;
        if (getch() == 'y') zip_and_return_backup(backup_path);
#endif
    }

    // ── 1. reward signal ──────────────────────────────────────────
    // Called every frame BEFORE predict().  Arguments are the raw Human getters.
    void observe(int hp, int stamina, int damage_per_bullet, int damage_dealt, int kills,
                 int target_dist = -1, int aligned = 0) {
        auto r = shaper_.observe({hp, stamina, damage_per_bullet, damage_dealt, kills, target_dist, aligned});
        if (!training_ || !pending_) return;
        rewards_.push_back(r.total);
        pending_ = false;
        if ((int)rewards_.size() >= cfg_.rollout_len) ready_ = true;
    }

    // ── 2. action selection ───────────────────────────────────────
    // expert_idx: index (0..8) of the action the scripted teacher would take in this state, -1 = no label.
    int predict(const std::vector<float>& obs_vec, int expert_idx = -1) {
        auto state = torch::tensor(obs_vec, torch::kFloat32)
                         .view({1, num_channels_, grid_x_, grid_y_});

        if (training_ && ready_) {                     // rollout complete: update, then keep playing
            float v_boot;
            { torch::NoGradGuard ng; v_boot = denorm(std::get<1>(model_->forward(state)).item<float>()); }
            train(v_boot);
            ready_ = false;
        }

        torch::NoGradGuard ng;
        auto [probs, value, z] = model_->forward(state);
        auto pc = probs.contiguous();
        const float* p = pc.data_ptr<float>();

        int a = 0;
        bool used_expert = false;
        if (!training_ && cfg_.deterministic_eval) {
            for (int i = 1; i < num_actions_; ++i) if (p[i] > p[a]) a = i;
        } else {
            std::discrete_distribution<int> dist(p, p + num_actions_);
            a = dist(rng_);
        }
        if (!training_ && expert_idx >= 0 && expert_idx < num_actions_) {          // probe (no effect on the action unless probe_teacher)
            int top = 0;
            for (int i = 1; i < num_actions_; ++i) if (p[i] > p[top]) top = i;
            ++s_pr_n_; ++s_pr_top_[top]; ++s_pr_lab_[expert_idx];
            if (top == expert_idx) ++s_pr_match_;
            if (expert_idx == 2) { ++s_pr_z_; s_pr_pz_ += p[2]; if (top == 2) ++s_pr_zhit_; }
            if (cfg_.probe_teacher) a = expert_idx;                                 // teacher drives: identical states for every checkpoint
        }
        if (training_ && expert_idx >= 0 && expert_idx < num_actions_) {
            const float beta = cfg_.expert_mix * schedule();
            if (beta > 0.f && std::uniform_real_distribution<float>(0.f, 1.f)(rng_) < beta) {
                a = expert_idx; used_expert = true;      // teacher acts on this step
            }
        }
        if (training_) {
            states_.push_back(state);
            values_.push_back(denorm(value.item<float>()));
            logps_.push_back(std::log(p[a] + 1e-8f));
            expert_.push_back(expert_idx >= 0 && expert_idx < num_actions_ ? expert_idx : -1);
            wmask_.push_back(used_expert ? 0.f : 1.f);   // 0 = not on-policy: excluded from the PPO policy loss
            if (used_expert) ++expert_steps_;
        }
        return a;
    }

    // ── 3. store the executed action ──────────────────────────────
    void update(int action, bool is_human) {
        auto one_hot = torch::zeros({num_actions_});
        one_hot[action] = 1.0f;
        model_->update_action_history(one_hot);        // backbone conditions on the previous action

        if (!training_) return;
        if (is_human) {                                // off-policy frame: drop it from the rollout
            pop_dangling();
            pending_ = false;
            return;
        }
        actions_.push_back(action);
        pending_ = true;
    }

    // ── 4. episode end ────────────────────────────────────────────
    // Called by Human::deleteAgent() (macro SHAPED_REWARD) just before the agent is destroyed.
    void on_terminal(int hp, int stamina, int damage_per_bullet, int damage_dealt, int kills) {
        if (terminated_) return;
        terminated_ = true;
        const bool died = hp <= 0;
        auto r = shaper_.finish({hp, stamina, damage_per_bullet, damage_dealt, kills}, died);

        const auto& m = shaper_.metrics();
        log("EPISODE T_alive=" + std::to_string(m.T_alive) + " kills=" + std::to_string(m.N_kills) +
            " A_HP=" + fmt((float)m.A_HP) + " A_stamina=" + fmt((float)m.A_stamina) +
            " A_damage=" + fmt((float)m.A_damage) + " died=" + std::to_string(died) +
            " shaped_return=" + fmt((float)m.ret_total));
        sf_ppo::RewardShaper::append_csv(cfg_.metrics_csv, m);

        if (!training_) return;
        pop_dangling();
        if (pending_) { rewards_.push_back(r.total); pending_ = false; }

        if ((int)actions_.size() >= cfg_.min_train_steps && rewards_.size() == actions_.size()) {
            float v_boot = died || values_.empty() ? 0.f : values_.back();   // death = true terminal; otherwise truncation
            train(v_boot);
        }
    }

    // ── interface expected by gameplay.hpp ────────────────────────
    bool is_manual() { return flip_; }                 // PPO needs no human phases; flip() lets you take over manually
    bool flip()      { flip_ = !flip_; return true; }
    bool in_training() { return false; }               // training is synchronous

private:
    // config / state
    bool training_;
    std::string backup_dir_;
    sf_ppo::PPOConfig cfg_;
    sf_ppo::ShapingConfig shaping_cfg_;
    sf_ppo::RewardShaper shaper_;
    std::mt19937 rng_;
    bool flip_ = false, pending_ = false, ready_ = false, terminated_ = false;
    // critic normalisation state (PopArt): V_raw = vn_.mu + vn_.sigma * V_head
    // (process-wide, see ctor)  mu, sigma, nu / update + step counters, persisted in train_state.txt
    inline static AgentModel s_model_{nullptr};
    inline static std::unique_ptr<torch::optim::Adam> s_opt_;
    inline static sf_ppo::ValueNorm s_vn_;
    inline static long s_updates_ = 0, s_steps_ = 0;
    inline static bool s_init_ = false, s_training_ = true;
    inline static int  s_episodes_since_save_ = 0;
    inline static std::string s_backup_dir_;
    // frozen-run probe (logging only): how often does the policy's top action equal the teacher's label?
    inline static sf_ppo::PPOConfig s_cfg_;
    inline static long s_pr_n_ = 0, s_pr_match_ = 0, s_pr_z_ = 0, s_pr_zhit_ = 0, s_pr_top_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, s_pr_lab_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    inline static double s_pr_pz_ = 0;
    sf_ppo::ValueNorm& vn_ = s_vn_;
    long& n_updates_   = s_updates_;
    long& total_steps_ = s_steps_;

    const int num_actions_ = 9, num_channels_ = 32, grid_x_ = 31, grid_y_ = 31, hidden_size_ = 160;

    AgentModel model_{nullptr};
    torch::optim::Adam* optimizer_ = nullptr;      // owned by s_opt_
    std::ofstream log_file_;

    // rollout buffer (all vectors are kept the same length when train() starts)
    std::vector<torch::Tensor> states_;
    std::vector<float> values_, logps_, rewards_, wmask_;
    std::vector<int>   actions_, expert_;
    long expert_steps_ = 0;
    double cls_sum_[4] = {0, 0, 0, 0}; long cls_n_[4] = {0, 0, 0, 0};   // p_teacher by label class (diagnostic)                        // teacher-executed steps in the current rollout (diagnostics)

    // teacher schedule: 1 at the first update, 0 after bc_decay_updates
    float schedule() const { return sf_ppo::teacher_schedule(n_updates_, cfg_.bc_decay_updates); }
    // predict() pushed a state for which update() never stored an action (human frame / game end): drop it
    void pop_dangling() {
        if (states_.size() > actions_.size()) {
            states_.pop_back(); values_.pop_back(); logps_.pop_back(); expert_.pop_back(); wmask_.pop_back();
        }
    }

    template<typename V> void log(const V& m) { log_file_ << m << "\n"; log_file_.flush(); }
    static std::string fmt(float v) { return std::to_string(v); }

    void clear_buffers() {
        states_.clear(); values_.clear(); logps_.clear(); rewards_.clear(); actions_.clear();
        expert_.clear(); wmask_.clear(); expert_steps_ = 0;
        pending_ = false;
        model_->reset_memory();
    }

    // ── critic helpers ───────────────────────────────────────────
    float denorm(float v) const { return cfg_.value_norm ? vn_.denorm(v) : v; }
    float normalize(float r) const { return cfg_.value_norm ? vn_.normalize(r) : r; }

    // PopArt: update (mu, sigma) from the rollout returns, then rescale the critic's last layer so that
    // every raw-scale prediction V = mu + sigma * head(x) is unchanged by the statistics update.
    void update_value_stats(const std::vector<float>& ret) {
        double old_mu, old_sigma;
        vn_.update(ret, cfg_.value_norm_beta, cfg_.value_sigma_min, /*first=*/n_updates_ == 0, old_mu, old_sigma);
        model_->popart_rescale(old_mu, old_sigma, vn_.mu, vn_.sigma);
    }

    // ── pipeline persistence ─────────────────────────────────────
    void load_train_state() {
        std::ifstream f(backup_dir_ + "/train_state.txt");
        if (!f) return;
        double m, sg, nu; long nu_up, ts;
        if (f >> m >> sg >> nu >> nu_up >> ts) {
            vn_.mu = m; vn_.sigma = sg; vn_.nu = nu; n_updates_ = nu_up; total_steps_ = ts;
            log("Loaded train state: mu=" + fmt((float)vn_.mu) + " sigma=" + fmt((float)vn_.sigma) +
                " updates=" + std::to_string(n_updates_));
        }
    }
    void append_stats(int T, float mean_r, float ret_mean, float ret_std, float pl, float vl, float ent,
                      float kl, float clipfrac, float ev, int epochs_run,
                      float bc, float p_teacher, float lambda_bc, float expert_frac) {
        if (cfg_.stats_csv.empty()) return;
        bool fresh = !std::ifstream(cfg_.stats_csv).good();
        std::ofstream f(cfg_.stats_csv, std::ios::app);
        if (!f) return;
        if (fresh)
            f << "update,total_steps,T,mean_reward,return_mean,return_std,policy_loss,value_loss,entropy,"
                 "approx_kl,clip_frac,explained_var,epochs_run,popart_mu,popart_sigma,bc_loss,p_teacher,lambda_bc,expert_frac\n";
        f << n_updates_ << ',' << total_steps_ << ',' << T << ',' << mean_r << ',' << ret_mean << ',' << ret_std
          << ',' << pl << ',' << vl << ',' << ent << ',' << kl << ',' << clipfrac << ',' << ev << ','
          << epochs_run << ',' << vn_.mu << ',' << vn_.sigma << ',' << bc << ',' << p_teacher << ',' << lambda_bc
          << ',' << expert_frac << '\n';
    }

    // ── one PPO update ───────────────────────────────────────────
    void train(float last_value) {
        int T = (int)std::min({states_.size(), values_.size(), logps_.size(), rewards_.size(), actions_.size(), expert_.size(), wmask_.size()});
        if (T < 2) { clear_buffers(); return; }
        states_.resize(T); values_.resize(T); logps_.resize(T); rewards_.resize(T); actions_.resize(T); expert_.resize(T); wmask_.resize(T);

        // (1) advantages and critic targets on the RAW reward scale
        std::vector<float> adv, ret;
        sf_ppo::gae(rewards_, values_, last_value, cfg_.gamma, cfg_.gae_lambda, adv, ret);

        float r_sum = 0.f, ret_mean = 0.f, ret_var = 0.f;
        for (float r : rewards_) r_sum += r;
        for (float r : ret) ret_mean += r;
        ret_mean /= T;
        for (float r : ret) ret_var += (r - ret_mean) * (r - ret_mean);
        const float ret_std = std::sqrt(ret_var / T);
        const float ev = sf_ppo::explained_variance(values_, ret);              // measured BEFORE this update

        // (2) critic normalisation: statistics first, output-preserving rescale, then normalised targets
        if (cfg_.value_norm) update_value_stats(ret);
        std::vector<float> ret_n(T);
        for (int t = 0; t < T; ++t) ret_n[t] = normalize(ret[t]);

        // (3) advantage standardisation (AFTER the critic target was formed)
        if (cfg_.normalize_adv) {
            float mu = 0.f, var = 0.f, wsum = 0.f;                       // statistics over on-policy steps only
            for (int t = 0; t < T; ++t) { mu += wmask_[t] * adv[t]; wsum += wmask_[t]; }
            mu /= std::max(wsum, 1.f);
            for (int t = 0; t < T; ++t) var += wmask_[t] * (adv[t] - mu) * (adv[t] - mu);
            // divide by max(std, floor): a rollout whose returns barely differ (no kill, no hit) carries no
            // information and must not be blown up to unit variance (that made the policy chase noise)
            float sd = std::max(std::sqrt(var / std::max(wsum, 1.f)), cfg_.adv_std_floor);
            for (float& a : adv) a = (a - mu) / sd;
        }

        auto adv_t = torch::tensor(adv), ret_t = torch::tensor(ret_n), old_logp = torch::tensor(logps_);

        // teacher labels: weight 1 where a label exists (index clamped to 0 where it does not)
        std::vector<float> lab_w(T);
        float lab_sum = 0.f, pol_sum = 0.f;
        for (int t = 0; t < T; ++t) { lab_w[t] = expert_[t] >= 0 ? 1.f : 0.f; lab_sum += lab_w[t]; pol_sum += wmask_[t]; }
        const float lambda_bc = cfg_.bc_coef * std::max(cfg_.bc_floor, schedule());
        const float inv_pol = 1.f / std::max(pol_sum, 1.f), inv_lab = 1.f / std::max(lab_sum, 1.f);
        auto w_t = torch::tensor(wmask_), lw_t = torch::tensor(lab_w);
        float last_bc = 0, last_pexp = 0, expert_frac = (float)expert_steps_ / (float)T;

        // (4) clipped-surrogate epochs over the ordered rollout (GRU memory replayed from zero)
        float last_pl = 0, last_vl = 0, last_ent = 0, last_kl = 0, last_clip = 0; int epochs_run = 0;
        for (int epoch = 0; epoch < cfg_.num_epochs; ++epoch) {
            model_->reset_memory();
            std::vector<torch::Tensor> lp(T), vv(T), en(T), ce(T);
            for (int t = 0; t < T; ++t) {
                auto [probs, value, z] = model_->forward(states_[t]);
                auto oh = torch::zeros({num_actions_});
                oh[actions_[t]] = 1.0f;
                model_->update_action_history(oh);
                lp[t] = torch::log(probs[actions_[t]] + 1e-8f);
                vv[t] = value;                                          // normalised units
                en[t] = -(probs * torch::log(probs + 1e-8f)).sum();
                ce[t] = -torch::log(probs[expert_[t] >= 0 ? expert_[t] : 0] + 1e-8f);   // -log pi(teacher action | s)
            }
            auto logp  = torch::stack(lp), vals = torch::stack(vv), ent = torch::stack(en), cet = torch::stack(ce);
            auto ratio = torch::exp(logp - old_logp);
            auto s1 = ratio * adv_t;
            auto s2 = torch::clamp(ratio, 1.f - cfg_.ppo_clip, 1.f + cfg_.ppo_clip) * adv_t;
            auto p_loss = -inv_pol * (torch::min(s1, s2) * w_t).sum();          // on-policy steps only
            auto v_loss = torch::mse_loss(vals, ret_t);                 // regression in normalised units
            auto entropy = ent.mean();
            auto bc_loss = inv_lab * (cet * lw_t).sum();                          // imitation of the teacher on every visited state
            auto loss = cfg_.policy_coef * p_loss + cfg_.value_coef * v_loss - cfg_.entropy_coef * entropy + lambda_bc * bc_loss;

            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.max_grad_norm);
            optimizer_->step();

            last_pl = p_loss.item<float>(); last_vl = v_loss.item<float>();
            last_ent = entropy.item<float>();
            last_bc = bc_loss.item<float>();
            last_pexp = inv_lab * (torch::exp(-cet) * lw_t).sum().item<float>();   // mean probability given to the teacher's action
            {   // same quantity split by teacher label class: punch (z) | turn (q,e) | move (a,w,s,d) | idle/x
                auto cc = torch::exp(-cet.detach()).contiguous(); const float* q = cc.data_ptr<float>();
                for (int k = 0; k < 4; ++k) { cls_sum_[k] = 0; cls_n_[k] = 0; }
                for (int t = 0; t < T; ++t) if (expert_[t] >= 0) {
                    const int k = expert_[t] == 2 ? 0 : (expert_[t] == 3 || expert_[t] == 4) ? 1 : (expert_[t] >= 5 ? 2 : 3);
                    cls_sum_[k] += q[t]; ++cls_n_[k];
                }
            }
            last_kl = (inv_pol * ((old_logp - logp.detach()) * w_t).sum()).item<float>();   // approx KL(old || new)
            last_clip = (inv_pol * (((ratio.detach() - 1.0f).abs() > cfg_.ppo_clip).to(torch::kFloat32) * w_t).sum()).item<float>();
            ++epochs_run;
            if (cfg_.policy_coef > 0.f && last_kl > 1.5f * cfg_.target_kl) break;                   // policy moved enough
        }

        // (5) bookkeeping: counters, logs, CSV, checkpoints
        ++n_updates_; total_steps_ += T;
        log("PPO#" + std::to_string(n_updates_) + " T=" + std::to_string(T) +
            " mean_r=" + fmt(r_sum / T) + " policy_loss=" + fmt(last_pl) + " value_loss=" + fmt(last_vl) +
            " entropy=" + fmt(last_ent) + " approx_kl=" + fmt(last_kl) + " clip_frac=" + fmt(last_clip) +
            " EV=" + fmt(ev) + " bc_loss=" + fmt(last_bc) + " p_teacher=" + fmt(last_pexp) + " lambda_bc=" + fmt(lambda_bc) + " expert_frac=" + fmt(expert_frac) +
            " | p_teacher by label: punch=" + fmt(cls_n_[0] ? (float)(cls_sum_[0] / cls_n_[0]) : -1.f) + " (n=" + std::to_string(cls_n_[0]) + ")" +
            " turn=" + fmt(cls_n_[1] ? (float)(cls_sum_[1] / cls_n_[1]) : -1.f) + " (n=" + std::to_string(cls_n_[1]) + ")" +
            " move=" + fmt(cls_n_[2] ? (float)(cls_sum_[2] / cls_n_[2]) : -1.f) + " (n=" + std::to_string(cls_n_[2]) + ")" +
            " idle/x=" + fmt(cls_n_[3] ? (float)(cls_sum_[3] / cls_n_[3]) : -1.f) + " (n=" + std::to_string(cls_n_[3]) + ")" + " |" + " mu=" + fmt((float)vn_.mu) + " sigma=" + fmt((float)vn_.sigma) +
            " epochs_run=" + std::to_string(epochs_run));
        append_stats(T, r_sum / T, ret_mean, ret_std, last_pl, last_vl, last_ent, last_kl, last_clip, ev, epochs_run, last_bc, last_pexp, lambda_bc, expert_frac);
        if (cfg_.save_every_updates > 0 && n_updates_ % cfg_.save_every_updates == 0) {
            torch::save(model_, backup_dir_ + "/checkpoints/model_u" + std::to_string(n_updates_) + ".pt");
        }
        clear_buffers();
    }
};
