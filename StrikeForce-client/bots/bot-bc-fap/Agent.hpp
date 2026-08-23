/*
MIT License

Copyright (c) 2025 bistoyek21 R.I.C.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system && ./app
// Add -lws2_32 on Windows

#include "Modules.hpp"          // brings PlayerPolicyNet, TORCH_MODULE etc.

const std::string bot_code = "bot-bc-fap", backup_path = "bots/bot-bc-fap/backup";

class Agent {
public:
    Agent(bool data_gathering_mode = true,
          bool inference_mode = false,
          int T = 1054,
          const std::string &model_dir = "bots/bot-bc-fap/backup",
          const std::string &dataset_dir = "bots/bot-bc-fap/dataset/data_train")
        : data_gathering_mode_(data_gathering_mode),
          inference_mode_(inference_mode),
          T_(T),
          model_dir_(model_dir),
          dataset_dir_(dataset_dir)
    {
        prev_action_ = torch::tensor({5}, torch::kLong);
        
        // 1. Load model
        if(inference_mode_)
            model_ = PlayerPolicyNet();
        
        if (!model_dir_.empty()) {
            if (std::filesystem::exists(model_dir_)) {
                if (std::filesystem::exists(model_dir_ + "/model.pt") && inference_mode_)
                    torch::load(model_, model_dir_ + "/model.pt");
                 log_file_.open(model_dir_ + "/agent_log.log", std::ios::app);
            } else {
                std::filesystem::create_directories(model_dir_);
                log_file_.open(model_dir_ + "/agent_log.log", std::ios::app);
            }
        }

        model_->to(torch::kCPU);

        // 2. Prepare dataset directory
        if (!dataset_dir_.empty()) {
            if (!std::filesystem::exists(dataset_dir_))
                std::filesystem::create_directories(dataset_dir_);
            // Find next episode number
            episode_count_ = 0;
            for (const auto& entry : std::filesystem::directory_iterator(dataset_dir_)) {
                std::string fname = entry.path().filename().string();
                if (fname.find("episode_") == 0 && fname.size() > 10 && fname.substr(fname.size()-3) == ".pt") {
                    try {
                        int num = std::stoi(fname.substr(8, fname.size() - 11));
                        if (num + 1 > episode_count_)
                            episode_count_ = num + 1;
                    } catch (...) {}
                }
            }
        }

        if (inference_mode_)
            log("=========\nAgent parameters: " + std::to_string(param_count()));
        else
            log("=========\n---------");
        
        log("Inference mode : " + std::to_string(inference_mode_));
        log("Data gathering : " + std::to_string(data_gathering_mode_));
        log("T              : " + std::to_string(T_));
        log("Model path     : " + model_dir_ + "/model.pt");
        log("Dataset dir    : " + dataset_dir_);
        log("Next episode   : " + std::to_string(episode_count_));

        if (inference_mode_) {
            model_->eval();          // always eval – no training gradients needed
            auto dummy = torch::zeros({1, num_channels_, grid_size_, grid_size_});
            model_->forward(dummy, prev_action_);
            model_->reset_memory();
        }

        freq_ = torch::zeros({num_actions_});
    }

    ~Agent() {
        log_file_.close();
    }

    // ---------------------------------------------------------------
    //  predict – called every environment step
    // ---------------------------------------------------------------
    int predict(const std::vector<float>& obs) {
        if (cnt_ <= T_initial_)
            return 0;                     // initial warm‑up

        // During warm‑up (before T_warm_up_) we play randomly
        if (!warmup_finished_)
            return rand() % num_actions_;
        
        auto state = torch::tensor(obs, torch::kFloat32).view({1, num_channels_, grid_size_, grid_size_});

        // Store state for episode saving (if data gathering is on)
        if (data_gathering_mode_)
            episode_states_.push_back(state.clone());

        // After warm‑up, behaviour depends on inference_mode_
        if (inference_mode_) {
            // Model decides
            torch::NoGradGuard no_grad;
            auto out = model_->forward(state, prev_action_);
            auto logits = out[1];                     // [1, num_actions_]
            auto probs = torch::softmax(logits, 1);
            return probs.argmax().item<int>();
        }

        // inference_mode_ == false → human plays (manual), predict is not really used
        return 0;
    }

    // ---------------------------------------------------------------
    //  update – called after the action is actually executed
    // ---------------------------------------------------------------
    void update(int action, bool imitate) {
        if (cnt_ <= T_initial_ || !warmup_finished_)
            return;
        
        if(action < 0 || num_actions_ < action)
            action = 0;
        
        // Update RNN state for next step (always needed)
        prev_action_ = torch::tensor({action}, torch::kLong);

        freq_[action] += 1;

        // Recording: only if data_gathering_mode_ is true and we are past warm‑up
        if (data_gathering_mode_) {
            episode_actions_.push_back(prev_action_.clone());
            // When the buffer fills up to T_, a full episode is complete
            if (episode_actions_.size() == T_) {
                save_episode();
                cnt_warm_up_ = 0;
                warmup_finished_ = false;
            }
        }
    }

    // ---------------------------------------------------------------
    //  is_manual – tells the environment whether to listen to human input
    // ---------------------------------------------------------------
    bool is_manual() {
        if (cnt_ < T_initial_) {
            ++cnt_;
            return true;
        }

        // During session warm‑up (first T_warm_up_ frames after T_initial_) → random, not manual
        if (cnt_ >= T_initial_ && cnt_warm_up_ < T_warm_up_) {
            ++cnt_warm_up_;
            ++cnt_;
            if (cnt_warm_up_ == T_warm_up_) {
                warmup_finished_ = true;

                log("---------");
                
                return !inference_mode_;
            }
            return false;   // not manual yet
        }

        // After warm‑up, manual = NOT inference_mode_
        return !inference_mode_;
    }

    bool in_training() { return data_gathering_mode_; }

private:
    // ---------------------------------------------------------------
    //  Episode saving (data gathering)
    // ---------------------------------------------------------------
    void save_episode() {
        if (episode_states_.size() != episode_actions_.size()) {
            std::cerr << "Error: state/action count mismatch!\n";
            std::cerr << "states: " << episode_states_.size() << '\n';
            std::cerr << "actions:" << episode_actions_.size() << '\n';
            std::cerr << "=====================================\n";
            return;
        }

        // Stack into tensors: [1, T, C, H, W] and [1, T]
        auto states_tensor = torch::stack(episode_states_, 0).squeeze(1).unsqueeze(0);
        auto actions_tensor = torch::stack(episode_actions_, 0).squeeze(-1).unsqueeze(0);

        std::string path = dataset_dir_ + "/episode_" + std::to_string(episode_count_) + ".pt";
        torch::serialize::OutputArchive archive;
        archive.write("states", states_tensor);
        archive.write("actions", actions_tensor);
        archive.save_to(path);


        log("Episode saved: " + path + " (" + std::to_string(states_tensor.size(0)) + " frames)");
        log("frequency:");
        log(freq_ / states_tensor.size(0));

        episode_count_++;

        freq_ = torch::zeros({num_actions_});

        // Clean up
        episode_states_.clear();
        episode_actions_.clear();
        
        if (inference_mode_)
            model_->reset_memory();   // reset sliding window for next episode
        std::cout << "Episode saved!" << std::endl;
    }

    // ---------------------------------------------------------------
    //  logging helper
    // ---------------------------------------------------------------
    template<typename T>
    void log(const T& msg) {
        if (log_file_.is_open()) {
            log_file_ << msg << std::endl;
            log_file_.flush();
        }
    }

    int64_t param_count() {
        int64_t n = 0;
        for (const auto& p : model_->parameters())
            n += p.numel();
        return n;
    }

    // ---------------------------------------------------------------
    //  member variables
    // ---------------------------------------------------------------
    bool inference_mode_, data_gathering_mode_;
    bool warmup_finished_ = false;
    int T_;
    int cnt_ = 0, cnt_warm_up_ = 0;
    static constexpr int T_initial_ = 512, T_warm_up_ = 128;
    static constexpr int num_actions_ = 7, num_channels_ = 32, grid_size_ = 31;

    std::string model_dir_, dataset_dir_;
    PlayerPolicyNet model_{nullptr};

    torch::Tensor prev_action_, freq_;
    std::vector<torch::Tensor> episode_states_;   // raw observations [1,C,H,W] each step
    std::vector<torch::Tensor> episode_actions_;  // action tensors [1]

    int episode_count_ = 0;
    std::ofstream log_file_;
};