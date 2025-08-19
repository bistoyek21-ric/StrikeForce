#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <memory>
#include <algorithm>

// تعریف ResidualBlock
class ResidualBlock : public torch::nn::Module {
public:
    ResidualBlock(int size) {
        linear1 = register_module("linear1", torch::nn::Linear(size, size));
        linear2 = register_module("linear2", torch::nn::Linear(size, size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(linear1->forward(x));
        x = linear2->forward(x);
        return x + residual;
    }
    
private:
    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear linear2{nullptr};
};

// تعریف RewardNet (ساده‌شده)
class RewardNet {
public:
    RewardNet(bool training, int T, float learning_rate, float alpha, 
              const std::string& backup_dir, int num_actions, 
              int num_channels, int grid_size) {
        // پیاده‌سازی ساده RewardNet
    }

    float get_reward(int action, bool imitate, const torch::Tensor& state) {
        // پیاده‌سازی ساده get_reward
        return 1.0f; // مقدار نمونه
    }
};

class Agent {
private:
    bool manual;
    int cnt = 0;

    std::mutex mtx;
    std::condition_variable cv;
    bool is_training = false, not_training;
    std::thread trainThread;

    bool training, logging = false;
    int hidden_size, num_actions, T, num_epochs, num_channels, grid_size;
    float gamma, learning_rate, ppo_clip, alpha, c_v;
    
    std::string backup_dir;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<torch::Tensor> log_probs, states, values;

    torch::Tensor probs;

    torch::Device device;

    // --- تغییرات برای Hot-Swapping و حفظ stateها ---
    std::mutex model_mutex_;
    
    // استفاده از shared_ptr برای مدیریت مدل‌ها
    std::shared_ptr<torch::nn::Sequential> cnn_, policy_head_, value_head_, gate_, action_processor_;
    std::shared_ptr<torch::nn::GRU> gru_;
    std::shared_ptr<torch::nn::LayerNorm> gru_norm_, action_norm_;
    std::shared_ptr<torch::optim::AdamW> optimizer_;

    // stateهای داخلی که باید حفظ شوند
    torch::Tensor current_h_state;
    torch::Tensor current_action_input;

    std::ofstream log_file;

    // RewardNet را به صورت اشاره‌گر هوشمند تعریف می‌کنیم
    std::shared_ptr<RewardNet> reward_net;

    template<typename T>
    void log(const T& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    // تابع کمکی برای کپی کردن مدل‌ها
    template<typename ModuleType>
    std::shared_ptr<ModuleType> copy_module(const std::shared_ptr<ModuleType>& module) {
        // ایجاد یک کپی از ماژول با سریالایز و دی-سریالایز
        std::stringstream stream;
        torch::serialize::OutputArchive output_archive;
        module->save(output_archive);
        output_archive.save_to(stream);
        
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(stream);
        
        auto new_module = std::make_shared<ModuleType>();
        new_module->load(input_archive);
        return new_module;
    }

    // تابع کمکی برای گرفتن همه پارامترها
    std::vector<torch::Tensor> get_all_parameters() {
        std::vector<torch::Tensor> params;
        
        auto add_params = [&params](const torch::nn::Module& module) {
            for (const auto& param : module.parameters()) {
                params.push_back(param);
            }
        };
        
        add_params(*cnn_);
        add_params(*gru_);
        add_params(*action_processor_);
        add_params(*policy_head_);
        add_params(*value_head_);
        add_params(*gate_);
        add_params(*gru_norm_);
        add_params(*action_norm_);
        
        return params;
    }

    void saveProgress() {
        if (backup_dir.empty())
            return;
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_size << " " << num_actions << " ";
        dim.close();
        
        std::ofstream coef(backup_dir + "/c_v.txt");
        coef << c_v;
        coef.close();
        
        // قفل برای دسترسی امن به مدل‌ها
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        // ذخیره مدل‌ها
        torch::save(*cnn_, backup_dir + "/cnn.pt");
        torch::save(*gru_, backup_dir + "/gru.pt");
        torch::save(*action_processor_, backup_dir + "/action_processor.pt");
        torch::save(*policy_head_, backup_dir + "/policy_head.pt");
        torch::save(*value_head_, backup_dir + "/value_head.pt");
        torch::save(*gate_, backup_dir + "/gate.pt");
        torch::save(*gru_norm_, backup_dir + "/gru_norm.pt");
        torch::save(*action_norm_, backup_dir + "/action_norm.pt");
        
        // ذخیره stateها
        torch::save(current_h_state, backup_dir + "/h_state.pt");
        torch::save(current_action_input, backup_dir + "/action_input.pt");
        
        log("Progress saved to " + backup_dir);
    }

    void load_progress() {
        std::ifstream dim(backup_dir + "/dim.txt");
        if (!dim.is_open()) {
            log("Failed to open dim.txt");
            throw std::runtime_error("Failed to open dim.txt");
        }
        dim >> num_channels >> grid_size >> num_actions;
        log("Loaded dimensions: num_channels=" + std::to_string(num_channels) +
            ", grid_size=" + std::to_string(grid_size) +
            ", num_actions=" + std::to_string(num_actions));
        dim.close();
        
        std::ifstream coef(backup_dir + "/c_v.txt");
        coef >> c_v;
        coef.close();
        
        initializeNetwork();
        
        // قفل برای دسترسی امن به مدل‌ها
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        // بارگذاری مدل‌ها
        torch::load(*cnn_, backup_dir + "/cnn.pt");
        torch::load(*gru_, backup_dir + "/gru.pt");
        torch::load(*action_processor_, backup_dir + "/action_processor.pt");
        torch::load(*policy_head_, backup_dir + "/policy_head.pt");
        torch::load(*value_head_, backup_dir + "/value_head.pt");
        torch::load(*gate_, backup_dir + "/gate.pt");
        torch::load(*gru_norm_, backup_dir + "/gru_norm.pt");
        torch::load(*action_norm_, backup_dir + "/action_norm.pt");
        
        // بارگذاری stateها
        torch::load(current_h_state, backup_dir + "/h_state.pt");
        torch::load(current_action_input, backup_dir + "/action_input.pt");
        
        log("Progress loaded successfully");
    }

    void initializeNetwork() {
        log("Start:");
        hidden_size = 128;
        log("hidden_size set to " + std::to_string(hidden_size));
        
        // ایجاد مدل‌ها با shared_ptr
        cnn_ = std::make_shared<torch::nn::Sequential>(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 32, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size),
            torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1),
            torch::nn::Flatten()
        );
        
        gru_ = std::make_shared<torch::nn::GRU>(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2));
        
        action_processor_ = std::make_shared<torch::nn::Sequential>(
            torch::nn::Linear(num_actions, hidden_size),
            torch::nn::ReLU(),
            ResidualBlock(hidden_size)
        );
        
        gate_ = std::make_shared<torch::nn::Sequential>(
            torch::nn::Linear(2 * hidden_size, hidden_size),
            torch::nn::ReLU(),
            ResidualBlock(hidden_size)
        );
        
        policy_head_ = std::make_shared<torch::nn::Sequential>(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            torch::nn::Linear(hidden_size, num_actions),
            torch::nn::ReLU()
        );
        
        value_head_ = std::make_shared<torch::nn::Sequential>(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            torch::nn::Linear(hidden_size, 1),
            torch::nn::Sigmoid()
        );
        
        gru_norm_ = std::make_shared<torch::nn::LayerNorm>(torch::nn::LayerNormOptions({hidden_size}));
        action_norm_ = std::make_shared<torch::nn::LayerNorm>(torch::nn::LayerNormOptions({hidden_size}));

        current_h_state = torch::zeros({2, 1, hidden_size});
        current_action_input = torch::zeros({num_actions});
    }

    torch::Tensor computeReturns() {
        torch::Tensor returns = torch::zeros({static_cast<int64_t>(rewards.size())}, device);
        double running = 0;
        for (int i = static_cast<int>(rewards.size()) - 1; i >= 0; --i) {
            running = rewards[i] + gamma * running;
            returns[i] = running;
        }
        return returns;
    }

    std::vector<torch::Tensor> forward(const torch::Tensor &state) {
        // قفل برای دسترسی امن به مدل‌ها
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        auto feat = cnn_->forward(state);
        feat = feat.view({1, 1, -1});
        auto r = gru_->forward(feat, current_h_state);
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = gru_norm_->forward(out_seq);
        current_h_state = std::get<1>(r).detach();
        auto a_feat = action_processor_->forward(current_action_input);
        a_feat = action_norm_->forward(a_feat.view({-1}));
        auto gate_input = torch::cat({out_seq, a_feat});
        auto gated = gate_->forward(gate_input);
        auto logits = policy_head_->forward(gated);
        auto p = torch::softmax(logits, -1);
        auto v = 2 * value_head_->forward(gated).squeeze() - 1;
        return {p, v};
    }

    // تابع forward جداگانه برای آموزش که مدل‌ها را به عنوان پارامتر می‌گیرد
    std::vector<torch::Tensor> forward_train(const torch::Tensor &state, 
                                           const std::shared_ptr<torch::nn::Sequential>& cnn,
                                           const std::shared_ptr<torch::nn::GRU>& gru,
                                           const std::shared_ptr<torch::nn::Sequential>& action_processor,
                                           const std::shared_ptr<torch::nn::Sequential>& policy_head,
                                           const std::shared_ptr<torch::nn::Sequential>& value_head,
                                           const std::shared_ptr<torch::nn::Sequential>& gate,
                                           const std::shared_ptr<torch::nn::LayerNorm>& gru_norm,
                                           const std::shared_ptr<torch::nn::LayerNorm>& action_norm,
                                           torch::Tensor& h_state_train,
                                           torch::Tensor& action_input_train) {
        auto feat = cnn->forward(state);
        feat = feat.view({1, 1, -1});
        auto r = gru->forward(feat, h_state_train);
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = gru_norm->forward(out_seq);
        h_state_train = std::get<1>(r).detach();
        auto a_feat = action_processor->forward(action_input_train);
        a_feat = action_norm->forward(a_feat.view({-1}));
        auto gate_input = torch::cat({out_seq, a_feat});
        auto gated = gate->forward(gate_input);
        auto logits = policy_head->forward(gated);
        auto p = torch::softmax(logits, -1);
        auto v = 2 * value_head->forward(gated).squeeze() - 1;
        return {p, v};
    }

    void train() {
        // 1. ایجاد کپی از تمام مدل‌ها و optimizer
        std::shared_ptr<torch::nn::Sequential> cnn_copy, policy_head_copy, value_head_copy, gate_copy, action_processor_copy;
        std::shared_ptr<torch::nn::GRU> gru_copy;
        std::shared_ptr<torch::nn::LayerNorm> gru_norm_copy, action_norm_copy;
        std::shared_ptr<torch::optim::AdamW> optimizer_copy;
        
        // 2. ذخیره stateهای فعلی قبل از قفل
        torch::Tensor saved_h_state;
        torch::Tensor saved_action_input;
        
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            saved_h_state = current_h_state.clone();
            saved_action_input = current_action_input.clone();
            
            // کپی کردن مدل‌ها
            cnn_copy = copy_module(cnn_);
            gru_copy = copy_module(gru_);
            action_processor_copy = copy_module(action_processor_);
            policy_head_copy = copy_module(policy_head_);
            value_head_copy = copy_module(value_head_);
            gate_copy = copy_module(gate_);
            gru_norm_copy = copy_module(gru_norm_);
            action_norm_copy = copy_module(action_norm_);
            
            // کپی کردن پارامترهای optimizer
            auto params = get_all_parameters();
            optimizer_copy = std::make_shared<torch::optim::AdamW>(params, torch::optim::AdamWOptions().lr(learning_rate).weight_decay(1e-4));
            
            // کپی کردن state optimizer
            torch::serialize::OutputArchive archive;
            optimizer_->save(archive);
            optimizer_copy->load(archive);
        }
        
        // انتقال کپی‌ها به device مناسب
        cnn_copy->to(device);
        gru_copy->to(device);
        action_processor_copy->to(device);
        policy_head_copy->to(device);
        value_head_copy->to(device);
        gate_copy->to(device);
        gru_norm_copy->to(device);
        action_norm_copy->to(device);
        
        // 3. آموزش روی کپی‌ها
        auto returns = computeReturns();
        std::vector<torch::Tensor> advantages;
        for (size_t i = 0; i < rewards.size(); ++i) {
            advantages.push_back(returns[i] - values[i]);
        }
        torch::Tensor adv_tensor = torch::stack(advantages);
        torch::Tensor adv_mean = adv_tensor.mean();
        torch::Tensor adv_std = adv_tensor.std();
        if (adv_std.item<float>() < 1e-8) {
            adv_tensor = adv_tensor - adv_mean;
        } else {
            adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-5);
        }
        
        // استفاده از stateهای ذخیره شده برای آموزش
        torch::Tensor h_state_train = saved_h_state.clone();
        torch::Tensor action_input_train = saved_action_input.clone();
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            torch::Tensor p_loss = torch::zeros({}, device);
            torch::Tensor v_loss = torch::zeros({}, device);
            
            for (size_t i = 0; i < states.size(); ++i) {
                auto output = forward_train(states[i], cnn_copy, gru_copy, action_processor_copy, 
                                          policy_head_copy, value_head_copy, gate_copy,
                                          gru_norm_copy, action_norm_copy,
                                          h_state_train, action_input_train);
                
                action_input_train = action_input_train * alpha;
                action_input_train[actions[i]] += 1 - alpha;
                
                auto current_logp = torch::log(output[0])[actions[i]];
                auto diff = torch::clamp(current_logp - log_probs[i], -100, 10);
                auto ratio = torch::exp(diff);
                auto clipped = torch::clamp(ratio, 1 - ppo_clip, 1 + ppo_clip);
                auto adv = adv_tensor[i];
                
                p_loss -= torch::min(ratio * adv, clipped * adv);
                v_loss += torch::mse_loss(output[1], returns[i]);
            }
            
            v_loss /= static_cast<float>(states.size());
            auto loss = p_loss + c_v * v_loss;
            
            log("Epoch " + std::to_string(epoch) + ": loss=" + std::to_string(loss.item<float>()));
            
            optimizer_copy->zero_grad();
            loss.backward();
            optimizer_copy->step();
            
            log("*");
        }
        
        // 4. ذخیره stateهای نهایی از آموزش
        torch::Tensor final_h_state = h_state_train.clone();
        torch::Tensor final_action_input = action_input_train.clone();
        
        // 5. جایگزینی مدل اصلی با مدل آموزش دیده و stateهای جدید
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            cnn_ = cnn_copy;
            gru_ = gru_copy;
            action_processor_ = action_processor_copy;
            policy_head_ = policy_head_copy;
            value_head_ = value_head_copy;
            gate_ = gate_copy;
            gru_norm_ = gru_norm_copy;
            action_norm_ = action_norm_copy;
            optimizer_ = optimizer_copy;
            
            // به روز رسانی stateهای داخلی با stateهای آموزش دیده
            current_h_state = final_h_state;
            current_action_input = final_action_input;
        }
        
        log("Training completed");
        actions.clear();
        rewards.clear();
        log_probs.clear();
        states.clear();
        values.clear();
        not_training = true;
    }

public:
    Agent(bool _training, int _T, int _num_epochs, float _gamma, float _learning_rate, float _ppo_clip,
          float _alpha, const std::string& _backup_dir = "bots/bot-1/backup/agent_backup", int _num_channels = 1,
          int _grid_size = 1, int _num_actions = 1)
        : training(_training), T(_T), num_epochs(_num_epochs), gamma(_gamma), learning_rate(_learning_rate),
          ppo_clip(_ppo_clip), alpha(_alpha), backup_dir(_backup_dir),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          num_channels(_num_channels), grid_size(_grid_size), num_actions(_num_actions) {
        c_v = 0.05;
        
        if (!backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        } else {
            if (!backup_dir.empty()) {
                std::filesystem::create_directories(backup_dir);
            }
            initializeNetwork();
        }
        
        log_file.open(backup_dir + "/agent_log.log");
        log(std::to_string(torch::cuda::is_available()));
        
        cnn_->to(device);
        gru_->to(device);
        action_processor_->to(device);
        policy_head_->to(device);
        value_head_->to(device);
        gate_->to(device);
        gru_norm_->to(device);
        action_norm_->to(device);
        
        current_h_state = current_h_state.to(device);
        current_action_input = current_action_input.to(device);
        
        auto params = get_all_parameters();
        optimizer_ = std::make_shared<torch::optim::AdamW>(params, torch::optim::AdamWOptions().lr(learning_rate).weight_decay(1e-4));
        
        reward_net = std::make_shared<RewardNet>(true, T, learning_rate, alpha, backup_dir + "/../reward_backup", num_actions, num_channels, grid_size);
        
        log("--------------------------------------------------------");
        auto t = time(nullptr);
        log(std::string(ctime(&t)));
        log("--------------------------------------------------------");
        log("Agent initialized with T=" + std::to_string(T) + ", num_epochs=" + std::to_string(num_epochs) +
            ", gamma=" + std::to_string(gamma) + ", alpha=" + std::to_string(alpha) + ", learning_rate=" + std::to_string(learning_rate) +
            ", ppo_clip=" + std::to_string(ppo_clip));
        
        cnt = T + 1;
    }

    ~Agent() {
        if (is_training && trainThread.joinable()) {
            std::cout << "Agent Network is updating...\nthis might take a few seconds" << std::endl;
            trainThread.join();
            std::cout << "done!" << std::endl;
        }
        
        if (training) {
            saveProgress();
        }
        
        log_file.close();
    }

    int predict(const std::vector<float>& obs) {
        if (cnt <= T) {
            return 0;
        }
        
        if (is_training) {
            if (not_training) {
                is_training = false;
                cv.notify_all();
                if (trainThread.joinable()) {
                    trainThread.join();
                }
            } else {
                return 0;
            }
        }
        
        auto state_tensor = torch::tensor(obs, torch::dtype(torch::kFloat32).device(device)).view({1, num_channels, grid_size, grid_size});
        states.push_back(state_tensor);
        
        auto output = forward(state_tensor);
        values.push_back(output[1].detach());
        probs = output[0];
        
        std::vector<float> v(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        
        return dist(gen);
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T) {
            return;
        }
        
        float reward = reward_net->get_reward(action, imitate, states.back());
        if (reward == -2) {
            states.pop_back();
            values.pop_back();
            return;
        }
        
        rewards.push_back(reward);
        actions.push_back(action);
        log_probs.push_back(torch::log(probs.detach()[action] + 1e-8));
        
        // به روز رسانی current_action_input با قفل
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            current_action_input = current_action_input * alpha;
            current_action_input[action] += 1 - alpha;
        }
        
        if (actions.size() == T) {
            if (training) {
                is_training = true;
                not_training = false;
                trainThread = std::thread(&Agent::train, this);
            } else {
                actions.clear();
                rewards.clear();
                log_probs.clear();
                states.clear();
                values.clear();
            }
        }
    }

    bool is_manual() {
        if (!is_training && cnt <= T) {
            ++cnt;
        }
        
        if (is_training) {
            if (not_training) {
                is_training = false;
                cv.notify_all();
                if (trainThread.joinable()) {
                    trainThread.join();
                }
                std::mt19937 gen(std::random_device{}());
                std::uniform_int_distribution<> dist(0, 1);
                manual = dist(gen);
            } else {
                return true;
            }
        }
        
        if (cnt <= T) {
            manual = true;
        }
        
        if (cnt == T + 1) {
            std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<> dist(0, 1);
            manual = dist(gen);
            ++cnt;
        }
        
        if (actions.size() == T / 2) {
            manual = !manual;
        }
        
        return manual;
    }
};