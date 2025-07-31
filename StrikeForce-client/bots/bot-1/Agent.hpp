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
//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lsfml-graphics -lsfml-window -lsfml-system
#include "RewardNet.hpp"

class Agent {
private:
    bool training, logging = true;
    int hidden_size, num_actions, T, num_epochs, num_channels, grid_size;
    float gamma, learning_rate, ppo_clip, alpha;
    std::string backup_dir;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<torch::Tensor> log_probs, gated_outputs;

    torch::Tensor state, action_input;

    torch::Device device;

    torch::nn::GRU gru{nullptr};
    torch::nn::Linear policy_head{nullptr}, value_head{nullptr};
    torch::nn::Sequential cnn{nullptr}, gate_layer{nullptr};
    torch::optim::AdamW* optimizer{nullptr};

    std::ofstream log_file;

    RewardNet* reward_net;

    template<typename T>
    void log(const T& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    void saveProgress() {
        if (backup_dir.empty())
            return;
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_size << " " << num_actions << " ";
        dim.close();
        gru->to(torch::kCPU);
        policy_head->to(torch::kCPU);
        value_head->to(torch::kCPU);
        cnn->to(torch::kCPU);
        gate_layer->to(torch::kCPU);
        torch::save(gru, backup_dir + "/gru.pt");
        torch::save(policy_head, backup_dir + "/policy_head.pt");
        torch::save(value_head, backup_dir + "/value_head.pt");
        torch::save(cnn, backup_dir + "/cnn.pt");
        torch::save(gate_layer, backup_dir + "/gate_layer.pt");
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
        initializeNetwork();
        torch::load(cnn, backup_dir + "/cnn.pt");
        torch::load(gru, backup_dir + "/gru.pt");
        torch::load(policy_head, backup_dir + "/policy_head.pt");
        torch::load(value_head, backup_dir + "/value_head.pt");
        torch::load(gate_layer, backup_dir + "/gate_layer.pt");
        log("Progress loaded successfully");
    }

    void initializeNetwork() {
        log("Start:");
        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 32, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128), torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1), torch::nn::Flatten()
        );
        hidden_size = 128;
        log("hidden_size set to " + std::to_string(hidden_size));
        gru = torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2));
        gate_layer = torch::nn::Sequential(
            torch::nn::Linear(num_actions + hidden_size, hidden_size),
            torch::nn::ReLU()
        );
        policy_head = torch::nn::Linear(hidden_size, num_actions);
        value_head = torch::nn::Linear(hidden_size, 1);
    }

    torch::Tensor computeReturns() {
        torch::Tensor returns = torch::zeros({(int)rewards.size()}, device);
        double running = 0;
        for (int i = T - 1; i >= 0; --i) {
            running = rewards[i] + gamma * running;
            returns[i] = running;
        }
        return returns;
    }

    void train() {
        auto returns = computeReturns();
        std::vector<torch::Tensor> advantages;
        for (int i = 0; i < T; ++i) {
            auto value = value_head->forward(gated_outputs[i]).squeeze();
            advantages.push_back(returns[i] - value);
        }
        torch::Tensor adv_tensor = torch::stack(advantages);
        torch::Tensor adv_mean = adv_tensor.mean();
        torch::Tensor adv_std = adv_tensor.std();
        if (adv_std.item<float>() < 1e-8)
            adv_tensor = adv_tensor - adv_mean;
        else 
            adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-5);
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            torch::Tensor p_loss = torch::zeros({}, device);
            torch::Tensor v_loss = torch::zeros({}, device);
            for (int i = 0; i < T; ++i) {
                auto old_lp = log_probs[i][actions[i]];
                auto current_logits = policy_head->forward(gated_outputs[i]);
                auto current_logp = torch::log_softmax(current_logits, -1)[actions[i]];
                auto diff = current_logp - old_lp;
                if (diff.item<float>() > 10.0f)
                    diff = torch::tensor(10.0f).to(device);
                auto ratio = torch::exp(diff);
                auto clipped = torch::clamp(ratio, 1 - ppo_clip, 1 + ppo_clip);
                auto adv = adv_tensor[i];
                p_loss -= torch::min(ratio * adv, clipped * adv);
                auto current_value = value_head->forward(gated_outputs[i]).squeeze();
                v_loss += torch::mse_loss(current_value, returns[i]);
            }
            auto loss = p_loss + 0.5 * v_loss;
            log("Epoch " + std::to_string(epoch) + ": loss=" + std::to_string(loss.item<float>()));
            optimizer->zero_grad();
            loss.backward({}, /*retain_graph=*/true);
            optimizer->step();
        }
        log("Training completed");
    }

public:
    Agent(bool _training, int _T, int _num_epochs, float _gamma, float _learning_rate, float _ppo_clip,
          float _alpha, const std::string& _backup_dir = "bots/bot-1/backup/agent_backup", int _num_channels = 1,
          int _grid_size = 1, int _num_actions = 1)
        : training(_training), T(_T), num_epochs(_num_epochs), gamma(_gamma), learning_rate(_learning_rate),
          ppo_clip(_ppo_clip), alpha(_alpha), backup_dir(_backup_dir),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          num_channels(_num_channels), grid_size(_grid_size), num_actions(_num_actions){
        if (!backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        }
        else {
            if(!backup_dir.empty())
                std::filesystem::create_directories(backup_dir);
            initializeNetwork();
        }
        log_file.open(backup_dir + "/agent_log.log");
        log(std::to_string(torch::cuda::is_available()));
        cnn->to(device);
        gru->to(device);
        policy_head->to(device);
        value_head->to(device);
        gate_layer->to(device);
        std::vector<torch::Tensor> params;
        auto cnn_params = cnn->parameters();
        auto gru_params = gru->parameters();
        auto policy_params = policy_head->parameters();
        auto value_params = value_head->parameters();
        auto gate_params = gate_layer->parameters();
        params.insert(params.end(), cnn_params.begin(), cnn_params.end());
        params.insert(params.end(), gru_params.begin(), gru_params.end());
        params.insert(params.end(), policy_params.begin(), policy_params.end());
        params.insert(params.end(), value_params.begin(), value_params.end());
        params.insert(params.end(), gate_params.begin(), gate_params.end());
        optimizer = new torch::optim::AdamW(params, torch::optim::AdamWOptions().lr(learning_rate).weight_decay(1e-4));
        reward_net = new RewardNet(true, (T << 1), learning_rate, alpha, backup_dir + "/../reward_backup", num_actions, num_channels, grid_size);
        log("--------------------------------------------------------");
        auto t = time(nullptr);
        log(std::string(ctime(&t)));
        log("--------------------------------------------------------");
        log("Agent initialized with T=" + std::to_string(T) + ", num_epochs=" + std::to_string(num_epochs) +
            ", gamma=" + std::to_string(gamma) + ", alpha=" + std::to_string(alpha) + ", learning_rate=" + std::to_string(learning_rate) +
            ", ppo_clip=" + std::to_string(ppo_clip));
        action_input = torch::zeros({num_actions}, device);
    }

    ~Agent() {
        saveProgress();
        delete optimizer;
        delete reward_net;
        log_file.close();
    }

    int predict(const std::vector<float>& obs) {
        state = torch::tensor(obs, torch::dtype(torch::kFloat32).device(device)).view({1, num_channels, grid_size, grid_size});
        auto feat = cnn->forward(state);
        auto gru_in = feat.view({1, 1, -1});
        auto [gru_out, _] = gru->forward(gru_in);
        gru_out = gru_out.squeeze(0).squeeze(0);
        auto gated_out = gate_layer->forward(torch::cat({gru_out.view({hidden_size}), action_input}));
        gated_outputs.push_back(gated_out.detach());
        auto logits = policy_head->forward(gated_out);
        auto probs = torch::softmax(logits, -1);
        auto logp = torch::log(probs + 1e-10);
        log_probs.push_back(logp.detach());
        std::vector<float> v(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        int action = dist(gen);
        return action;
    }

    void update(int action, bool imitate) {
        actions.push_back(action);
        rewards.push_back(reward_net->get_reward(action, imitate, state));
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        if (actions.size() == T) {
            if (training)
                train();
            actions.clear();
            rewards.clear();
            log_probs.clear();
            gated_outputs.clear();
        }
    }
};
