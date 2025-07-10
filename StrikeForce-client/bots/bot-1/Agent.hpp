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
#include "Item.hpp"
#include <torch/torch.h>
#include <random>
#include <filesystem>
#include <deque>
//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -lc10 -lsfml-graphics -lsfml-window -lsfml-system

class Agent {
private:
    torch::Device device;

    bool training;
    int hidden_size, num_actions, T, num_channels, grid_size;
    double gamma, learning_rate, ppo_clip;
    std::string backup_dir;
    std::vector<int> action;
    std::vector<double> rewards;
    std::deque<std::tuple<torch::Tensor, int, double, torch::Tensor>> experience_buffer; // state, action, reward, next_state

    torch::nn::GRU gru{nullptr};
    torch::nn::Linear policy_head{nullptr}, value_head{nullptr};
    torch::nn::Sequential cnn{nullptr};
    std::vector<int> input_shape;
    torch::optim::AdamW optimizer;

    void saveProgress() {
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_size << " " << hidden_size << " " << num_actions;
        dim.close();
        torch::save(gru, backup_dir + "/gru.pt");
        torch::save(policy_head, backup_dir + "/policy_head.pt");
        torch::save(value_head, backup_dir + "/value_head.pt");
        torch::save(cnn, backup_dir + "/cnn.pt");
    }

    void initializeNetwork() {
        gru = torch::nn::GRU(hidden_size, hidden_size, 2); // 2 layers for better sequence modeling
        policy_head = torch::nn::Linear(hidden_size, num_actions);
        value_head = torch::nn::Linear(hidden_size, 1);
        gru->to(device);
        policy_head->to(device);
        value_head->to(device);
    }

    torch::Tensor computeReturns(const std::vector<double>& rewards) {
        torch::Tensor returns = torch::zeros({(int)rewards.size()}, torch::kFloat32).to(device);
        double running_return = 0;
        for (int t = rewards.size() - 1; t >= 0; --t) {
            running_return = rewards[t] + gamma * running_return;
            returns[t] = running_return;
        }
        return returns;
    }

public:
    Agent(bool training, int T, double gamma, double learning_rate, double ppo_clip,
          const std::string& backup_dir = "bots/bot-1/progress", int num_channels = 1, int grid_size = 1, int hidden_size = 1,
          int num_actions = 1, std::vector<int> shape = {})
            : training(training), T(T), gamma(gamma), learning_rate(learning_rate),
              ppo_clip(ppo_clip), backup_dir(backup_dir),
              device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
              num_channels(num_channels), grid_size(grid_size), hidden_size(hidden_size),
              num_actions(num_actions), input_shape(shape){
        torch::set_default_dtype(caffe2::TypeMeta::Make<float>());

        // Deep CNN architecture
        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 32, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1),
            torch::nn::Flatten()
        );
        cnn->to(device);

        // Calculate observation_size dynamically
        torch::Tensor dummy_input = torch::randn({1, num_channels, grid_size, grid_size}).to(device);
        torch::Tensor dummy_output = cnn->forward(dummy_input);
        this->hidden_size = dummy_output.size(1); // Adjust hidden_size dynamically

        initializeNetwork();

        optimizer = torch::optim::AdamW(
            torch::nn::Module::parameters({cnn, gru, policy_head, value_head}),
            torch::optim::AdamWOptions().lr(learning_rate).weight_decay(1e-4)
        );

        if (std::filesystem::exists(backup_dir)) {
            try {
                std::ifstream dim(backup_dir + "/dim.txt");
                int saved_num_channels, saved_grid_size, saved_hidden_size, saved_num_actions;
                dim >> saved_num_channels >> saved_grid_size >> saved_hidden_size >> saved_num_actions;
                if (saved_num_channels != num_channels || saved_grid_size != grid_size ||
                    saved_hidden_size != hidden_size || saved_num_actions != num_actions)
                    throw std::runtime_error("Dimensions do not match saved model.");
                dim.close();
                torch::load(gru, backup_dir + "/gru.pt");
                torch::load(policy_head, backup_dir + "/policy_head.pt");
                torch::load(value_head, backup_dir + "/value_head.pt");
                torch::load(cnn, backup_dir + "/cnn.pt");
            } catch (const std::exception& e) {
                initializeNetwork();
            }
        }
    }

    ~Agent() {
        saveProgress();
    }

    int predict(const std::vector<double>& obs) {
        torch::NoGradGuard no_grad;
        torch::Tensor input = torch::tensor(obs, device).view(input_shape);
        torch::Tensor grid_features = cnn->forward(input);
        torch::Tensor gru_input = grid_features.view({1, 1, -1});
        auto [gru_output, _] = gru->forward(gru_input);
        torch::Tensor logits = policy_head->forward(gru_output.squeeze(0));
        torch::Tensor probs = torch::softmax(logits, -1);
        std::vector<float> probs_vec(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs_vec.begin(), probs_vec.end());
        return dist(gen);
    }

    void storeExperience(const torch::Tensor& state, int action, double reward, const torch::Tensor& next_state) {
        if (experience_buffer.size() == T)
            experience_buffer.pop_front();
        experience_buffer.push_back({state.clone(), action, reward, next_state.clone()});
    }

    void update(int a_t, double reward, const std::vector<double>& next_obs) {
        torch::Tensor state = torch::tensor(next_obs, device).view(input_shape);
        action.push_back(a_t);
        rewards.push_back(reward);
        storeExperience(state, a_t, reward, torch::tensor(next_obs, device).view(input_shape));
        if (training && action.size() == T)
            train();
    }

    void train() {
        std::vector<torch::Tensor> states, next_states;
        std::vector<int> actions;
        std::vector<double> rewards_vec;
        for (int i = 0; i < T; ++i) {
            auto [state, action, reward, next_state] = experience_buffer[i];
            states.push_back(state);
            actions.push_back(action);
            rewards_vec.push_back(reward);
            next_states.push_back(next_state);
        }

        torch::Tensor state_batch = torch::stack(states);
        torch::Tensor next_state_batch = torch::stack(next_states);
        torch::Tensor action_batch = torch::tensor(actions).to(device);
        torch::Tensor returns = computeReturns(rewards_vec);

        // Forward pass
        torch::Tensor features = cnn->forward(state_batch);
        auto [gru_out, _] = gru->forward(features.view({-1, 1, hidden_size}));
        torch::Tensor logits = policy_head->forward(gru_out.squeeze(1));
        torch::Tensor values = value_head->forward(gru_out.squeeze(1)).squeeze(-1);
        torch::Tensor probs = torch::softmax(logits, -1);
        torch::Tensor old_probs = probs.detach();

        // PPO Loss
        torch::Tensor advantages = returns - values.detach();
        torch::Tensor ratio = probs.gather(1, action_batch.unsqueeze(1)) / (old_probs.gather(1, action_batch.unsqueeze(1)) + 1e-5);
        torch::Tensor policy_loss = -torch::min(ratio, torch::clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)) * advantages;
        torch::Tensor value_loss = torch::mse_loss(values, returns);
        torch::Tensor loss = policy_loss.mean() + 0.5 * value_loss;

        // Optimization step
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        action.clear();
        rewards.clear();
    }
};