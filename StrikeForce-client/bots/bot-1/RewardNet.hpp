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
#include "basic.hpp"
#include <torch/torch.h>
#include <random>
#include <filesystem>

class RewardNet {
private:
    bool training, logging = true;
    float learning_rate, alpha;
    int T, num_actions, num_channels, grid_size, action_history;

    std::string backup_dir;

    std::vector<float> probs;

    torch::Tensor action_buffer[2];

    torch::Device device;

    torch::nn::Sequential cnn{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential combined_processor{nullptr};
    torch::optim::Adam* optimizer{nullptr};

    std::vector<torch::Tensor> outputs, targets;

    std::ofstream log_file;

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
        cnn->to(torch::kCPU);
        action_processor->to(torch::kCPU);
        combined_processor->to(torch::kCPU);
        torch::save(cnn, backup_dir + "/cnn.pt");
        torch::save(action_processor, backup_dir + "/action_processor.pt");
        torch::save(combined_processor, backup_dir + "/combined_processor.pt");
        log("Progress saved to " + backup_dir);
    }

    void initializeNetwork() {
        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 32, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128), torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1), torch::nn::Flatten()
        );
        action_processor = torch::nn::Sequential(
            torch::nn::Linear(num_actions, 128), torch::nn::ReLU(),
            torch::nn::Linear(128, 128), torch::nn::ReLU()
        );
        combined_processor = torch::nn::Sequential(
            torch::nn::Linear(256, 128), torch::nn::ReLU(),
            torch::nn::Linear(128, 64), torch::nn::ReLU(),
            torch::nn::Linear(64, 1), torch::nn::Tanh()
        );
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
        torch::load(action_processor, backup_dir + "/action_processor.pt");
        torch::load(combined_processor, backup_dir + "/combined_processor.pt");
        log("Progress loaded successfully");
    }

    void update(torch::Tensor &output, torch::Tensor &target) {
        outputs.push_back(output);
        targets.push_back(target);
        if (outputs.size() < T)
            return;
        auto loss = torch::zeros({}, device);
        for (int i = 0; i < T; ++i)
            loss += torch::mse_loss(outputs[i], targets[i]);
        loss /= T;
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();
        outputs.clear(), targets.clear();
    }

    void punish_random(torch::Tensor &state) {
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        auto target = torch::tensor(-1.0f).to(device);
        auto output = forward(dist(gen), action_buffer[1], state);
        update(output, target);
    }

    torch::Tensor forward(int action, torch::Tensor &action_input, torch::Tensor &state) {
        auto features = cnn->forward(state);
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        auto action_features = action_processor->forward(action_input);
        auto combined = torch::cat({features.view({128}), action_features});
        return combined_processor->forward(combined);
    }

public:
    RewardNet(bool _training, int _T, float _learning_rate, float _alpha, const std::string &_backup_dir = "bots/bot-1/backup/reward_backup",
          int _num_actions = 1, int _num_channels = 1, int _grid_size = 1, int _action_history = 1)
          : training(_training), T(_T), learning_rate(_learning_rate), alpha(_alpha), backup_dir(_backup_dir),
          num_actions(_num_actions), num_channels(_num_channels), grid_size(_grid_size),
          action_history(_action_history),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
        if (!backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        }
        else {
            if (!backup_dir.empty())
                std::filesystem::create_directories(backup_dir);
            initializeNetwork();
        }
        log_file.open(backup_dir + "/reward_log.log");
        cnn->to(device);
        action_processor->to(device);
        combined_processor->to(device);
        std::vector<torch::Tensor> params;
        auto cnn_params = cnn->parameters();
        auto action_params = action_processor->parameters();
        auto combined_params = combined_processor->parameters();
        params.insert(params.end(), cnn_params.begin(), cnn_params.end());
        params.insert(params.end(), action_params.begin(), action_params.end());
        params.insert(params.end(), combined_params.begin(), combined_params.end());
        optimizer = new torch::optim::Adam(params, torch::optim::AdamOptions().lr(learning_rate));
        probs.assign(num_actions, 1.0 / num_actions);
        for (int i = 0; i < 2; ++i)
            action_buffer[i] = torch::zeros({num_actions}, device);
    }

    ~ RewardNet() {
        saveProgress();
        delete optimizer;
        log_file.close();
    }

    float get_reward(int action, bool imitate, torch::Tensor &state) {
        if (training)
            punish_random(state);
        auto output = forward(action, action_buffer[0], state);
        if (imitate) {
            auto target = torch::tensor(1.0f).to(device);
            if (training)
                update(output, target);
            output = target;
        }
        return output.item<float>();
    }
};