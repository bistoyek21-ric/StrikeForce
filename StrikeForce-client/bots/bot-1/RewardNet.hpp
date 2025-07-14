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

class RewardNet {
private:
    bool logging = true;
    float learning_rate;
    int num_actions, num_channels, grid_size;

    std::string backup_dir;

    torch::Device device;

    torch::nn::Sequential cnn{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential combined_processor{nullptr};
    torch::optim::Adam* optimizer{nullptr};

    std::ofstream log_file;

    void log(const std::string& message) {
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

    void initializeNetwork(){
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
        action_processor = torch::nn::Sequential(
            torch::nn::Linear(num_actions, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 128),
            torch::nn::ReLU()
        );
        combined_processor = torch::nn::Sequential(
            torch::nn::Linear(128 + 128, 256),
            torch::nn::ReLU(),
            torch::nn::Linear(256, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 64),
            torch::nn::ReLU(),
            torch::nn::Linear(64, 1),
            torch::nn::Tanh()
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

public:
    RewardNet(float _learning_rate, const std::string &_backup_dir = "bots/bot-1/agent_backup/reward",
          int _num_actions = 1, int _num_channels = 1, int _grid_size = 1)
          : learning_rate(_learning_rate), backup_dir(_backup_dir),
          num_actions(_num_actions), num_channels(_num_channels), grid_size(_grid_size),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){

        log_file.open("bots/bot-1/reward_log.log");

        if (!backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        }
        else
            initializeNetwork();

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
    }

    ~ RewardNet() {
        saveProgress();
        delete optimizer;
        log_file.close();
    }

    float forward(int action, bool imitate, torch::Tensor &state){
        auto features = cnn->forward(state);
        auto action_input = torch::zeros({num_actions}, device);
        action_input[action] = 1;
        auto action_features = action_processor->forward(action_input);
        auto combined = torch::cat({features, action_features.unsqueeze(0)}, /*dim=*/1);
        auto output = combined_processor->forward(combined);
        if (!imitate) {
            auto target = torch::tensor(0.75).to(device);
            auto loss = torch::mse_loss(output, target);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            return 0.75;
        }
        return output.item<float>();
    }
};