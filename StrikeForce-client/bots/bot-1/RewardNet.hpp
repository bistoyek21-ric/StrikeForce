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

class ResidualBlock : public torch::nn::Module {
public:
    ResidualBlock(int hidden_size) {
        linear1 = torch::nn::Linear(hidden_size, hidden_size);
        linear2 = torch::nn::Linear(hidden_size, hidden_size);
        layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(linear1->forward(x));
        x = linear2->forward(x);
        x = x + residual;
        x = layer_norm->forward(x);
        return torch::relu(x);
    }

private:
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
};

class RewardNet {
private:
    std::mutex mtx;
    std::condition_variable cv;
    bool is_training = false, done_training;
    std::thread trainThread;

    bool training, logging = false;
    float learning_rate, alpha;
    int T, num_actions, num_channels, grid_x, grid_y, hidden_size;

    std::vector<float> prev;

    std::string backup_dir;

    torch::Tensor action_input, h_state;

    torch::Device device;

    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential cnn{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential combined_processor{nullptr};
    torch::nn::LayerNorm gru_norm{nullptr};
    torch::nn::LayerNorm action_norm{nullptr};

    torch::optim::Adam* optimizer{nullptr};

    std::vector<torch::Tensor> outputs, targets;

    std::ofstream log_file;

    template<typename Type>
    void log(const Type& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    void saveProgress() {
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_x << " " << grid_y << " " << num_actions << " ";
        dim.close();
        cnn->to(torch::kCPU);
        gru->to(torch::kCPU);
        action_processor->to(torch::kCPU);
        combined_processor->to(torch::kCPU);
        gru_norm->to(torch::kCPU);
        action_norm->to(torch::kCPU);
        torch::save(cnn, backup_dir + "/cnn.pt");
        torch::save(gru, backup_dir + "/gru.pt");
        torch::save(action_processor, backup_dir + "/action_processor.pt");
        torch::save(combined_processor, backup_dir + "/combined_processor.pt");
        torch::save(gru_norm, backup_dir + "/gru_norm.pt");
        torch::save(action_norm, backup_dir + "/action_norm.pt");
        log("Progress saved to " + backup_dir);
    }

    void initializeNetwork() {
        hidden_size = 128;
        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_size, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_size, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size), torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1), torch::nn::Flatten()
        );
        gru = torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2));
        action_processor = torch::nn::Sequential(
            torch::nn::Linear(num_actions, hidden_size), torch::nn::ReLU(),
            ResidualBlock(hidden_size)
        );
        combined_processor = torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size, hidden_size), torch::nn::ReLU(),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            torch::nn::Linear(hidden_size, 1), torch::nn::Sigmoid()
        );
        gru_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
        action_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    }

    void load_progress() {
        std::ifstream dim(backup_dir + "/dim.txt");
        if (!dim.is_open()) {
            log("Failed to open dim.txt");
            throw std::runtime_error("Failed to open dim.txt");
        }
        dim >> num_channels >> grid_x >> grid_y >> num_actions;
        log("Loaded dimensions: num_channels=" + std::to_string(num_channels) +
            ", grid_x=" + std::to_string(grid_x) +
            ", grid_y=" + std::to_string(grid_y) +
            ", num_actions=" + std::to_string(num_actions));
        dim.close();
        initializeNetwork();
        torch::load(cnn, backup_dir + "/cnn.pt");
        torch::load(gru, backup_dir + "/gru.pt");
        torch::load(action_processor, backup_dir + "/action_processor.pt");
        torch::load(combined_processor, backup_dir + "/combined_processor.pt");
        torch::load(gru_norm, backup_dir + "/gru_norm.pt");
        torch::load(action_norm, backup_dir + "/action_norm.pt");
        log("Progress loaded successfully");
    }

    torch::Tensor compute_target(bool imitate, const torch::Tensor &state) {
        auto target = torch::tensor(imitate ? 0.8f : 0.0f).to(device);
        if (outputs.empty()) {
            prev = sit;
            return target;
        }
        std::vector<float> sit(num_channels);
        for (int i = 0; i < grid_x; ++i)
            for (int j = 0; j < grid_y; ++j)
                if (state[0][0][i][j].item<float>() == 1)
                    for (int k = 0; k < num_channels; ++k)
                        sit[k] = state[0][k][i][j].item<float>();
        // up  | kills, total-damage, total-effect = 1
        if (prev[12] < sit[12] || prev[31] < sit[31] || prev[32] < sit[32]) {
            target = torch::tensor(1.0f).to(device);
            prev = sit;
            return target;
        }
        // down| Hp, attack-damage, attack-effect = 0
        if (prev[19] > sit[19] || prev[25] > sit[25] || prev[26] > sit[26])
            target = torch::tensor(0.0f).to(device);
        // up  | Hp, attack-damage, attack-effect, stamina = 1
        else if (prev[19] < sit[19] || prev[25] < sit[25] || prev[26] < sit[26] || prev[27] < sit[27])
            target = torch::tensor(0.9f).to(device);
        // down| stamina = *0.9
        if (prev[27] > sit[27])
            target *= 0.9;
        prev = sit;
        return target;
    }

    void train() {
        auto loss = torch::zeros({}, device);
        for (int i = 0; i < T; ++i)
            loss += torch::mse_loss(outputs[i], targets[i]);
        loss /= T;
        log("loss=" + std::to_string(loss.item<float>()));
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();
        outputs.clear(), targets.clear();
        log("*");
        done_training = true;
    }

    void update(const torch::Tensor &output, const torch::Tensor &target) {
        outputs.push_back(output);
        targets.push_back(target);
        if (outputs.size() == T)
            if (training) {
                is_training = true;
                done_training = false;
                trainThread = std::thread(&RewardNet::train, this);
            }
            else
                outputs.clear(), targets.clear();
    }

    torch::Tensor forward(int action, const torch::Tensor &state) {
        auto feat = cnn->forward(state);
        feat = feat.view({1, 1, -1});
        auto r = gru->forward(feat, h_state);
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = gru_norm->forward(out_seq);
        h_state = std::get<1>(r).detach();
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        auto a_feat = action_processor->forward(action_input);
        a_feat = action_norm->forward(a_feat);
        auto combined = torch::cat({out_seq, a_feat});
        return combined_processor->forward(combined);
    }

public:
    RewardNet(bool _training, int _T, float _learning_rate, float _alpha, const std::string &_backup_dir = "bots/bot-1/backup/reward_backup",
              int _num_actions = 12, int _num_channels = 33, int _grid_x = 15, int _grid_y = 49)
        : training(_training), T(_T), learning_rate(_learning_rate), alpha(_alpha), backup_dir(_backup_dir),
          num_actions(_num_actions), num_channels(_num_channels), grid_x(_grid_x), grid_y(_grid_y),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        if (std::filesystem::exists(backup_dir)) {
            log_file.open(backup_dir + "/reward_log.log");
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        } else {
            std::filesystem::create_directories(backup_dir);
            log_file.open(backup_dir + "/reward_log.log");
            initializeNetwork();
        }
        cnn->to(device);
        gru->to(device);
        action_processor->to(device);
        combined_processor->to(device);
        gru_norm->to(device);
        action_norm->to(device);
        std::vector<torch::Tensor> params;
        auto cnn_params = cnn->parameters();
        auto gru_params = gru->parameters();
        auto action_params = action_processor->parameters();
        auto combined_params = combined_processor->parameters();
        auto gru_norm_params = gru_norm->parameters();
        auto action_norm_params = action_norm->parameters();
        params.insert(params.end(), cnn_params.begin(), cnn_params.end());
        params.insert(params.end(), gru_params.begin(), gru_params.end());
        params.insert(params.end(), action_params.begin(), action_params.end());
        params.insert(params.end(), combined_params.begin(), combined_params.end());
        params.insert(params.end(), gru_norm_params.begin(), gru_norm_params.end());
        params.insert(params.end(), action_norm_params.begin(), action_norm_params.end());
        optimizer = new torch::optim::Adam(params, torch::optim::AdamOptions().lr(learning_rate));
        action_input = torch::zeros({num_actions}, device);
        h_state = torch::zeros({2, 1, hidden_size}, device);
        prev.assign(num_channels, 0);
    }

    ~RewardNet() {
        if (is_training)
            if (trainThread.joinable()) {
                std::cout << "Reward Network is updating...\nthis might take a few seconds" << std::endl;
                trainThread.join();
                std::cout << "done!" << std::endl;
            }
        if (training)
            saveProgress();
        delete optimizer;
        log_file.close();
    }

    float get_reward(int action, bool imitate, const torch::Tensor &state) {
        if (is_training) {
            if (done_training) {
                is_training = false;
                cv.notify_all();
                if (trainThread.joinable())
                    trainThread.join();
                action_input = torch::zeros({num_actions}, device);
                h_state = torch::zeros({2, 1, hidden_size}, device);
            }
            else
                return -2;
        }
        auto output = forward(action, state);
        auto target = compute_target(imitate, state);
        update(output, target);
        float reward = (imitate ? target.item<float>() : output.item<float>()) * 2 - 1;
        return reward;
    }
};