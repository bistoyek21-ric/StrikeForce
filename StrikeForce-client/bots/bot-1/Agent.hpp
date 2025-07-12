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
//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -lc10 -lsfml-graphics -lsfml-window -lsfml-system
#include "Item.hpp"
#include <torch/torch.h>
#include <random>
#include <filesystem>

class Agent {
private:
    torch::Device device;

    bool training;
    int hidden_size, num_actions, T, num_epochs, num_channels, grid_size;
    double gamma, learning_rate, ppo_clip;
    std::string backup_dir;
    std::vector<int> actions;
    std::vector<double> rewards;
    std::vector<torch::Tensor> log_probs, gru_outputs;

    torch::nn::GRU gru{nullptr};
    torch::nn::Linear policy_head{nullptr}, value_head{nullptr};
    torch::nn::Sequential cnn{nullptr};
    std::vector<int> input_shape;
    torch::optim::AdamW optimizer;

    void saveProgress() {
        if(backup_dir.empty())
            return;
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_size << " " << num_actions << " ";
        dim << input_shape.size() << " ";
        for(int i = 0; i < input_shape.size(); ++i)
            dim << input_shape[i] << " ";
        dim.close();
        torch::save(gru, backup_dir + "/gru.pt");
        torch::save(policy_head, backup_dir + "/policy_head.pt");
        torch::save(value_head, backup_dir + "/value_head.pt");
        torch::save(cnn, backup_dir + "/cnn.pt");
    }

    void initializeNetwork() {
        gru = torch::nn::GRU(hidden_size, hidden_size, /*layers=*/2);
        policy_head = torch::nn::Linear(hidden_size, num_actions);
        value_head = torch::nn::Linear(hidden_size, 1);
        gru->to(device);
        policy_head->to(device);
        value_head->to(device);
    }

    torch::Tensor computeReturns(const std::vector<double>& rewards) {
        torch::Tensor returns = torch::zeros({T}).to(device);
        double running = 0;
        for (int i = T - 1; ~i; --i) {
            running = rewards[i] + gamma * running;
            returns[i] = running;
        }
        return returns;
    }

public:
    Agent(bool _training, int _T, int _num_epochs, double _gamma, double _learning_rate, double _ppo_clip,
          const std::string& _backup_dir = "bots/bot-1/progress", int _num_channels = 1,
          int _grid_size = 1, int _num_actions = 1, std::vector<int> _input_shape = {})
        : training(_training), T(_T), num_epochs(_num_epochs), gamma(_gamma), learning_rate(_learning_rate),
          ppo_clip(_ppo_clip), backup_dir(_backup_dir),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          num_channels(_num_channels), grid_size(_grid_size),
          num_actions(_num_actions), input_shape(_input_shape) {
        torch::set_default_dtype(caffe2::TypeMeta::Make<float>());

        if (!backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            try {
                std::ifstream dim(backup_dir + "/dim.txt");
                dim >> num_channels >> grid_size >> num_actions;
                int sz, num;
                dim >> sz;
                for(int i = 0; i < sz; ++i){
                    dim >> num;
                    input_shape.push_back(num);
                }
                dim.close();
                torch::load(gru, backup_dir + "/gru.pt");
                torch::load(policy_head, backup_dir + "/policy_head.pt");
                torch::load(value_head, backup_dir + "/value_head.pt");
                torch::load(cnn, backup_dir + "/cnn.pt");
            } catch (const std::exception& e) {
                initializeNetwork();
            }
        }
        else
            initializeNetwork();

        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 32, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128), torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1), torch::nn::Flatten()
        );
        cnn->to(device);

        torch::Tensor dummy = torch::randn({1, num_channels, grid_size, grid_size}).to(device);
        hidden_size = cnn->forward(dummy).size(1);

        optimizer = torch::optim::AdamW(
            torch::nn::Module::parameters({cnn, gru, policy_head, value_head}),
            torch::optim::AdamWOptions().lr(learning_rate).weight_decay(1e-4)
        );
    }

    ~Agent() {
        saveProgress();
    }

    int predict(const std::vector<double>& obs) {
        torch::NoGradGuard g;
        auto state = torch::tensor(obs, device).view(input_shape);
        auto feat = cnn->forward(state);
        auto gru_in = feat.view({1, 1, -1});
        auto [gru_out, _] = gru->forward(gru_in);
        gru_outputs.push_back(gru_out.detach());

        auto logits = policy_head->forward(gru_out.squeeze(0));
        auto probs = torch::softmax(logits, -1);
        auto logp = torch::log(probs + 1e-10);
        log_probs.push_back(logp.detach());

        std::vector<float> v(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        return dist(gen);
    }

    void update(int action, bool imitate) {
        actions.push_back(action);
        rewards.push_back(imitate ? 0.75 : 0);
        if (training && actions.size() == T)
            train();
    }

    void train() {
        auto returns = computeReturns(rewards);
        std::vector<torch::Tensor> advantages;

        for (int i = 0; i < T; ++i) {
            auto value = value_head->forward(gru_outputs[i].squeeze(0)).squeeze();
            advantages.push_back(returns[i] - value);
        }

        torch::Tensor adv_tensor = torch::stack(advantages);
        torch::Tensor adv_mean = adv_tensor.mean();
        torch::Tensor adv_std = adv_tensor.std();
        adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-5);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            torch::Tensor p_loss = torch::zeros({}).to(device);
            torch::Tensor v_loss = torch::zeros({}).to(device);

            for (int i = 0; i < T; ++i) {
                auto old_lp = log_probs[i][actions[i]];
                auto current_logits = policy_head->forward(gru_outputs[i].squeeze(0));
                auto current_logp = torch::log_softmax(current_logits, -1)[actions[i]];

                auto ratio = torch::exp(current_logp - old_lp);
                auto clipped = torch::clamp(ratio, 1 - ppo_clip, 1 + ppo_clip);
                auto adv = adv_tensor[i];

                p_loss -= torch::min(ratio * adv, clipped * adv);
                auto current_value = value_head->forward(gru_outputs[i].squeeze(0)).squeeze();
                v_loss += torch::mse_loss(current_value, returns[i]);
            }

            auto loss = p_loss + 0.5 * v_loss;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        actions.clear();
        rewards.clear();
        log_probs.clear();
        gru_outputs.clear();
    }
};