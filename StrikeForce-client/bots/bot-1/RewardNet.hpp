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
#include "../../basic.hpp"
#include <torch/torch.h>

#define LAYER_INDEX 3

struct ResBImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    int num_layers;

    ResBImpl(int hidden_size, int num_layers) : num_layers(num_layers) {
        for (int i = 0; i < num_layers; ++i)
            layers.push_back(
                register_module("lin" + std::to_string(i), torch::nn::Linear(hidden_size, hidden_size))
            );
    }

    torch::Tensor forward(torch::Tensor X) {
        auto y = X.clone();
        auto x = y * y.numel() / (y.abs().sum().detach() + 1e-8);
        for (int i = 0; i < num_layers; ++i) {
            y = torch::relu(layers[i]->forward(x)) + x;
            x = y * y.numel() / (y.abs().sum().detach() + 1e-8);
        }
        return x;
    }

};
TORCH_MODULE(ResB);

struct GameCNNImpl : torch::nn::Module {
    std::vector<torch::nn::Conv2d> conv;
    int channels, d_out, layers, n;

    GameCNNImpl(int channels, int d_out, int layers, int n): channels(channels), d_out(d_out), layers(layers), n(n) {
        for (int i = 0; i < layers; ++i)
            conv.push_back(
                register_module("conv" + std::to_string(i),
                torch::nn::Conv2d(torch::nn::Conv2dOptions((i ? d_out : channels), d_out, 3).stride(2).padding(0).bias(false)))
            );
    }

    torch::Tensor forward(torch::Tensor x) {
        auto y = x.clone();
        for (int i = 0; i < layers; ++i)
            y = conv[i]->forward(y);
        return y;
    }
};
TORCH_MODULE(GameCNN);

struct BackboneImpl : torch::nn::Module {
    GameCNN cnn{nullptr};
    torch::nn::GRU gru0{nullptr}, gru1{nullptr};
    torch::nn::Sequential combined_processor{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;
    torch::Tensor action_input, h_state[2];

    BackboneImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31, int hidden_size = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y),
          hidden_size(hidden_size), num_actions(num_actions) {
        cnn = register_module("cnn", GameCNN(num_channels, hidden_size, 4, grid_x));
        gru0 = register_module("gru0", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(1)));
        combined_processor = register_module("combined_processor", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size + num_actions, hidden_size)
        ));
        gru1 = register_module("gru1", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(1)));
        reset_memory();
    }

    void reset_memory() {
        action_input = torch::zeros({num_actions});
        action_input[0] += 1;
        h_state[0] = torch::zeros({1, 1, hidden_size});
        h_state[1] = torch::zeros({1, 1, hidden_size});
    }

    void update_actions(torch::Tensor one_hot) {
        action_input = one_hot.clone();
    }

    torch::Tensor forward(const torch::Tensor &x) {
        auto feat = cnn->forward(x);
        feat = feat * hidden_size / (feat.abs().sum().detach() + 1e-8);

        auto r0 = gru0->forward(feat.view({1, 1, -1}), h_state[0]);
        auto out_seq = std::get<0>(r0).view({-1});
        out_seq = out_seq * hidden_size / (out_seq.abs().sum().detach() + 1e-8);
        h_state[0] = std::get<1>(r0);

        std::vector<torch::Tensor> y;
        std::vector<std::vector<int>> d = {{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}};
        for (auto e: d)
            for (int j = 0; j < num_channels; ++j)
                y.push_back(
                    x[0][j][grid_x / 2 + e[0]][grid_y / 2 + e[1]].clone()
                );
        auto pov = torch::cat({torch::stack(y).view({-1}), action_input});
        auto combined = torch::cat({out_seq + feat.view({-1}), pov * hidden_size / (pov.abs().sum().detach() + 1e-8)});

        auto gated = combined_processor->forward(combined);
        gated = gated * hidden_size / (gated.abs().sum().detach() + 1e-8);

        auto r1 = gru1->forward(gated.view({1, 1, -1}), h_state[1]);
        auto out = std::get<0>(r1).view({-1});
        out = out * hidden_size / (out.abs().sum().detach() + 1e-8) + gated;
        h_state[1] = std::get<1>(r1);

        return out;
    }
};
TORCH_MODULE(Backbone);

struct RewardModelImpl : torch::nn::Module {
    Backbone backbone{nullptr};
    torch::nn::Sequential value_head{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;

    RewardModelImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31, int hidden_size = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y), hidden_size(hidden_size), num_actions(num_actions) {
        backbone = register_module("backbone", Backbone(num_channels, grid_x, grid_y, hidden_size, num_actions));
        value_head = register_module("value", torch::nn::Sequential(
            ResB(hidden_size, LAYER_INDEX), torch::nn::Linear(hidden_size, 1)
        ));
        backbone->reset_memory();
    }

    void reset_memory() {
        backbone->reset_memory();
    }

    void update_actions(torch::Tensor one_hot) {
        backbone->update_actions(one_hot);
    }

    torch::Tensor forward(torch::Tensor action, torch::Tensor x) {
        update_actions(action);
        auto gated = backbone->forward(x);
        auto value = value_head->forward(gated).view({-1});
        return torch::sigmoid(value);
    }
};
TORCH_MODULE(RewardModel);

class RewardNet {
public:
    RewardNet(bool training = true, int T = 1024, float learning_rate = 1e-3, 
        const std::string &backup_dir = "bots/bot-1/backup/reward_backup")
        : training(training), T(T), learning_rate(learning_rate), backup_dir(backup_dir) {
        model = RewardModel();
        if (!backup_dir.empty()) {
            if (std::filesystem::exists(backup_dir)) {
                log_file.open(backup_dir + "/reward_log.log", std::ios::app);
                try{
                    torch::load(model, backup_dir + "/model.pt");
                } catch(...){}
            } else {
                std::filesystem::create_directories(backup_dir);
                log_file.open(backup_dir + "/reward_log.log", std::ios::app);
            }
        }
#if defined(FREEZE_REWARDNET_BLOCK)
        this->training = training = false;
        log("Freezing Reward Network parameters.");
#endif
        coor[0] = snap_shot();
        int param_count = 0;
        for (auto &p: coor[0]) {
            initial.push_back(p.detach().clone());
            param_count += p.numel();
        }
        log("RewardNet's parameters: " + std::to_string(param_count));
        log("LAYER_INDEX=" + std::to_string(LAYER_INDEX));
#if defined(SLOWMOTION)
        log("SLOWMOTION");
#endif
        if (!training)
            model->eval();
        else {
            model->train();
            optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(learning_rate));
            if (!backup_dir.empty() && std::filesystem::exists(backup_dir + "/optimizer.pt")) {
                try {
                    torch::load(*optimizer, backup_dir + "/optimizer.pt");
                } catch (...) {}
            }
        }
        auto dummy = torch::zeros({1, num_channels, grid_x, grid_y});
        model->forward(model->backbone->action_input, dummy);
        model->reset_memory();
    }
    
    ~RewardNet() {
        if (is_training)
            if (trainThread.joinable()) {
                std::cout << "Reward Network is updating...\nthis might take a few seconds" << std::endl;
                trainThread.join();
                std::cout << "done!" << std::endl;
            }
        if (training) {
            coor[0].clear();
            for (auto &p: initial)
                coor[0].push_back(p.detach().clone());
            log("-------\nR total dist: step=" + std::to_string(calc_diff()));
            log("======================");
        }
        else {
            log("-------\nR total dist: step=0.000000");
            log("======================");
        }
        log_file.close();
        if (training && !backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            model->reset_memory();
            torch::save(model, backup_dir + "/model.pt");
            torch::save(*optimizer, backup_dir + "/optimizer.pt");
        }
    }

    torch::Tensor get_reward(torch::Tensor action, bool imitate, const torch::Tensor &state) {
        if (is_training) {
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
            }
            else
                return torch::tensor(-2.0f);
        }
        auto output = model->forward(action, state);
        auto target = torch::tensor(imitate ? 1.0f : 0.0f);
        update(output, target);
        auto reward = torch::log(output);
        return reward;
    }

    RewardModel get_model() {
        return model;
    }

    void train_epoch(const std::vector<int> &actions,
         const bool &manual, const std::vector<torch::Tensor> &states){
        time_t ts = time(0);
        outputs.clear(), targets.clear();
        model->reset_memory();
        for (int i = 0; i < T; ++i) {
            torch::Tensor one_hot = torch::zeros({num_actions});
            one_hot[actions[i]] += 1;
            get_reward(one_hot, (i < T / 2 ? !manual : manual), states[i]);
        }
        is_training = true;
        done_training = false;
        //trainThread = std::thread(&Agent::train, this);
        ///////////////////////////////////////////////
        if (training)
            std::cout << "RewardNet is training..." << std::endl;
        train();
        is_training = false;
        /////////////////////////////////////////////
        log("total time(s) = " + std::to_string(time(0) - ts));
        return;
    }

private:
    bool is_training = false, logging = true, training, done_training;
    std::thread trainThread;
    float learning_rate, alpha;
    int T;
    const int num_actions = 9, num_channels = 32, grid_x = 31, grid_y = 31, hidden_size = 160;
    std::string backup_dir;
    RewardModel model{nullptr};
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};
    std::vector<torch::Tensor> outputs, targets, coor[2], initial;
    std::ofstream log_file;

    std::vector<torch::Tensor> snap_shot(){
        std::vector<torch::Tensor> params;
        for (auto& p : model->parameters())
            params.push_back(p.detach().clone());
        return params;
    }

    double calc_diff(){
        coor[1] = snap_shot();
        double diff = 0;
        for (int i = 0; i < coor[0].size(); ++i)
            diff += (coor[1][i] - coor[0][i]).pow(2).sum().item<float>();
        coor[0].clear();
        for (auto& p: coor[1])
            coor[0].push_back(p.detach().clone());
        coor[1].clear();
        return std::sqrt(diff);
    }

    template<typename Type>
    void log(const Type& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    void train() {
        time_t ts = time(0);
        auto loss = torch::zeros({1}), human = torch::zeros({1}), agent = torch::zeros({1});
        int hum_cnt = 0, agent_cnt = 0;
        for (int i = 0; i < T; ++i) {
            loss += torch::binary_cross_entropy(outputs[i], targets[i]);
            if (targets[i].item<float>() == 1)
                human += outputs[i], ++hum_cnt;
            else
                agent += outputs[i], ++agent_cnt;
        }
        loss = loss / T, human /= hum_cnt, agent /= agent_cnt;
        if (training) {
            optimizer->zero_grad();
            loss.backward(at::Tensor(), true);
            optimizer->step();
        }
        outputs.clear(), targets.clear();
        model->reset_memory();
        log("R: loss=" + std::to_string(loss.item<float>()) +
         "|human_avg=" + std::to_string(human.item<float>()) +
         "|agent_avg=" + std::to_string(agent.item<float>()) +
         ",time(s)=" + std::to_string(time(0) - ts) +
         ",step=" + std::to_string(calc_diff()));
        done_training = true;
    }

    void update(const torch::Tensor &output, const torch::Tensor &target) {
        outputs.push_back(output);
        targets.push_back(target);
    }
};