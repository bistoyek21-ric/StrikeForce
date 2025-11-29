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

struct ResBImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    int num_layers;

    ResBImpl(int hidden_size, int num_layers) : num_layers(num_layers) {
        for (int i = 0; i < num_layers; ++i)
            layers.push_back(register_module("lin" + std::to_string(i), torch::nn::Linear(hidden_size, hidden_size)));
    }

    torch::Tensor forward(torch::Tensor X) {
        torch::Tensor y, x = X.clone();
        for (int i = 0; i < num_layers; ++i) {
            if (i % 2)
                x = torch::relu(layers[i]->forward(y) + x);
            else
                y = torch::relu(layers[i]->forward(x));
        }
        return x;
    }
};
TORCH_MODULE(ResB);

struct GameCNNImpl : torch::nn::Module {
    std::vector<torch::nn::Conv2d> conv;
    torch::nn::Linear lin{nullptr};
    int channels, d_out, layers, n;

    GameCNNImpl(int channels, int d_out, int layers, int n): channels(channels), d_out(d_out), layers(layers), n(n) {
        for (int i = 0; i < channels; ++i)
            conv.push_back(
                register_module("conv" + std::to_string(i),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(9, 9, 3).stride(1).padding(0).bias(false)))
            );
        int m = (n + 2) / 3 - 2 * layers;
        lin = register_module("lin", torch::nn::Linear(9 * channels * m * m, d_out));
    }

    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> out;
        for (int i = 0; i < channels; ++i) {
            out.push_back(x[i].clone());
            for (int j = 0; j < layers; ++j)
                out[i] = conv[i]->forward(out[i]);
        }
        return lin->forward(torch::stack(out).view({-1}));
    }
};
TORCH_MODULE(GameCNN);

struct RewardModelImpl : torch::nn::Module {
    GameCNN cnn{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential combined_processor{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;
    const float alpha;

    torch::Tensor action_input, h_state;
 
    RewardModelImpl(int num_channels = 32, int grid_x = 39, int grid_y = 39, int hidden_size = 160, int num_actions = 9, float alpha = 0.9f)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y), hidden_size(hidden_size), num_actions(num_actions), alpha(alpha) {
        cnn = register_module("cnn", GameCNN(num_channels, hidden_size, 6, grid_x));
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2)));
        combined_processor = register_module("combined_proc", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size + num_actions, hidden_size), torch::nn::ReLU(),
            ResB(hidden_size, 3), torch::nn::Linear(hidden_size, 1),
            torch::nn::Sigmoid()
        ));
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    void reset_memory() {
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    torch::Tensor forward(int action, torch::Tensor x) {
        auto feat = cnn->forward(x);
        feat = feat * hidden_size / (feat.abs().sum().detach() + 1e-8);
        auto r = gru->forward(feat.view({1, 1, -1}), h_state);
        feat = feat.view({-1});
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = out_seq * hidden_size / (out_seq.abs().sum().detach() + 1e-8);
        h_state = std::get<1>(r);
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        auto combined = torch::cat({out_seq, feat, action_input * hidden_size / (action_input.abs().sum().detach() + 1e-8)});
        return combined_processor->forward(combined);
    }
};
TORCH_MODULE(RewardModel);

class RewardNet {
public:
    RewardNet(bool training = true, int T = 1024, float learning_rate = 1e-3, float alpha = 0.9,
        const std::string &backup_dir = "bots/bot-1/backup/reward_backup")
        : training(training), T(T), learning_rate(learning_rate), alpha(alpha), backup_dir(backup_dir) {
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
        coor[0] = snap_shot();
        int param_count = 0;
        for (auto &p: coor[0]) {
            initial.push_back(p.detach().clone());
            param_count += p.numel();
        }
        //log("RewardNet's parameters: " + std::to_string(param_count));
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
        auto dummy = torch::zeros({num_channels, 1, 9, (grid_x + 2) / 3, (grid_y + 2) / 3});
        model->forward(0, dummy);
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
        log_file.close();
        if (training && !backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            model->reset_memory();
            torch::save(model, backup_dir + "/model.pt");
            torch::save(*optimizer, backup_dir + "/optimizer.pt");
        }
    }

    float get_reward(int action, bool imitate, const torch::Tensor &state) {
        if (is_training) {
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
            }
            else
                return -2;
        }
        auto output = model->forward(action, state);
        auto target = torch::tensor((float)imitate);
        auto reward = compute_reward(action, state, (imitate ? target : output / 2));
        update(output, target);
        return reward.item<float>() * 2 - 1;
    }

    RewardModel get_model() {
        return model;
    }

private:
    bool is_training = false, logging = true, training, done_training;
    std::thread trainThread;
    float learning_rate, alpha;
    int T;
    const int num_actions = 9, num_channels = 32, grid_x = 39, grid_y = 39, hidden_size = 160;
    std::string backup_dir;
    RewardModel model{nullptr};
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};
    std::vector<torch::Tensor> outputs, targets, coor[2], initial;
    std::ofstream log_file;
    std::vector<float> prev;

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

    torch::Tensor compute_reward(int action, const torch::Tensor &state, const torch::Tensor &output) {
        auto reward = output * 0.6 + 0.2;
        std::vector<float> sit(num_channels);
        int x = (grid_x + 2) / 6, y = (grid_y + 2) / 6;
        for (int k = 0; k < num_channels; ++k)
            sit[k] = state[k][0][4][x][y].item<float>();
        if (outputs.empty()) {
            prev = sit;
            return reward;
        }
        if (prev[11] < sit[11] || prev[30] < sit[30] || prev[31] < sit[31])
            reward = torch::tensor(1.0f);// up  | kills, total-damage, total-effect = 1
        else if (prev[18] > sit[18] || prev[24] > sit[24] || prev[25] > sit[25])
            reward = torch::tensor(0.0f);// down| Hp, cnnack-damage, cnnack-effect = 0
        else if (prev[18] < sit[18] || prev[24] < sit[24] || prev[25] < sit[25] || prev[26] < sit[26])
            reward = torch::tensor(0.9f);// up  | Hp, cnnack-damage, cnnack-effect, stamina = 0.9
        else if (prev[26] > sit[26])
            reward -= 0.15;// down | stamina -= 0.15
        else if(!action)
            reward -= 0.1;// nothing | stamina -= 0.1 for doing nothing
        prev = sit;
        return reward;
    }

    void train() {
        time_t ts = time(0);
        auto loss = torch::zeros({}), human = torch::zeros({1}), agent = torch::zeros({1});
        for (int i = 0; i < T; ++i) {
            loss += torch::binary_cross_entropy(outputs[i], targets[i]);
            if (targets[i].item<float>() == 1)
                human += outputs[i];
            else
                agent += outputs[i];
            /////////////////////////////////////////////////////
            //log("output: " + std::to_string(outputs[i].item<float>()) + ", target:" + std::to_string(targets[i].item<float>()));
            /////////////////////////////////////////////////////
        }
        loss /= T;
        optimizer->zero_grad();
        loss.backward();
        /////////////////////////////////////////////////////
        /*
        for (const auto &p: model->named_parameters())
            if (p.value().grad().defined()){
                std::string key = p.key(), k;
                double v = p.value().norm().item<double>();
                double g = p.value().grad().norm().item<double>();
                double coef = g / (v + 1e-8);
                for(int i = 0; i < 30; ++i)
                    if(i >= key.size())
                        k.push_back(' ');
                    else
                        k.push_back(key[i]);
                log(k + " || value-norm: " + std::to_string(v) + " || grad--norm: " + std::to_string(g) + " || grad/value: " + std::to_string(coef));
            }
        */
        /////////////////////////////////////////////////////
        optimizer->step();
        outputs.clear(), targets.clear();
        model->reset_memory();
        log("R: loss=" + std::to_string(loss.item<float>()) + ", time(s)=" + std::to_string(time(0) - ts) + ", step=" + std::to_string(calc_diff())
         + "|| human_avg=" + std::to_string(human.item<float>() / (T / 2)) + ", agent_avg=" + std::to_string(agent.item<float>() / (T / 2)));
        done_training = true;
    }

    void update(const torch::Tensor &output, const torch::Tensor &target) {
        outputs.push_back(output);
        targets.push_back(target);
        if (outputs.size() == T)
            if (training) {
                is_training = true;
                done_training = false;
                //trainThread = std::thread(&RewardNet::train, this);
                ///////////////////////////////////////////////
                std::cout << "RewardNet is training..." << std::endl;
                train();
                is_training = false;
                std::cout << "done!" << std::endl;
                /////////////////////////////////////////////
            }
            else
                outputs.clear(), targets.clear();
    }
};