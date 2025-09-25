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

struct ResidualBlockImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    int num_layers;

    ResidualBlockImpl(int hidden_size, int n_layers = 2) : num_layers(n_layers) {
        for (int i = 0; i < n_layers; ++i) {
            int in_dim  = (i % 2 ? hidden_size / 2 : hidden_size);
            int out_dim = (i % 2 ? hidden_size : hidden_size / 2);
            auto lin = torch::nn::Linear(in_dim, out_dim);
            layers.push_back(register_module("lin" + std::to_string(i), lin));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor y;
        for (int i = 0; i < num_layers; ++i) {
            if (i % 2)
                x = torch::tanh(layers[i]->forward(y) + x);
            else
                y = torch::relu(layers[i]->forward(x));
        }
        return x;
    }
};
TORCH_MODULE(ResidualBlock);

struct ResidualConvBlockImpl : torch::nn::Module {
    std::vector<torch::nn::Conv2d> layers;
    int type, num_layers;

    ResidualConvBlockImpl(int filters, int type = 1, int n_layers = 2) 
        : type(type), num_layers(n_layers) {
        for (int i = 0; i < n_layers; ++i) {
            int in_ch  = (i % 2 ? filters / 2 : filters);
            int out_ch = (i % 2 ? filters : filters / 2);
            torch::nn::Conv2d conv{nullptr};
            if (type == 1)
                conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).stride(1).padding(0));
            else
                conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, {1, 3}).stride({1, 2}).padding(0));
            layers.push_back(register_module("conv" + std::to_string(i), conv));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor y;
        for (int i = 0; i < num_layers; ++i) {
            if (i % 2) {
                auto out = layers[i]->forward(y);
                int h = (int)x.size(2), w = (int)x.size(3);
                int h1 = (int)out.size(2), w1 = (int)out.size(3);
                int k = (w - w1) / 2, m = (h - h1) / 2;
                auto cropped = x.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(m, h - m),
                    torch::indexing::Slice(k, w - k)
                });
                x = torch::tanh(out + cropped);
            }
            else
                y = torch::relu(layers[i]->forward(x));
        }
        return x;
    }
};
TORCH_MODULE(ResidualConvBlock);

struct MyCNNImpl : torch::nn::Module {
    std::vector<torch::nn::Conv2d> conv_layers;
    std::vector<ResidualConvBlock> conv_blocks;    

    MyCNNImpl(int in_channels, int hidden_size) {
        conv_layers.push_back(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 80, 3).stride(1).padding(1))));
        conv_blocks.push_back(register_module("convb1", ResidualConvBlock(80, 1, 4)));//27x87=>19x79
        conv_layers.push_back(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(80, hidden_size, 3).stride(2).padding(0))));//19x79=>9x39
        conv_blocks.push_back(register_module("convb2", ResidualConvBlock(hidden_size, 1, 4)));//9x39=>1x31
        conv_blocks.push_back(register_module("convb3", ResidualConvBlock(hidden_size, 2, 4)));//1x31=>1x1
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor cropped, out = conv_layers[0]->forward(x);
        out = conv_blocks[0]->forward(out);
        cropped = out.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(5, 14),
            torch::indexing::Slice(20, 59)
        });
        auto pad = torch::zeros({cropped.size(0), 24, cropped.size(2), cropped.size(3)});
        auto residual = torch::cat({pad, cropped, pad}, 1);
        out = conv_layers[1]->forward(out) + residual;
        out = conv_blocks[1]->forward(out);
        out = conv_blocks[2]->forward(out);
        return out;
    }
};
TORCH_MODULE(MyCNN);

struct RewardModelImpl : torch::nn::Module {
    MyCNN cnn{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential combined_processor{nullptr};
    torch::nn::LayerNorm gru_norm{nullptr}, action_norm{nullptr};

    const int num_channels = 32, hidden_size = 128, num_actions = 10;
    const float alpha = 0.9f;

    torch::Tensor action_input, h_state;

    RewardModelImpl(int n_channels=32, int hidden=128, int n_actions=10, float alpha_=0.9f)
        : num_channels(n_channels), hidden_size(hidden), num_actions(n_actions), alpha(alpha_){
        cnn = register_module("cnn", MyCNN(num_channels, hidden_size));
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2)));
        action_processor = register_module("action_proc", torch::nn::Sequential(
            torch::nn::Linear(num_actions, hidden_size), torch::nn::ReLU(), ResidualBlock(hidden_size, 2)
        ));
        combined_processor = register_module("combined_proc", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size, hidden_size), torch::nn::ReLU(),
            ResidualBlock(hidden_size, 6), torch::nn::Linear(hidden_size, 1),
            torch::nn::Sigmoid()
        ));
        gru_norm = register_module("gru_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
        action_norm = register_module("action_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
        reset_memory();
    }

    void reset_memory() {
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    torch::Tensor forward(int action, const torch::Tensor &state) {
        auto feat = cnn->forward(state);
        feat = feat.view({1, 1, -1});
        auto r = gru->forward(feat, h_state);
        auto out_seq = std::get<0>(r).view({-1}) + feat.view({-1});
        out_seq = gru_norm->forward(out_seq);
        h_state = std::get<1>(r).detach();
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        auto a_feat = action_processor->forward(action_input);
        a_feat = action_norm->forward(a_feat);
        auto combined = torch::cat({out_seq, a_feat});
        return combined_processor->forward(combined);
    }
};
TORCH_MODULE(RewardModel);

class RewardNet {
public:
    RewardNet(bool _training = true, int _T = 256, float _learning_rate = 1e-2, float _alpha = 0.9,
        const std::string &_backup_dir = "bots/bot-1/backup/reward_backup")
        : training(_training), T(_T), learning_rate(_learning_rate), alpha(_alpha), backup_dir(_backup_dir){
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
        for (auto &p: coor[0])
            initial.push_back(p.clone());
        model->to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        if (!training)
            model->eval();
        else
            optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(learning_rate));
        auto dummy = torch::zeros({1, num_channels, grid_x, grid_y});
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
        coor[0].clear();
        for (auto &p: initial)
            coor[0].push_back(p.clone());
        log("-------\nR total dist: " + calc_diff());
        log("======================");
        log_file.close();
        if (training && !backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            model->reset_memory();
            model->to(torch::kCPU);
            torch::save(model, backup_dir + "/model.pt");
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
        auto target = compute_target(action, imitate, state);
        update(output, target);
        float reward = (imitate ? target.item<float>() : output.item<float>()) * 2 - 1;
        return reward;
    }

private:
    bool is_training = false, logging = true, training, done_training;
    std::thread trainThread;
    float learning_rate, alpha;
    int T;
    const int num_actions = 10, num_channels = 32, grid_x = 27, grid_y = 87, hidden_size = 128;
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

    std::string calc_diff(){
        coor[1] = snap_shot();
        double diff = 0;
        for (int i = 0; i < coor[0].size(); ++i)
            diff += (coor[1][i] - coor[0][i]).pow(2).sum().item<float>();
        coor[0].clear();
        for (auto& p: coor[1])
            coor[0].push_back(p.clone());
        coor[1].clear();
        return "step=" + std::to_string(std::sqrt(diff));
    }

    template<typename Type>
    void log(const Type& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    torch::Tensor compute_target(int action, bool imitate, const torch::Tensor &state) {
        auto target = torch::tensor(imitate ? 0.8f : 0.0f);
        std::vector<float> sit(num_channels);
        for (int k = 0; k < num_channels; ++k)
            sit[k] = state[0][k][grid_x / 2][grid_y / 2].item<float>();
        if (outputs.empty()) {
            prev = sit;
            return target;
        }
        if (prev[12] < sit[12] || prev[31] < sit[31] || prev[32] < sit[32]) {
            target = torch::tensor(1.0f);// up  | kills, total-damage, total-effect = 1
            prev = sit;
            return target;
        }
        if (prev[19] > sit[19] || prev[25] > sit[25] || prev[26] > sit[26])
            target = torch::tensor(0.0f);// down| Hp, attack-damage, attack-effect = 0
        else if (prev[19] < sit[19] || prev[25] < sit[25] || prev[26] < sit[26] || prev[27] < sit[27])
            target = torch::tensor(0.9f);// up  | Hp, attack-damage, attack-effect, stamina = 1
        if (prev[27] > sit[27])
            target *= 0.85;// down| stamina = *0.85 or do nothing
        else if(!action)
            target *= 0.9;// nothing| stamina = *0.9 for doing nothing
        prev = sit;
        return target;
    }

    void train() {
        time_t ts = time(0);
        auto loss = torch::zeros({});
        for (int i = 0; i < T; ++i)
            loss += torch::mse_loss(outputs[i], targets[i]);
        loss /= T;
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();
        outputs.clear(), targets.clear();
        model->reset_memory();
        log("R: loss=" + std::to_string(loss.item<float>()) + ", time(s)=" + std::to_string(time(0) - ts) + ", " + calc_diff());
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
};