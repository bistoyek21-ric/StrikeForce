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

struct GamingCNNImpl : torch::nn::Module {
    torch::nn::Conv2d conv[2] = {{nullptr}, {nullptr}};
    int cIn, n;

    GamingCNNImpl(int cIn, int n) : cIn(n), n(n){
        for (int i = 0; i < 2; ++i)
            conv[i] = register_module("conv" + std::to_string(i), torch::nn::Conv2d(
                torch::nn::Conv2dOptions(cIn, n, /*kernel_size=*/1).stride(1).padding(0).bias(true)
            ));
    }
    // forward: A shape (B, cIn, H, W) -> output (B, 5n, H-2, W-2)
    torch::Tensor forward(torch::Tensor x) {
        //TORCH_CHECK(x.dim() == 4, "input must be 4D (B,C,H,W)");
        int64_t B = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
        //TORCH_CHECK(C_in == cIn, "input channels mismatch");
        //TORCH_CHECK(H >= 3 && W >= 3, "height and width must be >= 3 to produce neighbors");
        // y[0] and y[1]: shapes (B, n, H(-2 for y[0]), W(-2 for y[0]))
        torch::Tensor y[2];
        // crop indices: we want final spatial size (H-2, W-2).
        // mapping: target (t_i, t_j) corresponds to original (i = t_i+1, j = t_j+1)
        // so we slice accordingly.
        // center from y[0]: conv[0]->forward(x[:, :, 1:-1, 1:-1])
        y[0] = conv[0]->forward(x.index({torch::indexing::Slice(), torch::indexing::Slice(),
            torch::indexing::Slice(1, H-1), torch::indexing::Slice(1, W-1)}));
        auto center = y[0]; // (B, n, H-2, W-2)
        y[1] = conv[1]->forward(x);
        // y[1] shifts:
        auto up = y[1].index({torch::indexing::Slice(), torch::indexing::Slice(),
            torch::indexing::Slice(0, H-2), torch::indexing::Slice(1, W-1)}); // i-1
        auto down = y[1].index({torch::indexing::Slice(), torch::indexing::Slice(),
            torch::indexing::Slice(2, H),   torch::indexing::Slice(1, W-1)}); // i+1
        auto left = y[1].index({torch::indexing::Slice(), torch::indexing::Slice(),
            torch::indexing::Slice(1, H-1), torch::indexing::Slice(0, W-2)}); // j-1
        auto right = y[1].index({torch::indexing::Slice(), torch::indexing::Slice(),
            torch::indexing::Slice(1, H-1), torch::indexing::Slice(2, W)});   // j+1
        auto out = torch::cat({center, up, left, down, right}, /*dim=*/1); // (B, 5n, H-2, W-2)
        return out;
    }
};
TORCH_MODULE(GamingCNN);

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

struct ResGameCNNImpl : torch::nn::Module {
    std::vector<GamingCNN> layers;
    int num_layers, cIn, filters;

    ResGameCNNImpl(int cIn, int filters, int num_layers): cIn(cIn), filters(filters), num_layers(num_layers) {
        //TORCH_CHECK(filters % 5 == 0, "filters must be devidsible by 5");
        layers.push_back(register_module("gamecnn0", GamingCNN(cIn, filters / 5)));
        for (int i = 1; i < num_layers; ++i)
            layers.push_back(register_module("gamecnn" + std::to_string(i), GamingCNN(filters, filters / 5)));
    }

    torch::Tensor forward(torch::Tensor X) {
        torch::Tensor y, x = X.clone();
        for (int i = 0; i < num_layers; ++i) {
            if (i % 2) {
                auto out = layers[i]->forward(y);
                if (i == 1 && cIn != filters) {
                    x = out;
                    continue;
                }
                int h = x.size(2), w = x.size(3);
                int h1 = out.size(2), w1 = out.size(3);
                int k = (w - w1) / 2, m = (h - h1) / 2;
                auto cropped = x.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(m, h - m),
                    torch::indexing::Slice(k, w - k)
                });
                x = torch::relu(out + cropped);
            }
            else
                y = torch::relu(layers[i]->forward(x));
        }
        return x;
    }
};
TORCH_MODULE(ResGameCNN);

struct RewardModelImpl : torch::nn::Module {
    ResGameCNN cnn{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential combined_processor{nullptr};

    int num_channels, hidden_size, num_actions;
    const float alpha;

    torch::Tensor action_input, h_state;
 
    RewardModelImpl(int num_channels = 32, int hidden_size = 160, int num_actions = 9, float alpha = 0.9f)
        : num_channels(num_channels), hidden_size(hidden_size), num_actions(num_actions), alpha(alpha) {
        cnn = register_module("cnn", ResGameCNN(num_channels, hidden_size, 40));
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2)));
        combined_processor = register_module("combined_proc", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size + num_actions, hidden_size), torch::nn::ReLU(),
            ResB(hidden_size, 3), torch::nn::Linear(hidden_size, 1),
            torch::nn::Sigmoid()
        ));
        action_input = register_buffer("a_input", torch::zeros({num_actions}));
        h_state = register_buffer("h_state", torch::zeros({2, 1, hidden_size}));
    }

    void reset_memory() {
        action_input = action_input.detach();
        h_state = h_state.detach();
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    torch::Tensor forward(int action, const torch::Tensor &state) {
        auto feat = cnn->forward(state);
        feat = feat * std::sqrt(feat.numel()) / (feat.norm() + 1e-8);
        auto r = gru->forward(feat.view({1, 1, -1}), h_state);
        feat = feat.view({-1});
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = out_seq * std::sqrt(out_seq.numel()) / (out_seq.norm() + 1e-8);
        h_state = std::get<1>(r);
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        auto combined = torch::cat({out_seq, feat, action_input});
        return combined_processor->forward(combined);
    }
};
TORCH_MODULE(RewardModel);

class RewardNet {
public:
    RewardNet(bool _training = true, int _T = 256, float _learning_rate = 1e-3, float _alpha = 0.9,
        const std::string &_backup_dir = "bots/bot-1/backup/reward_backup")
        : training(_training), T(_T), learning_rate(_learning_rate), alpha(_alpha), backup_dir(_backup_dir) {
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
            initial.push_back(p.clone());
            param_count += p.numel();
        }
        //log("RewardNet's parameters: " + std::to_string(param_count));
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
        log("-------\nR total dist: step=" + std::to_string(calc_diff()));
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
        auto target = torch::tensor((float)imitate);
        auto reward = compute_reward(action, state, (imitate ? target : output)) * 2 - 1;
        update(output, target);
        return reward.item<float>();
    }

private:
    bool is_training = false, logging = true, training, done_training;
    std::thread trainThread;
    float learning_rate, alpha;
    int T;
    const int num_actions = 10, num_channels = 32, grid_x = 81, grid_y = 81, hidden_size = 160;
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
            coor[0].push_back(p.clone());
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
        for (int k = 0; k < num_channels; ++k)
            sit[k] = state[0][k][grid_x / 2][grid_y / 2].item<float>();
        if (outputs.empty()) {
            prev = sit;
            return reward;
        }
        if (prev[11] < sit[11] || prev[30] < sit[30] || prev[31] < sit[31])
            reward = torch::tensor(1.0f);// up  | kills, total-damage, total-effect = 1
        else if (prev[18] > sit[18] || prev[24] > sit[24] || prev[25] > sit[25])
            reward = torch::tensor(0.0f);// down| Hp, attack-damage, attack-effect = 0
        else if (prev[18] < sit[18] || prev[24] < sit[24] || prev[25] < sit[25] || prev[26] < sit[26])
            reward = torch::tensor(0.9f);// up  | Hp, attack-damage, attack-effect, stamina = 0.9
        else if (prev[26] > sit[26])
            reward -= 0.15;// down | stamina -= 0.15
        else if(!action)
            reward -= 0.1;// nothing | stamina -= 0.1 for doing nothing
        prev = sit;
        return reward;
    }

    void train() {
        time_t ts = time(0);
        auto loss = torch::zeros({});
        for (int i = 0; i < T; ++i) {
            loss += torch::binary_cross_entropy(outputs[i], targets[i]);
            /////////////////////////////////////////////////////
            log("output: " + std::to_string(outputs[i].item<float>()) + ", target:" + std::to_string(targets[i].item<float>()));
            /////////////////////////////////////////////////////
        }
        loss /= T;
        optimizer->zero_grad();
        loss.backward();
        /////////////////////////////////////////////////////
        for (const auto &p: model->named_parameters())
            if (p.value().grad().defined()){
                std::string key = p.key(), k;
                double v = p.value().norm().item<double>();
                double g = p.value().grad().norm().item<double>();
                double coef = g / (v + 1e-8);
                for(int i = 0; i < 32; ++i)
                    if(i >= key.size())
                        k.push_back(' ');
                    else
                        k.push_back(key[i]);
                log(k + " || value-norm: " + std::to_string(v) + " || grad--norm: " + std::to_string(g) + " || grad/value: " + std::to_string(coef));
            }
        /////////////////////////////////////////////////////
        optimizer->step();
        outputs.clear(), targets.clear();
        model->reset_memory();
        log("R: loss=" + std::to_string(loss.item<float>()) + ", time(s)=" + std::to_string(time(0) - ts) + ", step=" + std::to_string(calc_diff()));
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