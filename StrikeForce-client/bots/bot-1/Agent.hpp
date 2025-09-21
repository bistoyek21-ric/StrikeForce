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

const std::string bot_code = "bot-1", backup_path = "bots/bot-1/backup";

struct AgentModelImpl : torch::nn::Module {
    MyCNN cnn{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential policy_head{nullptr}, value_head{nullptr}, gate{nullptr};
    torch::nn::LayerNorm gru_norm{nullptr}, action_norm{nullptr};

    const int num_channels = 32, hidden_size = 128, num_actions = 10;
    const float alpha = 0.9f;

    torch::Tensor action_input, h_state;

    AgentModelImpl(int n_channels=32, int hidden=128, int n_actions=10, float alpha_=0.9f)
        : num_channels(n_channels), hidden_size(hidden), num_actions(n_actions), alpha(alpha_){
        cnn = register_module("cnn", MyCNN(num_channels, hidden_size));
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2)));
        action_processor = register_module("action_proc", torch::nn::Sequential(
            torch::nn::Linear(num_actions, hidden_size), torch::nn::ReLU(),
            ResidualBlock(hidden_size, 2)
        ));
        gate = register_module("gate", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size, hidden_size), torch::nn::ReLU()
        ));
        value_head = register_module("value", torch::nn::Sequential(
            ResidualBlock(hidden_size, 6), torch::nn::Linear(hidden_size, 1),
            torch::nn::Sigmoid()
        ));
        policy_head = register_module("policy", torch::nn::Sequential(
            ResidualBlock(hidden_size, 6), torch::nn::Linear(hidden_size, num_actions)
        ));
        gru_norm = register_module("gru_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
        action_norm = register_module("action_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
        reset_memory();
    }

    void reset_memory() {
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    void update_actions(int action) {
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
    }

    std::vector<torch::Tensor> forward(const torch::Tensor &state) {
        auto feat = cnn->forward(state);
        feat = feat.view({1, 1, -1});
        auto r = gru->forward(feat, h_state);
        auto out_seq = std::get<0>(r).view({-1}) + feat.view({-1});
        out_seq = gru_norm->forward(out_seq);
        h_state = std::get<1>(r).detach();
        auto a_feat = action_processor->forward(action_input);
        a_feat = action_norm->forward(a_feat.view({-1}));
        auto gate_input = torch::cat({out_seq, a_feat});
        auto gated = gate->forward(gate_input);
        auto logits = policy_head->forward(gated);
        auto p = torch::softmax(logits, -1) + 1e-8;
        auto v = 2 * value_head->forward(gated).squeeze() - 1;
        return {p, v};
    }
};
TORCH_MODULE(AgentModel);

class Agent {
public:
    Agent(bool _training = true, int _T = 256, int _num_epochs = 4, float _gamma = 0.99, float _learning_rate = 1e-2,
         float _ppo_clip = 0.2, float _alpha = 0.9, const std::string &_backup_dir = "bots/bot-1/backup/agent_backup")
        : training(_training), T(_T), num_epochs(_num_epochs), gamma(_gamma), learning_rate(_learning_rate),
        ppo_clip(_ppo_clip), alpha(_alpha), backup_dir(_backup_dir){
#if defined(CROWDSOURCED_TRAINING)
        std::cout << "loading backup ..." << std::endl;
        request_and_extract_backup(backup_path, bot_code);
        std::cout << "done!" << std::endl;
        std::cout << "press space to continue" << std::endl;
        while (getch() != ' ');
#endif
        reward_net = new RewardNet();
        model = AgentModel();
        if (!backup_dir.empty()) {
            if (std::filesystem::exists(backup_dir)) {
                log_file.open(backup_dir + "/agent_log.log", std::ios::app);
                try{
                    torch::load(model, backup_dir + "/model.pt");
                } catch(...){}
            } else {
                std::filesystem::create_directories(backup_dir);
                log_file.open(backup_dir + "/agent_log.log", std::ios::app);
                coor[0] = snap_shot();
                int sum = 0;
                for (auto &p: coor[0]) {
                    initial.push_back(p.clone());
                    sum += p.numel();
                }
                log("A parameters: " + std::to_string(sum));
            }
        }
        model->to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        if (!training)
            model->eval();
        else
            optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(learning_rate));
        auto dummy = torch::zeros({1, num_channels, grid_x, grid_y});
        model->forward(dummy);
        model->reset_memory();
    }
    
    ~Agent() {
        delete reward_net;
        if (is_training)
            if (trainThread.joinable()) {
                std::cout << "Agent Network is updating...\nthis might take a few seconds" << std::endl;
                trainThread.join();
                std::cout << "done!" << std::endl;
            }
        coor[0].clear();
        for (auto &p: initial)
            coor[0].push_back(p.clone());
        log("-------\nA total dist: " + calc_diff());
        log("======================");
        log_file.close();
        if (training && !backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            model->reset_memory();
            model->to(torch::kCPU);
            torch::save(model, backup_dir + "/model.pt");
        }
#if defined(CROWDSOURCED_TRAINING)
        std::cout << "Do you want to submit your backup into our server?\n(y:yes/any other key:no)" << std::endl;
        if (getch() == 'y') {
            std::cout << "this might takes a few seconds....\n------------" << std::endl;
            zip_and_return_backup(backup_path);
            std::cout << "\n------------\ndone!" << std::endl;
            std::cout << "press space to continue" << std::endl;
            while (getch() != ' ');
        }
#endif
    }

    int predict(const std::vector<float>& obs) {
        if (cnt <= T)
            return 0;
        if (is_training) {
#if !defined(CROWDSOURCED_TRAINING)
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
            }
            else
               return 0;
#else
            return 0;
#endif
        }
        auto state = torch::clamp(torch::tensor(obs, torch::dtype(torch::kFloat32)).view({1, num_channels, grid_x, grid_y}), 0, 3);
        states.push_back(state);
        auto output = model->forward(state);
        values.push_back(output[1]);
        log_probs.push_back(torch::log(output[0]));
        std::vector<float> v(log_probs.back().data_ptr<float>(), log_probs.back().data_ptr<float>() + log_probs.back().numel());
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        return dist(gen);
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T)
            return;
        rewards.push_back(reward_net->get_reward(action, imitate, states.back()));
        if (rewards.back() == -2 && training) {
            log_probs.clear();
            rewards.clear();
            states.clear();
            values.clear();
            return;
        }
        model->update_actions(action);
        actions.push_back(action);
        if (actions.size() == T) {
            if (training) {
                is_training = true;
                done_training = false;
                trainThread = std::thread(&Agent::train, this);
            }
            else {
                actions.clear();
                rewards.clear();
                log_probs.clear();
                states.clear();
                values.clear();
            }
        }
    }

#if defined(CROWDSOURCED_TRAINING)
    bool is_manual() {
        if (!is_training && cnt <= T)
            ++cnt;
        if (is_training) {
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
                std::mt19937 gen(std::random_device{}());
                std::uniform_int_distribution<> dist(0, 1);
                manual = dist(gen);
            }
            else
               return true;
        }
        if (cnt <= T)
            manual = true;
        if (cnt == T + 1) {
            std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<> dist(0, 1);
            manual = dist(gen);
            ++cnt;
        }
        else if (cnt == T + 2)
            ++cnt;
        if (actions.size() == T / 2)
            manual = !manual;
        return manual;
    }
#endif

    bool in_training(){
        return is_training;
    }

private:
    bool is_training = false, logging = true, training, done_training, manual;
    std::thread trainThread;
    float learning_rate, alpha, gamma, ppo_clip;
    int T, num_epochs, cnt = 0;
    const int num_actions = 10, num_channels = 32, grid_x = 27, grid_y = 87, hidden_size = 128;
    std::string backup_dir;
    AgentModel model{nullptr};
    RewardNet* reward_net;
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};
    std::vector<torch::Tensor> states, log_probs, values;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::ofstream log_file;

    std::vector<torch::Tensor> coor[2], initial;
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

    torch::Tensor computeReturns() {
        torch::Tensor returns = torch::zeros({T});
        double running = 0;
        for (int i = T - 1; i >= 0; --i) {
            running = rewards[i] + gamma * running;
            returns[i] = running;
        }
        return returns;
    }

    void train() {
        time_t ts = time(0);
        auto returns = computeReturns();
        std::vector<torch::Tensor> advantages;
        for (int i = 0; i < T; ++i)
            advantages.push_back(returns[i] - values[i].detach());
        torch::Tensor adv_tensor = torch::stack(advantages);
        torch::Tensor adv_mean = adv_tensor.mean();
        torch::Tensor adv_std = adv_tensor.std();
        if (adv_std.item<float>() < 1e-8)
            adv_tensor = adv_tensor - adv_mean;
        else
            adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-5);
        float loss_g;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            torch::Tensor p_loss = torch::zeros({});
            torch::Tensor v_loss = torch::zeros({});
            model->reset_memory();
            for (int i = 0; i < T; ++i) {
                torch::Tensor current_logp;
                std::vector<torch::Tensor> output;
                if (!epoch) {
                    current_logp = log_probs[i][actions[i]];
                    log_probs[i] = log_probs[i].detach();
                }
                else {
                    output = model->forward(states[i]);
                    model->update_actions(actions[i]);
                    current_logp = torch::log(output[0][actions[i]]);
                }
                auto diff = torch::clamp(current_logp - log_probs[i][actions[i]], -100, 10);
                auto ratio = torch::exp(diff);
                auto clipped = torch::clamp(ratio, 1 - ppo_clip, 1 + ppo_clip);
                p_loss -= torch::min(ratio * adv_tensor[i], clipped * adv_tensor[i]);
                if (!epoch)
                    v_loss += torch::mse_loss(values[i], returns[i]);
                else
                    v_loss += torch::mse_loss(output[1], returns[i]);
            }
            auto loss = (p_loss + 0.25 * v_loss) / T;
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            loss_g = loss.item<float>();
        }
        actions.clear(), rewards.clear(), log_probs.clear();
        states.clear(), values.clear();
        model->reset_memory();
        log("A: loss=" + std::to_string(loss_g) + ", time(s)=" + std::to_string(time(0) - ts) + "," + calc_diff());
        done_training = true;
    }
};