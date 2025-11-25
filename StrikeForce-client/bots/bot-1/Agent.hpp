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
    GameCNN cnn{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    torch::nn::Sequential policy_head{nullptr}, value_head{nullptr}, gate{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;
    const float alpha;

    torch::Tensor action_input, h_state;

    AgentModelImpl(int num_channels = 32, int grid_x = 39, int grid_y = 39, int hidden_size = 160, int num_actions = 9, float alpha = 0.9f)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y), hidden_size(hidden_size), num_actions(num_actions), alpha(alpha) {
        cnn = register_module("cnn", GameCNN(num_channels, hidden_size, 6, grid_x));
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2)));
        gate = register_module("gate", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size + num_actions, hidden_size),
            torch::nn::ReLU()
        ));
        value_head = register_module("value", torch::nn::Sequential(
            ResB(hidden_size, 3), torch::nn::Linear(hidden_size, 1),
            torch::nn::Sigmoid()
        ));
        policy_head = register_module("policy", torch::nn::Sequential(
            ResB(hidden_size, 3), torch::nn::Linear(hidden_size, num_actions)
        ));
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    void reset_memory() {
        action_input = torch::zeros({num_actions});
        h_state = torch::zeros({2, 1, hidden_size});
    }

    void update_actions(int action) {
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
    }

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        auto feat = cnn->forward(x);
        feat = feat * hidden_size / (feat.abs().sum().detach() + 1e-8);
        auto r = gru->forward(feat.view({1, 1, -1}), h_state);
        feat = feat.view({-1});
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = out_seq * hidden_size / (out_seq.abs().sum().detach() + 1e-8);
        h_state = std::get<1>(r);
        auto gate_input = torch::cat({out_seq, feat, action_input * hidden_size / (action_input.abs().sum().detach() + 1e-8)});
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
    Agent(bool training = true, int T = 512, int num_epochs = 4, float gamma = 0.99, float learning_rate = 1e-3,
         float ppo_clip = 0.2, float alpha = 0.9, float cv = 0.5, const std::string &backup_dir = "bots/bot-1/backup/agent_backup")
        : training(training), T(T), num_epochs(num_epochs), gamma(gamma), learning_rate(learning_rate),
        ppo_clip(ppo_clip), alpha(alpha), cv(cv), backup_dir(backup_dir) {
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
            }
        }
        coor[0] = snap_shot();
        int param_count = 0;
        for (auto &p: coor[0]) {
            initial.push_back(p.detach().clone());
            param_count += p.numel();
        }
        //log("Agent's parameters: " + std::to_string(param_count));
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
            coor[0].push_back(p.detach().clone());
        log("-------\nA total dist: step=" + std::to_string(calc_diff()));
        log("======================");
        log_file.close();
        if (training && !backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            model->reset_memory();
            torch::save(model, backup_dir + "/model.pt");
            torch::save(*optimizer, backup_dir + "/optimizer.pt");
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
        if (cnt <= T_initial)
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
        auto state = torch::tensor(obs, torch::dtype(torch::kFloat32)).view({num_channels, 1, 9, (grid_x + 2) / 3, (grid_y + 2) / 3});
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
        if (is_training || cnt <= T_initial)
            return;
        rewards.push_back(reward_net->get_reward(action, imitate, states.back()));
        if (rewards.back() == -2 && training) {
            actions.clear(), rewards.clear(), log_probs.clear();
            states.clear(), values.clear();
            model->reset_memory();
            return;
        }
        model->update_actions(action);
        actions.push_back(action);
        if (actions.size() == 2 * T) {
            if (training) {
                is_training = true;
                done_training = false;
                //trainThread = std::thread(&Agent::train, this);
                ///////////////////////////////////////////////
                std::cout << "Agent is training..." << std::endl;
                train();
                is_training = false;
                std::cout << "done!" << std::endl;
                /////////////////////////////////////////////
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
        if (!is_training && cnt <= T_initial)
            ++cnt;
        if (is_training) {
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
            }
            else
               return true;
        }
        if (cnt <= T_initial)
            manual = true;
        else if (actions.empty()) {
            std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<> dist(0, 1);
            manual = dist(gen);
            /////////////////////////////
            if (manual) {
                std::cout << "manual part! press space button to continue" << std::endl;
                while(getch() != ' ');
                std::cout << "space button pressed!" << std::endl;
            }
            ////////////////////////////
        }
        else if (actions.size() == T / 2 || actions.size() == T * 3 / 2) {
            manual = !manual;
            /////////////////////////////
            if (manual) {
                std::cout << "manual part! press space button to continue" << std::endl;
                while(getch() != ' ');
                std::cout << "space button pressed!" << std::endl;
            }
            ////////////////////////////
        }
        return manual;
    }
#endif

    bool in_training(){
        return is_training;
    }

private:
    bool is_training = false, logging = true, training, done_training, manual;
    std::thread trainThread;
    float learning_rate, alpha, gamma, ppo_clip, cv;
    int T, num_epochs, cnt = 0, T_initial = 128;
    const int num_actions = 9, num_channels = 32, grid_x = 39, grid_y = 39, hidden_size = 160;
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

    std::vector<torch::Tensor> computeReturns() {
        std::vector<torch::Tensor> returns(2 * T);
        auto ret = torch::zeros({});
        for (int i = 2 * T - 1; i >= 0; --i) {
            ret = rewards[i] + gamma * ret;
            returns[i] = ret * (1 - gamma);
        }
        return returns;
    }

    void train() {
        auto returns = computeReturns();
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            time_t ts = time(0);
            auto p_loss = torch::zeros({});
            auto v_loss = torch::zeros({});
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
                if (!epoch) {
                    v_loss += torch::mse_loss(values[i], returns[i]);
                    values[i] = values[i].detach();
                }
                else
                    v_loss += torch::mse_loss(output[1], returns[i]);
                auto diff = torch::clamp(current_logp - log_probs[i][actions[i]], -100, 10);
                auto ratio = torch::exp(diff);
                auto adv = returns[i] - values[i];
                auto clipped = torch::clamp(ratio, 1 - (epoch + 1) * ppo_clip, 1 + (epoch + 1) * ppo_clip);
                p_loss -= torch::min(ratio * adv, clipped * adv);
            }
            auto loss = (p_loss + cv * v_loss) / T;
            optimizer->zero_grad();
            loss.backward();
            /////////////////////////////////////////////////////
            /*
            for (const auto &p: model->named_parameters())
                if (p.value().grad().defined()) {
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
            log("A: loss=" + std::to_string(loss.item<float>()) + "| p_loss=" + std::to_string(p_loss.item<float>() / T) + "| v_loss=" + std::to_string(v_loss.item<float>() / T) + ", time(s)=" + std::to_string(time(0) - ts) + ", step=" + std::to_string(calc_diff()));
        }
        actions.clear(), rewards.clear(), log_probs.clear();
        states.clear(), values.clear();
        model->reset_memory();
        done_training = true;
    }
};