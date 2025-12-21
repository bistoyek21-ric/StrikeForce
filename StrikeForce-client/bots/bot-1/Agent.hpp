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
    Backbone backbone{nullptr};
    torch::nn::Sequential value_head{nullptr}, policy_head{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;

    AgentModelImpl(int num_channels = 32, int grid_x = 39, int grid_y = 39, int hidden_size = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y), hidden_size(hidden_size), num_actions(num_actions) {
        backbone = register_module("backbone", Backbone(num_channels, grid_x, grid_y, hidden_size, num_actions));
        value_head = register_module("value", torch::nn::Sequential(
            ResB(hidden_size, LAYER_INDEX), torch::nn::Linear(hidden_size, 1)
        ));
        policy_head = register_module("policy", torch::nn::Sequential(
            ResB(hidden_size, LAYER_INDEX), torch::nn::Linear(hidden_size, num_actions)
        ));
        backbone->reset_memory();
    }

    void import_tl_block_rewardnet(const RewardModel& reward_model) {
        for (auto& p : reward_model->backbone->named_parameters())
            backbone->named_parameters()[p.key()] = p.value().detach().clone();
    }

    void freeze_tl_block(){
        for (auto& p : backbone->parameters())
            p.set_requires_grad(false);
    }

    void reset_memory() {
        backbone->reset_memory();
    }

    void update_actions(int action) {
        backbone->update_actions(action);
    }

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        auto gated = backbone->forward(x);

        auto logits = policy_head->forward(gated);
        auto p = torch::softmax(logits, -1) + 1e-8;
        
        auto v = torch::sigmoid(value_head->forward(gated).squeeze());

        return {p, v};
    }
};
TORCH_MODULE(AgentModel);

class Agent {
public:
    Agent(bool training = true, int T = 1024, int num_epochs = 4, float gamma = 0.98, float learning_rate = 1e-3,
         float ppo_clip = 0.2, float cv = 0.5, const std::string &backup_dir = "bots/bot-1/backup/agent_backup")
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
#if defined(FREEZE_AGENT_BLOCK)
        this->training = training = false;
        log("Freezing Agent Network parameters.");
#endif
#if defined(TL_IMPORT_REWARDNET)
        model->import_tl_block_rewardnet(reward_net->get_model());
        log("Imported TL block from Reward Network.");
#endif
        coor[0] = snap_shot();
        int param_count = 0;
        for (auto &p: coor[0]) {
            initial.push_back(p.detach().clone());
            param_count += p.numel();
        }
        log("Agent's parameters: " + std::to_string(param_count));
        log("LAYER_INDEX=" + std::to_string(LAYER_INDEX));
#if defined(SLOWMOTION)
        log("SLOWMOTION");
#endif
        if (!training)
            model->eval();
        else {
            model->train();
            optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(learning_rate));
#if defined(FREEZE_TL_BLOCK)
            model->freeze_tl_block();
            log("Frozen TL block parameters.");
#endif
            if (!backup_dir.empty() && std::filesystem::exists(backup_dir + "/optimizer.pt")) {
                try {
                    torch::load(*optimizer, backup_dir + "/optimizer.pt");
                } catch (...) {}
            }
        }
        auto dummy = torch::zeros({num_channels, 1, 9, (grid_x + 2) / 3, (grid_y + 2) / 3});
        action_input_checkpoint = model->backbone->action_input.clone().detach();
        h_state_checkpoint[0] = model->backbone->h_state[0].clone().detach();
        h_state_checkpoint[1] = model->backbone->h_state[1].clone().detach();
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
        if (training) {
            coor[0].clear();
            for (auto &p: initial)
                coor[0].push_back(p.detach().clone());
            log("-------\nA total dist: step=" + std::to_string(calc_diff()));
            log("======================");
        }
        else {
            log("-------\nA total dist: step=0.000000");
            log("======================");
        }
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
        std::vector<float> v;
        for (int i = 0; i < num_actions; ++i)
            v.push_back(output[0][i].item<float>());
#if !defined(SLOWMOTION)
        for (int i = 1; i < num_actions; ++i)
            v[i] *= 0.5f / (1 - v[0] + 1e-5f);
        v[0] = 0.5f;
#endif
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        return dist(gen);
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T_initial)
            return;
        rewards.push_back(reward_net->get_reward(action, imitate, states.back().detach().clone()));
        if (rewards.back() == -2 && training) {
            actions.clear(), rewards.clear(), log_probs.clear();
            states.clear(), values.clear();
            return;
        }
        model->update_actions(action);
        actions.push_back(action);
        if (actions.size() == T) {
            is_training = true;
            done_training = false;
            //trainThread = std::thread(&Agent::train, this);
            ///////////////////////////////////////////////
            if (training)
                std::cout << "Agent is training..." << std::endl;
            train();
            is_training = false;
            if (training)
                std::cout << "done!" << std::endl;
            /////////////////////////////////////////////
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
            if (manual) {
                std::cout << "manual part! press space button to continue" << std::endl;
                while(getch() != ' ');
                std::cout << "space button pressed!" << std::endl;
            }
        }
        else if (actions.size() == T / 2) {
            manual = !manual;
            if (manual) {
                std::cout << "manual part! press space button to continue" << std::endl;
                while(getch() != ' ');
                std::cout << "space button pressed!" << std::endl;
            }
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
    int T, num_epochs, cnt = 0, T_initial = 512;
    const int num_actions = 9, num_channels = 32, grid_x = 39, grid_y = 39, hidden_size = 160;
    std::string backup_dir;
    AgentModel model{nullptr};
    RewardNet* reward_net;
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};
    std::vector<torch::Tensor> states, log_probs, values;
    torch::Tensor action_input_checkpoint;
    torch::Tensor h_state_checkpoint[2];
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
        std::vector<torch::Tensor> returns(T);
        auto ret = torch::zeros({});
        for (int i = T - 1; i >= 0; --i) {
            ret = gamma * ret + (1 - gamma) * rewards[i];
            returns[i] = ret.detach().clone();
        }
        return returns;
    }

    void train() {
        float sum_rewards[2] = {}, nothing[2] = {};
        for (int i = 0; i < T; ++i){
            sum_rewards[i / (T / 2)] += rewards[i];
            nothing[i / (T / 2)] += (int)(!actions[i]);
        }
        log("A stats: r_avg0=" + std::to_string(sum_rewards[0] / T) +
            "|r_avg1=" + std::to_string(sum_rewards[1] / T) +
            "|n_avg0=" + std::to_string(nothing[0] / (T / 2)) +
            "|n_avg1=" + std::to_string(nothing[1] / (T / 2)) + "|manual=" + std::to_string(manual));
        auto sum = torch::exp(log_probs[0].detach().clone());
        for (int i = 1; i < T; ++i)
            sum += torch::exp(log_probs[i].detach().clone());
        sum /= T;
        log("A probs:");
        std::string pref;
        for (int i = 0; i < num_actions; ++i)
            pref += std::to_string(sum[i].item<float>()) + "|";
        log(pref);
        auto returns = computeReturns();
        auto tmp_a_i = model->backbone->action_input.detach().clone();
        torch::Tensor tmp_h_s[2] = {model->backbone->h_state[0].detach().clone(), 
            model->backbone->h_state[1].detach().clone()};
        int times = 1;
#if defined(FREEZE_TL_BLOCK)
        times = 2;
#endif
        for (int epoch = 0; epoch < num_epochs * times; ++epoch) {
            time_t ts = time(0);
            auto p_loss = torch::zeros({});
            auto v_loss = torch::zeros({});
            model->backbone->action_input = action_input_checkpoint.clone().detach();
            model->backbone->h_state[0] = h_state_checkpoint[0].clone().detach();
            model->backbone->h_state[1] = h_state_checkpoint[1].clone().detach();
            for (int i = 0; i < T * 3 / 4; ++i) {
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
                    v_loss += torch::mse_loss(values[i], returns[i]) * (1 + i / (T / 2));
                    values[i] = values[i].detach();
                }
                else
                    v_loss += torch::mse_loss(output[1], returns[i]) * (1 + i / (T / 2));
                auto diff = torch::clamp(current_logp - log_probs[i][actions[i]], -100, 10);
                auto ratio = torch::exp(diff);
                float ub = std::min(2.0f, 1 + (epoch + 1) * ppo_clip);
                float lb = std::max(0.5f, 1.0f / ub);
                auto clipped = torch::clamp(ratio, lb, ub);
                auto adv = returns[i] - values[i];
                p_loss -= torch::min(ratio * adv, clipped * adv) * (1 + i / (T / 2));
            }
            p_loss = p_loss / T;
            v_loss = v_loss / T;
            auto loss = p_loss + cv * v_loss;
            if (training) {
                optimizer->zero_grad();
                loss.backward();
                optimizer->step();
            }
            log("A: loss=" + std::to_string(loss.item<float>()) +
             "|p_loss=" + std::to_string(p_loss.item<float>()) + 
             "|v_loss=" + std::to_string(v_loss.item<float>()) + 
             ",time(s)=" + std::to_string(time(0) - ts) +
             ",step=" + std::to_string(calc_diff()));
        }
        actions.clear(), rewards.clear(), log_probs.clear();
        states.clear(), values.clear();
        action_input_checkpoint = tmp_a_i.clone().detach();
        h_state_checkpoint[0] = tmp_h_s[0].clone().detach();
        h_state_checkpoint[1] = tmp_h_s[1].clone().detach();
        model->backbone->action_input = action_input_checkpoint.clone().detach();
        model->backbone->h_state[0] = h_state_checkpoint[0].clone().detach();
        model->backbone->h_state[1] = h_state_checkpoint[1].clone().detach();
        done_training = true;
    }
};