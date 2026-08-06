/*
MIT License

Copyright (c) 2026 bistoyek21 R.I.C.

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
// Add -lws2_32 on Windows

#include "Modules.hpp"

const std::string bot_code = "bot-bc-focal", backup_path = "bots/bot-bc-focal/backup";

class Agent {
public:
    Agent(bool training = true, bool auto_pilot = false, int T = 1054, float learning_rate = 1e-3,
         const std::string &backup_dir = "bots/bot-bc-fap/backup/agent_backup")
        : training(training), auto_pilot(auto_pilot), T(T), learning_rate(learning_rate), backup_dir(backup_dir) {

#if defined(DISTRIBUTED_LEARNING)
        this->backup_dir = backup_dir = "bots/bot-bc-focal/server_checkpoint";
        std::cout << "=== DISTRIBUTED LEARNING MODE ===" << std::endl;
        client = std::make_unique<AgentClient>(backup_dir);
        
        // Load checkpoint from server
        if (std::filesystem::exists(backup_dir + "/checkpoint.pt")) {
            std::vector<torch::Tensor> checkpoint;
            torch::load(checkpoint, backup_dir + "/checkpoint.pt");
            
            model = PlayerPolicyNet();
            auto params = model->parameters();
            if (checkpoint.size() == params.size()) {
                for (size_t i = 0; i < params.size(); ++i) {
                    params[i].data().copy_(checkpoint[i]);
                }
                std::cout << "Loaded model from server checkpoint" << std::endl;
            } else {
                std::cerr << "Checkpoint size mismatch!" << std::endl;
            }
        }
#else
        std::cout << "=== LOCAL LEARNING MODE ===" << std::endl;
        
    #if defined(CROWDSOURCED_TRAINING) && !defined(DISTRIBUTED_LEARNING)
        std::cout << "Loading backup..." << std::endl;
        request_and_extract_backup(backup_path, bot_code);
        std::cout << "Press space to continue" << std::endl;
        #if !defined(AUTOMATIC)
        while (getch() != ' ');
        #endif
    #endif

        model = PlayerPolicyNet();
        if (!backup_dir.empty()) {
            if (std::filesystem::exists(backup_dir)) {
                log_file.open(backup_dir + "/agent_log.log", std::ios::app);
                try {
                    torch::load(model, backup_dir + "/model.pt");
                } catch(...) {}
            } else {
                std::filesystem::create_directories(backup_dir);
                log_file.open(backup_dir + "/agent_log.log", std::ios::app);
            }
        }
#endif

#if defined(FREEZE_AGENT_BLOCK)
        this->training = training = false;
        log("Freezing Agent Network parameters.");
#endif

        coor[0] = snap_shot();
        int param_count = 0;
        for (auto &p: coor[0]) {
            initial.push_back(p.detach().clone());
            param_count += p.numel();
        }
        log("Agent's parameters: " + std::to_string(param_count));

#if defined(SLOWMOTION)
        log("SLOWMOTION (training = "
             + std::to_string(training) + 
             ", auto_pilot = " + std::to_string(auto_pilot) + ")");
#else
        log("(training = "
             + std::to_string(training) + 
             ", auto_pilot = " + std::to_string(auto_pilot) + ")");
#endif

        if (!training) {
            model->eval();
        } else {
            model->train();
            
#if !defined(DISTRIBUTED_LEARNING)
            // Only create optimizer in local mode
            optimizer = std::make_unique<torch::optim::AdamW>(
                model->parameters(), 
                torch::optim::AdamWOptions(learning_rate)
            );
            
    #if defined(FREEZE_TL_BLOCK)
            model->freeze_backbone();
            log("Frozen TL block parameters.");
    #endif

            if (!backup_dir.empty() && std::filesystem::exists(backup_dir + "/optimizer.pt")) {
                try {
                    torch::load(*optimizer, backup_dir + "/optimizer.pt");
                } catch (...) {}
            }
#else
            log("Distributed learning: optimizer managed by server");
#endif
        }
        
        auto dummy = torch::zeros({1, num_channels, grid_x, grid_y});

        prev = torch::tensor({5}, torch::kLong);
        model->forward(dummy, prev);
        model->reset_memory();
    }
    
    ~Agent() {
        if (is_training && trainThread.joinable()) {
            std::cout << "Agent Network is updating...\n";
            trainThread.join();
            std::cout << "done!" << std::endl;
        }
        
        if (training) {
            coor[0].clear();
            for (auto &p: initial)
                coor[0].push_back(p.detach().clone());
            log("-------\nA total dist: step=" + std::to_string(calc_diff()));
            log("======================");
        } else {
            log("-------\nA total dist: step=0.000000");
            log("======================");
        }
        log_file.close();
        
#if !defined(DISTRIBUTED_LEARNING)
        if (training && !backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            model->reset_memory();
            torch::save(model, backup_dir + "/model.pt");
            if (optimizer) {
                torch::save(*optimizer, backup_dir + "/optimizer.pt");
            }
        }

    #if defined(CROWDSOURCED_TRAINING)
        std::cout << "Submit backup to server? (y/n)" << std::endl;
        #if !defined(AUTOMATIC)
        if (getch() == 'y') {
            std::cout << "Submitting..." << std::endl;
            zip_and_return_backup(backup_path);
            std::cout << "Done! Press space" << std::endl;
            while (getch() != ' ');
        }
        #endif
    #endif

#endif
    }

    int predict(const std::vector<float>& obs) {
        if (cnt <= T_initial)
            return 0;
            
        if (is_training) {
#if !defined(CROWDSOURCED_TRAINING) && !defined(DISTRIBUTED_LEARNING)
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
            } else {
                return 0;
            }
#else
            return 0;
#endif
        }
    
        if(!manual)
            return rand() % num_actions;

        auto state = torch::tensor(obs, torch::dtype(torch::kFloat32)).view({1, num_channels, grid_x, grid_y});

        states.push_back(state);
        auto output = model->forward(state, prev);

        
        values.push_back(output.first[0]);

        auto prob = torch::softmax(output.second, /*dim=*/1)[0];
        log_probs.push_back(torch::log(prob));
        
        std::vector<float> v;
        for (int i = 0; i < num_actions; ++i)
            v.push_back(prob[i].item<float>());
        
        return max_element(v.begin(), v.end()) - v.begin();
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T_initial)
            return;
        
        if (!manual)
            return;
        
        prev = torch::tensor({action}, torch::kLong);
        actions.push_back(prev.detach().clone());
        
        if (actions.size() == T + W) {
            is_training = true;
            done_training = false;
            if (training)
                std::cout << "Agent is training..." << std::endl;
            train();
            is_training = false;
            if (training)
                std::cout << "done!" << std::endl;
        }
    }

#if defined(CROWDSOURCED_TRAINING) || defined(DISTRIBUTED_LEARNING)
    bool is_manual() {
        
        if (!is_training && cnt <= T_initial)
            ++cnt;
        if (is_training) {
            if (done_training) {
                is_training = false;
                if (trainThread.joinable())
                    trainThread.join();
            } else {
                return true;
            }
        }
        
        if (cnt <= T_initial) {
            manual = true;
            cnt_warm_up = 0;
        } else if (actions.empty()) {
            if (cnt_warm_up == T_warm_up) {
                manual = !auto_pilot;
                if (manual) {
                    std::cout << "manual part! press space button to continue" << std::endl;
                    #if !defined(AUTOMATIC)
                    while(getch() != ' ');
                    #endif
                    std::cout << "space button pressed!" << std::endl;
                }
            }
            else
                manual = false;
            if (cnt_warm_up < T_warm_up)
                ++cnt_warm_up;
        }
        return manual;
    }
#endif

    bool in_training() {
        return is_training;
    }

private:
    bool is_training = false, logging = true, training, auto_pilot, done_training = false, manual = false;
    std::thread trainThread;
    float learning_rate;
    int T, W = 30, cnt = 0, cnt_warm_up = 0, T_initial = 512, T_warm_up = 100;
    const int num_actions = 9, num_channels = 32, grid_x = 31, grid_y = 31, hidden_size = 160;
    std::string backup_dir;
    PlayerPolicyNet model{nullptr};
    
#if !defined(DISTRIBUTED_LEARNING)
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};
#else
    std::unique_ptr<AgentClient> client{nullptr};
#endif
    
    std::vector<torch::Tensor> states, log_probs, values, rewards, actions;
    torch::Tensor prev;
    std::ofstream log_file;
    std::vector<torch::Tensor> coor[2], initial;

    std::vector<torch::Tensor> snap_shot() {
        std::vector<torch::Tensor> params;
        for (auto& p : model->parameters())
            params.push_back(p.detach().clone());
        return params;
    }

    double calc_diff() {
        coor[1] = snap_shot();
        double diff = 0;
        for (size_t i = 0; i < coor[0].size(); ++i)
            diff += (coor[1][i] - coor[0][i]).pow(2).sum().item<float>();
        coor[0].clear();
        for (auto& p: coor[1])
            coor[0].push_back(p.detach().clone());
        coor[1].clear();
        return std::sqrt(diff);
    }

    template<typename Type>
    void log(const Type& message) {
        if (!logging) return;
        log_file << message << std::endl;
        log_file.flush();
    }

    void train() {
        time_t ts = time(0);

        std::vector<std::vector<int>> groups = {
            {0, 5, 6, 7, 8},
            {1, 2},
            {3, 4}
        };

        // Count how many times each action appeared
        std::vector<int> counts(num_actions, 0), group_counts(groups.size(), 0);
        for (int t = W; t < T + W; ++t)
            ++counts[actions[t].item<int>()];

        for (int g = 0; g < groups.size(); ++g)
            for (auto a: groups[g])
                group_counts[g] += counts[a];

        // Build per-group average loss tensors (differentiable)
        std::vector<torch::Tensor> per_group_losses(groups.size());
        auto H = torch::zeros({1}), b_loss = torch::zeros({1});
        
        double acc = 0;

        for (int t = W; t < T + W; ++t) {
            H -= (log_probs[t] * torch::exp(log_probs[t])).sum();
            b_loss -= log_probs[t][actions[t]];
            
            int is_true = 1;
            for(int act = 0; act < num_actions && is_true; ++act)
                if(actions[t].item<int>() != act && 
                    log_probs[t][actions[t].item<int>()].item<float>() <= log_probs[t][act].item<float>())
                    is_true = 0;
            acc += is_true;
        }

        acc /= T;
        
        b_loss /= T, H /= T;
        b_loss = b_loss.detach(), H = H.detach();
        auto loss = b_loss - 0.05 * H;

        std::vector<std::vector<torch::Tensor>> group_grads;  // per active group

        std::vector<double> g_acc;
        
        int last = -1;

        for (int g = 0; g < groups.size(); ++g)
            if(group_counts[g] > 0)
                last = g;
        
        for (int g = 0; g < groups.size(); ++g) {
            per_group_losses[g] = torch::zeros({1});

            g_acc.push_back(-1);

            if (group_counts[g] == 0){
#if defined(DISTRIBUTED_LEARNING)
                if(training){
                    std::vector<torch::Tensor> grads;
                    for (auto& p : model->parameters())
                        grads.push_back(torch::zeros_like(p));
                    group_grads.push_back(grads);            
                }
#endif
                continue;
            }
            
            g_acc.back() = 0;

            for (int t = W; t < T + W; ++t)
                for (auto a: groups[g])
                    if (actions[t].item<int>() == a) {
                        per_group_losses[g] -= log_probs[t][actions[t]];
                        int is_true = 1;
                        for(int act = 0; act < num_actions && is_true; ++act)
                            if(actions[t].item<int>() != act && 
                                log_probs[t][actions[t].item<int>()].item<float>() <= log_probs[t][act].item<float>())
                                is_true = 0;
                        g_acc.back() += is_true;
                    }
            
            per_group_losses[g] /= group_counts[g];

            g_acc.back() /= group_counts[g];
#if defined(DISTRIBUTED_LEARNING)
            if(training){
                model->zero_grad();
                if (g + 1 == groups.size() || g == last)
                    per_group_losses[g].backward();
                else
                    per_group_losses[g].backward({}, /*retain_graph=*/true);
                std::vector<torch::Tensor> grads;
                for (auto& p : model->parameters())
                    if (p.grad().defined())
                        grads.push_back(p.grad().detach().clone());
                    else
                        grads.push_back(torch::zeros_like(p));
                group_grads.push_back(grads);            
            }
#endif
        }
    
        
        model->zero_grad();


        if (training) {
#if defined(DISTRIBUTED_LEARNING)
            // Distributed: send per-group gradients + counts to server
            int sz = model->parameters().size(), idx = 0;
            std::vector<std::vector<torch::Tensor>> _group_grads;
            for (int g = 0; g < groups.size(); ++g) {
                _group_grads.push_back({});
                if(group_counts[g] > 0) {
                    for (int p = 0; p < sz; ++p)
                        _group_grads.back().push_back(group_grads[idx][p].clone().detach());
                    ++idx;
                    continue;
                }
                for (int p = 0; p < sz; ++p)
                    _group_grads.back().push_back(torch::zeros_like(model->parameters()[p]));
            }
            client->send_gradient(group_counts, torch::stack(per_group_losses).view({-1}), _group_grads);
            auto update_vector = client->get_update_vector();
            auto params = model->parameters();
            for (size_t i = 0; i < params.size() && i < update_vector.size(); ++i)
                params[i].data() += update_vector[i];

            log("A: loss=" + std::to_string(loss.item<float>()) +
            ", b_loss=" + std::to_string(b_loss.item<float>()) +
            ", H=" + std::to_string(H.item<float>()) +
            ", time(s)=" + std::to_string(time(0) - ts) +
            ", step=" + std::to_string(calc_diff()) +
            "\nPer Group Losses=[" +
            [&]{
                std::string s;
                for (int g = 0; g < groups.size(); ++g) {
                    if (g)
                        s += ", ";
                    s += std::to_string(per_group_losses[g].item<double>());
                }
                return s;
            }() +
            "]");
#else
            auto total_loss = torch::zeros({1});
            for (int i = 0; i < groups.size(); ++i)
                total_loss += per_group_losses[i] * per_group_losses[i];
            total_loss.backward();
            optimizer->step();
            log("A: loss=" + std::to_string(loss.item<float>()) +
            ", b_loss=" + std::to_string(b_loss.item<float>()) +
            ", total_loss=" + std::to_string(total_loss.item<float>()) +
            ", H=" + std::to_string(H.item<float>()) +
            ", time(s)=" + std::to_string(time(0) - ts) +
            ", step=" + std::to_string(calc_diff()) +
            "\nPer Group Losses=[" +
            [&]{
                std::string s;
                for (int g = 0; g < groups.size(); ++g) {
                    if (g)
                        s += ", ";
                    s += std::to_string(per_group_losses[g].item<double>());
                }
                return s;
            }() + "]");
#endif
        }
        
        else {
            auto total_loss = torch::zeros({1});
            for (int i = 0; i < groups.size(); ++i)
                total_loss += per_group_losses[i] * per_group_losses[i];

            log("A: loss=" + std::to_string(loss.item<float>()) +
            ", b_loss=" + std::to_string(b_loss.item<float>()) +
            ", total_loss=" + std::to_string(total_loss.item<float>()) +
            ", H=" + std::to_string(H.item<float>()) +
            ", time(s)=" + std::to_string(time(0) - ts) +
            ", step=" + std::to_string(calc_diff()) +
            "\nPer Group Losses=[" +
            [&]{
                std::string s;
                for (int g = 0; g < groups.size(); ++g) {
                    if (g)
                        s += ", ";
                    s += std::to_string(per_group_losses[g].item<double>());
                }
                return s;
            }() + "]");
        }
        
        log("Total Accuracy=" + std::to_string(acc));

        log("Per Group Accuracies=[" +
            [&]{
                std::string s;
                for (size_t i=0; i< groups.size(); ++i) {
                    if (i) s += ", ";
                    s += std::to_string(g_acc[i]);
                }
                return s;
            }() +
            "]");

        // Cleanup
        actions.clear();
        rewards.clear();
        log_probs.clear();
        states.clear();
        values.clear();
        model->reset_memory();
        done_training = true;
    }
};