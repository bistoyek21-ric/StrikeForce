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

const std::string bot_code = "bot-bc-mgda", backup_path = "bots/bot-bc-mgda/backup";

class Agent {
public:
    Agent(bool training = true, int T = 1024, float learning_rate = 1e-3,
         const std::string &backup_dir = "bots/bot-bc-mgda/backup/agent_backup")
        : training(training), T(T), learning_rate(learning_rate), backup_dir(backup_dir) {

#if defined(DISTRIBUTED_LEARNING)
        this->backup_dir = backup_dir = "bots/bot-bc-mgda/server_checkpoint";
        std::cout << "=== DISTRIBUTED LEARNING MODE ===" << std::endl;
        client = std::make_unique<AgentClient>(backup_dir);
        
        // Load checkpoint from server
        if (std::filesystem::exists(backup_dir + "/checkpoint.pt")) {
            std::vector<torch::Tensor> checkpoint;
            torch::load(checkpoint, backup_dir + "/checkpoint.pt");
            
            model = AgentModel();
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
        while (getch() != ' ');
    #endif

        model = AgentModel();
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
        log("LAYER_INDEX=" + std::to_string(LAYER_INDEX));

#if defined(SLOWMOTION)
        log("SLOWMOTION");
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
        model->forward(dummy);
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
        if (getch() == 'y') {
            std::cout << "Submitting..." << std::endl;
            zip_and_return_backup(backup_path);
            std::cout << "Done! Press space" << std::endl;
            while (getch() != ' ');
        }
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
        auto output = model->forward(state);

        
        values.push_back(output[1]);
        log_probs.push_back(torch::log(output[0]));
        
        std::vector<float> v;
        for (int i = 0; i < num_actions; ++i)
            v.push_back(output[0][i].item<float>());
        
        return max_element(v.begin(), v.end()) - v.begin();
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T_initial)
            return;
        
        if (!manual)
            return;
        
        auto one_hot = torch::zeros({num_actions});
        one_hot[action] += 1;
        model->update_actions(one_hot);
        actions.push_back(action);
        
        if (actions.size() == T) {
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
                manual = training;
                if (manual) {
                    std::cout << "manual part! press space button to continue" << std::endl;
                    while(getch() != ' ');
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
    bool is_training = false, logging = true, training, done_training = false, manual = false;
    std::thread trainThread;
    float learning_rate;
    int T, cnt = 0, cnt_warm_up = 0, T_initial = 512, T_warm_up = 100;
    const int num_actions = 9, num_channels = 32, grid_x = 31, grid_y = 31, hidden_size = 160;
    std::string backup_dir;
    AgentModel model{nullptr};
    
#if !defined(DISTRIBUTED_LEARNING)
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};
#else
    std::unique_ptr<AgentClient> client{nullptr};
#endif
    
    std::vector<torch::Tensor> states, log_probs, values, rewards;
    std::vector<int> actions;
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
        for (int t = 0; t < T; ++t)
            ++counts[actions[t]];

        for (int g = 0; g < groups.size(); ++g)
            for (auto a: groups[g])
                group_counts[g] += counts[a];

        // Build per-group average loss tensors (differentiable)
        std::vector<torch::Tensor> per_group_losses(groups.size());
        auto H = torch::zeros({1}), b_loss = torch::zeros({1});
        
        for (int t = 0; t < T; ++t) {
            H -= (log_probs[t] * torch::exp(log_probs[t])).sum();
            b_loss -= log_probs[t][actions[t]];
        }
        
        b_loss /= T, H /= T;
        b_loss = b_loss.detach(), H = H.detach();
        auto loss = b_loss - 0.05 * H;

        std::vector<std::vector<torch::Tensor>> group_grads;  // per active group

        for (int g = 0; g < groups.size(); ++g) {
            per_group_losses[g] = torch::zeros({1});

            if (group_counts[g] == 0)
                continue;
            
            for (int t = 0; t < T; ++t)
                for (auto a: groups[g])
                    if (actions[t] == a)
                        per_group_losses[g] -= log_probs[t][a];
            
            per_group_losses[g] /= group_counts[g];
            
            //std::cout << "WE DID IT 1" << std::endl;

            model->zero_grad();
            per_group_losses[g].backward();

            //std::cout << "WE DID IT 2" << std::endl;

            std::vector<torch::Tensor> grads;
                
            for (auto& p : model->parameters())
                if (p.grad().defined())
                    grads.push_back(p.grad().detach().clone());
                else
                    grads.push_back(torch::zeros_like(p));
                
            group_grads.push_back(grads);
            
            //std::cout << "WE DID IT 3" << std::endl;

            log_probs.clear();
            values.clear();
            rewards.clear();
            model->reset_memory();

            if (g + 1 == groups.size())
                continue;
            
            for (int t = 0; t < T; ++t) {
                auto output = model->forward(states[t]);
                log_probs.push_back(torch::log(output[0]));
                auto one_hot = torch::zeros({num_actions});
                one_hot[actions[t]] += 1;
                model->update_actions(one_hot);
            }
            //std::cout << "WE DID IT 4" << std::endl;
        }
    
        //std::cout << "WE DID IT 5" << std::endl;
        
        model->zero_grad();

        //std::cout << "WE DID IT 7" << std::endl;

        // Apply optimizer step
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
            client->send_gradient(_group_grads, group_counts);
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
            
            // MGDA on active group objectives
            auto alphas = solve_mgda(group_grads);
        
            // Combine action gradients using MGDA weights
            auto params = model->parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                auto combined = torch::zeros_like(params[p]);
                int idx = 0;
                for (int g = 0; g < groups.size(); ++g)
                    if (group_counts[g] > 0) {
                        combined += alphas[idx] * group_grads[idx][p];
                        idx++;
                    }
                if (params[p].grad().defined())
                    params[p].mutable_grad() = combined;
                else
                    params[p].mutable_grad() = combined.clone().detach();
            }
            optimizer->step();
            //std::cout << "WE DID IT 9" << std::endl;

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
            "]\nMGDA_alphas=[" +
            [&]{
                std::string s;
                for (size_t i=0; i< alphas.size(); ++i) {
                    if (i) s += ", ";
                    s += std::to_string(alphas[i]);
                }
                return s;
            }() +
            "]");
#endif
        }
        
        else {
            auto alphas = solve_mgda(group_grads);
            
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
            "]\nMGDA_alphas=[" +
            [&]{
                std::string s;
                for (size_t i=0; i< alphas.size(); ++i) {
                    if (i) s += ", ";
                    s += std::to_string(alphas[i]);
                }
                return s;
            }() +
            "]");
        }

        // Cleanup
        actions.clear();
        rewards.clear();
        log_probs.clear();
        states.clear();
        values.clear();
        model->reset_memory();
        done_training = true;
    }

    std::vector<float> solve_mgda(const std::vector<std::vector<torch::Tensor>>& grad_list) {
        int K = grad_list.size();
        if (K == 0) return {};
        if (K == 1) return {1.0f};

        std::vector<std::vector<double>> M(K, std::vector<double>(K, 0.0));
        for (int i = 0; i < K; ++i)
            for (int j = i; j < K; ++j) {
                double dot = 0.0;
                for (size_t p = 0; p < grad_list[i].size(); ++p)
                    dot += (grad_list[i][p] * grad_list[j][p]).sum().item<double>();
                M[i][j] = M[j][i] = dot;
            }

        auto solve_linear = [](std::vector<std::vector<double>> A, std::vector<double> b) -> std::vector<double> {
            int n = A.size();
            for (int i = 0; i < n; ++i) {
                int pivot = i;
                for (int row = i + 1; row < n; ++row)
                    if (std::abs(A[row][i]) > std::abs(A[pivot][i])) pivot = row;
                if (std::abs(A[pivot][i]) < 1e-12) continue;
                std::swap(A[i], A[pivot]);
                std::swap(b[i], b[pivot]);

                double div = A[i][i];
                for (int j = i; j < n; ++j) A[i][j] /= div;
                b[i] /= div;

                for (int row = 0; row < n; ++row)
                    if (row != i) {
                        double factor = A[row][i];
                        if (std::abs(factor) > 1e-12) {
                            for (int j = i; j < n; ++j) 
                                A[row][j] -= factor * A[i][j];
                            b[row] -= factor * b[i];
                        }
                    }
            }
                return b;
        };

        double best_val = std::numeric_limits<double>::max();
        std::vector<double> best_alpha(K, 0.0);

        for (int mask = 1; mask < (1 << K); ++mask) {
            std::vector<int> S;
            for (int i = 0; i < K; ++i)
                if (mask & (1 << i)) S.push_back(i);
            int m = S.size();

            int n = m + 1;
            std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
            std::vector<double> b(n, 0.0);
            for (int r = 0; r < m; ++r) {
                for (int c = 0; c < m; ++c)
                    A[r][c] = M[S[r]][S[c]];
                A[r][m] = A[m][r] = 1.0;
            }
            A[m][m] = 0.0;
            b[m] = 1.0;

            auto sol = solve_linear(A, b);
            if (sol.empty())
                continue;

            std::vector<double> alpha_S(sol.begin(), sol.begin() + m);

            bool feasible = true;
            for (int i = 0; i < m; ++i)
                if (alpha_S[i] < -1e-10) { feasible = false; break; }
            if (!feasible) continue;

            double val = 0.0;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    val += alpha_S[i] * M[S[i]][S[j]] * alpha_S[j];
            val *= 0.5;

            if (val < best_val) {
                best_val = val;
                std::fill(best_alpha.begin(), best_alpha.end(), 0.0);
                for (int i = 0; i < m; ++i)
                    best_alpha[S[i]] = alpha_S[i];
            }
        }

        std::vector<float> result(K);
        for (int i = 0; i < K; ++i)
            result[i] = static_cast<float>(best_alpha[i]);
        return result;
    }
};