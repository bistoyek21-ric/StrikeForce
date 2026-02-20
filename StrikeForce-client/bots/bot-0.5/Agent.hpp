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

const std::string bot_code = "bot-0.5", backup_path = "bots/bot-0.5/backup";

class Agent {
public:
    Agent(bool training = true, int T = 1024, int num_epochs = 4, float gamma = 0.99, float learning_rate = 1e-3,
         float ppo_clip = 0.2, float cv = 0.5, const std::string &backup_dir = "bots/bot-0.5/backup/agent_backup")
        : training(training), T(T), num_epochs(num_epochs), gamma(gamma), learning_rate(learning_rate),
        ppo_clip(ppo_clip), cv(cv), backup_dir(backup_dir) {

#if defined(DISTRIBUTED_LEARNING)
        this->backup_dir = backup_dir = "bots/bot-0.5/server_checkpoint";
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
        } else if (actions.empty()) {
            manual = training;
            if (manual) {
                std::cout << "manual part! press space button to continue" << std::endl;
                while(getch() != ' ');
                std::cout << "space button pressed!" << std::endl;
            }
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
    float learning_rate, gamma, ppo_clip, cv;
    int T, num_epochs, cnt = 0, T_initial = 10;
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
        auto loss = torch::zeros({1});
        auto H = torch::zeros({1});
        
        for (size_t i = 0; i < T; ++i) {
            loss -= log_probs[i][actions[i]];
            H -= (log_probs[i] * torch::exp(log_probs[i])).sum();
        }
        loss = (loss + 0.05 * H) / T;
        
        if (training) {
#if defined(DISTRIBUTED_LEARNING)
            // Distributed mode: compute gradients and send to server
            model->zero_grad();
            loss.backward();
            
            std::vector<torch::Tensor> gradients;
            for (auto& p : model->parameters()) {
                if (p.grad().defined()) {
                    gradients.push_back(p.grad().detach().clone());
                } else {
                    gradients.push_back(torch::zeros_like(p));
                }
            }
            
            std::cout << "Sending gradients to server..." << std::endl;
            client->send_gradient(gradients);
            
            std::cout << "Requesting update vector..." << std::endl;
            auto update_vector = client->get_update_vector();
            
            std::cout << "Applying server update..." << std::endl;
            auto params = model->parameters();
            for (size_t i = 0; i < params.size() && i < update_vector.size(); ++i) {
                params[i].data() += update_vector[i];
            }
            
            std::cout << "Model synchronized with server" << std::endl;
#else
            // Local mode: standard gradient descent
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
#endif
        }
        
        log("A: loss=" + std::to_string(loss.item<float>()) +
            ",H=" + std::to_string(H.item<float>()) + 
            ",time(s)=" + std::to_string(time(0) - ts) +
            ",step=" + std::to_string(calc_diff()));
        
        actions.clear();
        rewards.clear();
        log_probs.clear();
        states.clear();
        values.clear();
        model->reset_memory();
        done_training = true;
    }
};