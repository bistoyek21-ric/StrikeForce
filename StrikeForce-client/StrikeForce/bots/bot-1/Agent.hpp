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
#include "Item.hpp"
#include <torch/torch.h>
#include <random>
#include <filesystem>

//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -lc10 -lsfml-graphics -lsfml-window -lsfml-system

class Agent {
private:
    torch::Device device;

    bool training, online;
    int hidden_size, num_actions, T;
    double gamma, learning_rate;
    std::string backup_dir;
    std::vector<int> action;
    std::vector<double> rewards;
    torch::Tensor H, O, b, D, d;
    std::vector<torch::Tensor> o, h, log_probs;

    void saveProgress(){
        std::cerr << "checking dir ..." << std::endl;
        if(!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::cerr << "saving parameters ..." << std::endl;
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << hidden_size << " " << num_actions;
        dim.close();
        torch::save(H.to(torch::kCPU), backup_dir + "/H.pt");
        torch::save(O.to(torch::kCPU), backup_dir + "/O.pt");
        torch::save(b.to(torch::kCPU), backup_dir + "/b.pt");
        torch::save(D.to(torch::kCPU), backup_dir + "/D.pt");
        torch::save(d.to(torch::kCPU), backup_dir + "/d.pt");
        std::cerr << "saved!" << std::endl;
    }

    void initializeTensors(int hidden_size, int num_actions){
        this->hidden_size = hidden_size;
        this->num_actions = num_actions;
        H = torch::randn({hidden_size, hidden_size}, device) * 0.01;
        O = torch::randn({hidden_size, hidden_size}) * 0.01;
        b = torch::zeros({hidden_size, 1}, device);
        D = torch::randn({num_actions, hidden_size}, device) * 0.01;
        d = torch::zeros({num_actions, 1}, device);
    }

public:
    Agent(bool online, bool training, int T, double gamma, 
        double learning_rate, int hidden_size = 1, int num_actions = 1,
        const std::string& backup_dir = "bots/bot-1/agent_backup")
            : training(training), T(T),
            gamma(gamma), learning_rate(learning_rate),
            backup_dir(backup_dir),
            device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
        torch::set_default_dtype(caffe2::TypeMeta::Make<double>());
        if(std::filesystem::exists(backup_dir)){
            try{
                std::ifstream dim(backup_dir + "/dim.txt");
                dim >> this->hidden_size >> this->num_actions;
                dim.close();
                torch::load(H, backup_dir + "/H.pt"); H = H.to(device);
                torch::load(O, backup_dir + "/O.pt"); O = O.to(device);
                torch::load(b, backup_dir + "/b.pt"); b = b.to(device);
                torch::load(D, backup_dir + "/D.pt"); D = D.to(device);
                torch::load(d, backup_dir + "/d.pt"); d = d.to(device);
            }
            catch(const std::exception& e){
                initializeTensors(hidden_size, num_actions);
            }
        }
        else
            initializeTensors(hidden_size, num_actions);
        h.push_back(torch::zeros({hidden_size, 1}, device));
    }

    ~Agent(){
        saveProgress();
    }
    
    int predict(const std::vector<double>& obs){
        o.push_back(torch::tensor(obs, device).view({hidden_size, 1}));
        torch::Tensor u = torch::matmul(H, h.back());
        u += torch::matmul(O, o.back()) + b;
        h.push_back(torch::tanh(u));
        torch::Tensor logits = torch::matmul(D, h.back()) + d;
        torch::Tensor probs = torch::softmax(logits, 0);
        log_probs.push_back(torch::log(probs + 1e-10));
        std::vector<double> probs_vec(probs.data_ptr<double>(), probs.data_ptr<double>() + probs.numel());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs_vec.begin(), probs_vec.end());
        int a_t = dist(gen);
        return a_t;
    }

    void update(int a_t){
        action.push_back(a_t);
        rewards.push_back(1);
        if(rewards.size() == T && training)
            train();
    }

    void train(){
        #if defined(__unix__) || defined(__APPLE__)
        restore_input_buffering();
        #endif
        std::cerr << "time to train the parameters" << std::endl;
        std::vector<double> G(T + 1);
        torch::Tensor Dt = D.t(), Ht = H.t();
        torch::Tensor pi_e, epsilon, delta = torch::zeros_like(h[0]);
        torch::Tensor grad_H = torch::zeros_like(H);
        torch::Tensor grad_O = torch::zeros_like(O);
        torch::Tensor grad_b = torch::zeros_like(b);
        torch::Tensor grad_D = torch::zeros_like(D);
        torch::Tensor grad_d = torch::zeros_like(d);
        double L = 0;
        G[T] = 0;
        std::cerr << "updating gradiants\n--------" << std::endl;
        #if defined(__unix__) || defined(__APPLE__)
        disable_input_buffering();
        #endif
        for(int t = T - 1; ~t; --t){
            G[t] = rewards[t] + gamma * G[t + 1];
            L -= G[t] * log_probs[t][action[t]].item<double>();
            pi_e = torch::exp(log_probs[t]), pi_e[action[t]] -= 1;
            epsilon = G[t] * pi_e;
            delta = torch::matmul(Dt, epsilon) + torch::matmul(Ht, delta);
            delta *= (1.0 - torch::square(h[t + 1]));
            grad_d += epsilon;
            grad_D += torch::matmul(epsilon, h[t + 1].t());
            grad_b += delta;
            grad_H += torch::matmul(delta, h[t].t());
            grad_O += torch::matmul(delta, o[t].t());
        }
        H -= learning_rate * grad_H;
        O -= learning_rate * grad_O;
        b -= learning_rate * grad_b;
        D -= learning_rate * grad_D;
        d -= learning_rate * grad_d;
        h.clear(), o.clear();
        h.push_back(torch::zeros({hidden_size, 1}, device));
        log_probs.clear();
        action.clear(), rewards.clear();
    }
};