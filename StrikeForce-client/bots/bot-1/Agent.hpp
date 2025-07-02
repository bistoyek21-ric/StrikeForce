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
#include "Network.hpp"
#include <torch/torch.h>
#include <random>
#include <filesystem>

//g++ -std=c++17 main.cpp -o app -ltorch -ltorch_cpu -lc10 -lsfml-graphics -lsfml-window -lsfml-system

class Agent {
private:
    Network* net;

    bool training, online;
    int hidden_size, num_actions, T;
    double gamma, learning_rate;
    std::string backup_dir;
    std::vector<int> action;
    std::vector<double> rewards;
    torch::Tensor H, C, O, b, D, d;
    std::vector<torch::Tensor> o, c, h, log_probs;

    void saveProgress(){
        puts("checking dir ...");
        if(!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        puts("saving parameters ...");
        torch::save(H, backup_dir + "/H.pt");
        torch::save(C, backup_dir + "/C.pt");
        torch::save(O, backup_dir + "/O.pt");
        torch::save(b, backup_dir + "/b.pt");
        torch::save(D, backup_dir + "/D.pt");
        torch::save(d, backup_dir + "/d.pt");
        puts("saved!");
    }

public:
    Agent(bool online, bool training, int hidden_size, int num_actions, int T, double gamma, 
                 double learning_rate, const std::string& backup_dir = "bots/bot-1/agent_backup")
        : training(training), 
          hidden_size(hidden_size), 
          num_actions(num_actions), 
          T(T), 
          gamma(gamma), 
          learning_rate(learning_rate),
          backup_dir(backup_dir){
        torch::set_default_dtype(caffe2::TypeMeta::Make<double>());
        if(std::filesystem::exists(backup_dir)){
            try{
                torch::load(H, backup_dir + "/H.pt");
                torch::load(C, backup_dir + "/C.pt");
                torch::load(O, backup_dir + "/O.pt");
                torch::load(b, backup_dir + "/b.pt");
                torch::load(D, backup_dir + "/D.pt");
                torch::load(d, backup_dir + "/d.pt");
            }
            catch(const std::exception& e){
                initializeTensors();
            }
        }
        else
            initializeTensors();
        if(h.empty())
            h.push_back(torch::zeros({hidden_size, 1}));
		if(online){
        	net = new Network(hidden_size * sizeof(double));
			std::vector<double> h_vec(h.back().data_ptr<double>(), h.back().data_ptr<double>() + h.back().numel());
        	net->give_h(h_vec);
		}
    }

    ~Agent(){
        saveProgress();
        if(online)
            delete net;
    }

private:
    void initializeTensors(){
        H = torch::randn({hidden_size, hidden_size}) * 0.01;
        C = torch::randn({hidden_size, hidden_size}) * 0.01;
        O = torch::randn({hidden_size, hidden_size}) * 0.01;
        b = torch::zeros({hidden_size, 1});
        D = torch::randn({num_actions, hidden_size}) * 0.01;
        d = torch::zeros({num_actions, 1});
    }

public:
    int predict(const std::vector<double>& obs){
		if(online)
        	c.push_back(torch::tensor(net->get_c()).view({hidden_size, 1}));
		else
			c.push_back(torch::zeros({hidden_size, 1}));
        o.push_back(torch::tensor(obs).view({hidden_size, 1}));
        torch::Tensor u = torch::matmul(H, h.back()) + torch::matmul(C, c.back()) + torch::matmul(O, o.back()) + b;
        h.push_back(torch::tanh(u));
		if(online){
			std::vector<double> h_vec(h.back().data_ptr<double>(), h.back().data_ptr<double>() + h.back().numel());
        	net->give_h(h_vec);
		}
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
        puts("init");
        std::vector<double> G(T + 1);
        torch::Tensor Dt = D.t(), Ht = H.t(), Ct = C.t();
        torch::Tensor pi_e, epsilon, delta = torch::zeros_like(h[0]);
        torch::Tensor grad_H = torch::zeros_like(H);
        torch::Tensor grad_C = torch::zeros_like(C);
        torch::Tensor grad_O = torch::zeros_like(O);
        torch::Tensor grad_b = torch::zeros_like(b);
        torch::Tensor grad_D = torch::zeros_like(D);
        torch::Tensor grad_d = torch::zeros_like(d);
        double L = 0;
        G[T] = 0;
        puts("T-1 => 0");
        for(int t = T - 1; ~t; --t){
            G[t] = rewards[t] + gamma * G[t + 1];
            L -= G[t] * log_probs[t][action[t]].item<double>();
            pi_e = torch::exp(log_probs[t]), pi_e[action[t]] -= 1;
            epsilon = G[t] * pi_e;
            delta = torch::matmul(Dt, epsilon) + torch::matmul(Ht + Ct, delta);
            delta *= (1.0 - torch::square(h[t + 1]));
            grad_d += epsilon;
            grad_D += torch::matmul(epsilon, h[t + 1].t());
            grad_b += delta;
            grad_H += torch::matmul(delta, h[t].t());
            grad_C += torch::matmul(delta, c[t].t());
            grad_O += torch::matmul(delta, o[t].t());
        }
        puts("upd");
        H -= learning_rate * grad_H;
        C -= learning_rate * grad_C;
        O -= learning_rate * grad_O;
        b -= learning_rate * grad_b;
        D -= learning_rate * grad_D;
        d -= learning_rate * grad_d;
        h.clear(), o.clear(), c.clear();
        h.push_back(torch::zeros({hidden_size, 1}));
        log_probs.clear();
        action.clear(), rewards.clear();
        #if defined(__unix__) || defined(__APPLE__)
        disable_input_buffering();
        #endif
    }
};