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

class Agent {
private:
    bool manual;
    int cnt = 0;

    std::mutex mtx;
    std::condition_variable cv;
    bool is_training = false, done_training;
    std::thread trainThread;

    bool training, logging = false;
    int hidden_size, num_actions, T, num_epochs, num_channels, grid_x, grid_y;
    float gamma, learning_rate, ppo_clip, alpha;
    
    std::string backup_dir;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<torch::Tensor> log_probs, states, values;

    torch::Tensor state, action_input, h_state, probs;

    torch::Device device;

    torch::nn::GRU gru{nullptr};
    
    torch::nn::Sequential policy_head{nullptr};
    torch::nn::Sequential value_head{nullptr};
    torch::nn::Sequential cnn{nullptr};
    torch::nn::Sequential gate{nullptr};
    torch::nn::Sequential action_processor{nullptr};
    
    torch::nn::LayerNorm gru_norm{nullptr};
    torch::nn::LayerNorm action_norm{nullptr};

    torch::optim::AdamW* optimizer{nullptr};

    std::ofstream log_file;

    RewardNet* reward_net;

    std::vector<torch::Tensor> coor[2], initial;
    std::vector<torch::Tensor> snap_shot(){
        std::vector<torch::Tensor> params;
        for (auto& p : cnn->parameters()) params.push_back(p.detach().clone());
        for (auto& p : gru->parameters()) params.push_back(p.detach().clone());
        for (auto& p : action_processor->parameters()) params.push_back(p.detach().clone());
        for (auto& p : policy_head->parameters()) params.push_back(p.detach().clone());
        for (auto& p : value_head->parameters()) params.push_back(p.detach().clone());
        for (auto& p : gate->parameters()) params.push_back(p.detach().clone());
        for (auto& p : gru_norm->parameters()) params.push_back(p.detach().clone());
        for (auto& p : action_norm->parameters()) params.push_back(p.detach().clone());
        return params;
    }

    void calc_diff(){
        coor[1] = snap_shot();
        double diff = 0;
        for (int i = 0; i < coor[0].size(); ++i)
            diff += (coor[1][i] - coor[0][i]).pow(2).sum().item<float>();
        counter << "step=" << std::sqrt(diff) << "\n";
        coor[0].clear();
        for (auto& p: coor[1])
            coor[0].push_back(p.clone());
        coor[1].clear();
    }

    template<typename Type>
    void log(const Type& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    void saveProgress() {
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_x << " " << grid_y << " " << num_actions << " ";
        dim.close();
        gru->to(torch::kCPU);
        action_processor->to(torch::kCPU);
        policy_head->to(torch::kCPU);
        value_head->to(torch::kCPU);
        cnn->to(torch::kCPU);
        gate->to(torch::kCPU);
        gru_norm->to(torch::kCPU);
        action_norm->to(torch::kCPU);
        torch::save(gru, backup_dir + "/gru.pt");
        torch::save(action_processor, backup_dir + "/action_processor.pt");
        torch::save(policy_head, backup_dir + "/policy_head.pt");
        torch::save(value_head, backup_dir + "/value_head.pt");
        torch::save(cnn, backup_dir + "/cnn.pt");
        torch::save(gate, backup_dir + "/gate.pt");
        torch::save(gru_norm, backup_dir + "/gru_norm.pt");
        torch::save(action_norm, backup_dir + "/action_norm.pt");
        log("Progress saved to " + backup_dir);
    }

    void load_progress() {
        std::ifstream dim(backup_dir + "/dim.txt");
        if (!dim.is_open()) {
            log("Failed to open dim.txt");
            throw std::runtime_error("Failed to open dim.txt");
        }
        dim >> num_channels >> grid_x >> grid_y >> num_actions;
        log("Loaded dimensions: num_channels=" + std::to_string(num_channels) +
            ", grid_x=" + std::to_string(grid_x) +
            ", grid_y=" + std::to_string(grid_y) +
            ", num_actions=" + std::to_string(num_actions));
        dim.close();
        initializeNetwork();
        torch::load(cnn, backup_dir + "/cnn.pt");
        torch::load(gru, backup_dir + "/gru.pt");
        torch::load(action_processor, backup_dir + "/action_processor.pt");
        torch::load(policy_head, backup_dir + "/policy_head.pt");
        torch::load(value_head, backup_dir + "/value_head.pt");
        torch::load(gate, backup_dir + "/gate.pt");
        torch::load(gru_norm, backup_dir + "/gru_norm.pt");
        torch::load(action_norm, backup_dir + "/action_norm.pt");
        log("Progress loaded successfully");
    }

    void initializeNetwork() {
        log("Start:");
        hidden_size = 128;
        log("hidden_size set to " + std::to_string(hidden_size));
        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_size, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_size, hidden_size, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(hidden_size), torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1), torch::nn::Flatten()
        );
        gru = torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(2));
        action_processor = torch::nn::Sequential(
            torch::nn::Linear(num_actions, hidden_size), torch::nn::ReLU(),
            ResidualBlock(hidden_size)
        );
        gate = torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size, hidden_size), torch::nn::ReLU(),
            ResidualBlock(hidden_size)
        );
        policy_head = torch::nn::Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            torch::nn::Linear(hidden_size, num_actions)
        );
        value_head = torch::nn::Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            torch::nn::Linear(hidden_size, 1), torch::nn::Sigmoid()
        );
        gru_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
        action_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    }

    torch::Tensor computeReturns() {
        torch::Tensor returns = torch::zeros({T}, device);
        double running = 0;
        for (int i = T - 1; i >= 0; --i) {
            running = rewards[i] + gamma * running;
            returns[i] = running;
        }
        return returns;
    }

    std::vector<torch::Tensor> forward(const torch::Tensor &state) {
        auto feat = cnn->forward(state);
        feat = feat.view({1, 1, -1});
        auto r = gru->forward(feat, h_state);
        auto out_seq = std::get<0>(r).view({-1});
        out_seq = gru_norm->forward(out_seq);
        h_state = std::get<1>(r).detach();
        auto a_feat = action_processor->forward(action_input);
        a_feat = action_norm->forward(a_feat.view({-1}));
        auto gate_input = torch::cat({out_seq, a_feat});
        auto gated = gate->forward(gate_input);
        auto logits = policy_head->forward(gated);
        auto p = torch::softmax(logits, -1);
        auto v = 2 * value_head->forward(gated).squeeze() - 1;
        return {p, v};
    }

    void train() {
        time_t ts = time(0);
        auto returns = computeReturns();
        std::vector<torch::Tensor> advantages;
        for (int i = 0; i < T; ++i)
            advantages.push_back(returns[i] - values[i]);
        torch::Tensor adv_tensor = torch::stack(advantages);
        torch::Tensor adv_mean = adv_tensor.mean();
        torch::Tensor adv_std = adv_tensor.std();
        if (adv_std.item<float>() < 1e-8)
            adv_tensor = adv_tensor - adv_mean;
        else
            adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-5);
        float loss_g;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            torch::Tensor p_loss = torch::zeros({}, device);
            torch::Tensor v_loss = torch::zeros({}, device);
            h_state = torch::zeros({2, 1, hidden_size}, device);
            action_input = torch::zeros({num_actions}, device);
            for (int i = 0; i < T; ++i) {
                auto output = forward(states[i]);
                action_input = action_input * alpha;
                action_input[actions[i]] += 1 - alpha;
                auto current_logp = torch::log(output[0])[actions[i]];
                auto diff = torch::clamp(current_logp - log_probs[i], -100, 10);
                auto ratio = torch::exp(diff);
                auto clipped = torch::clamp(ratio, 1 - ppo_clip, 1 + ppo_clip);
                p_loss -= torch::min(ratio * adv_tensor[i], clipped * adv_tensor[i]);
                v_loss += torch::mse_loss(output[1], returns[i]);
            }
            auto loss = (p_loss + 0.25 * v_loss) / T;
            log("Epoch " + std::to_string(epoch) + ": loss=" + std::to_string(loss.item<float>()));
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            log("*");
            loss_g = loss.item<float>();
        }
        log("Training completed");
        actions.clear();
        rewards.clear();
        log_probs.clear();
        states.clear();
        values.clear();
        counter << "A: loss=" << loss_g << ", time(s)=" << time(0) - ts << ",";
        calc_diff();
        counter.flush();
        done_training = true;
    }

public:
    Agent(bool _training, int _T, int _num_epochs, float _gamma, float _learning_rate, float _ppo_clip,
          float _alpha, int _num_channels = 33, int _grid_x = 15, int _grid_y = 49, int _num_actions = 10)
        : training(_training), T(_T), num_epochs(_num_epochs), gamma(_gamma), learning_rate(_learning_rate),
          ppo_clip(_ppo_clip), alpha(_alpha), device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          num_channels(_num_channels), grid_x(_grid_x), grid_y(_grid_y), num_actions(_num_actions) {
        backup_dir = backup_path + "/agent_backup";
#if defined(CROWDSOURCED_TRAINING)
        std::cout << "loading backup ..." << std::endl;
        request_and_extract_backup(backup_path, bot_code);
        std::cout << "done!" << std::endl;
        std::cout << "press space to continue" << std::endl;
        while (getch() != ' ');
#endif
        if (std::filesystem::exists(backup_dir)) {
            log_file.open(backup_dir + "/agent_log.log");
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        }
        else {
            std::filesystem::create_directories(backup_dir);
            log_file.open(backup_dir + "/agent_log.log");
            initializeNetwork();
        }
        log(std::to_string(torch::cuda::is_available()));
        cnn->to(device);
        gru->to(device);
        action_processor->to(device);
        policy_head->to(device);
        value_head->to(device);
        gate->to(device);
        gru_norm->to(device);
        action_norm->to(device);
        std::vector<torch::Tensor> params;
        auto cnn_params = cnn->parameters();
        auto gru_params = gru->parameters();
        auto action_params = action_processor->parameters();
        auto policy_params = policy_head->parameters();
        auto value_params = value_head->parameters();
        auto gate_params = gate->parameters();
        auto gru_norm_params = gru_norm->parameters();
        auto action_norm_params = action_norm->parameters();
        params.insert(params.end(), cnn_params.begin(), cnn_params.end());
        params.insert(params.end(), gru_params.begin(), gru_params.end());
        params.insert(params.end(), action_params.begin(), action_params.end());
        params.insert(params.end(), policy_params.begin(), policy_params.end());
        params.insert(params.end(), value_params.begin(), value_params.end());
        params.insert(params.end(), gate_params.begin(), gate_params.end());
        params.insert(params.end(), gru_norm_params.begin(), gru_norm_params.end());
        params.insert(params.end(), action_norm_params.begin(), action_norm_params.end());
        optimizer = new torch::optim::AdamW(params, torch::optim::AdamWOptions().lr(learning_rate).weight_decay(1e-4));
        reward_net = new RewardNet(true, T, learning_rate, alpha, backup_path + "/reward_backup", num_actions, num_channels, grid_x, grid_y);
        log("--------------------------------------------------------");
        auto t = time(nullptr);
        log(std::string(ctime(&t)));
        log("--------------------------------------------------------");
        log("Agent initialized with T=" + std::to_string(T) + ", num_epochs=" + std::to_string(num_epochs) +
            ", gamma=" + std::to_string(gamma) + ", alpha=" + std::to_string(alpha) + ", learning_rate=" + std::to_string(learning_rate) +
            ", ppo_clip=" + std::to_string(ppo_clip));
        h_state = torch::zeros({2, 1, hidden_size}, device);
        action_input = torch::zeros({num_actions}, device);
#if !defined(CROWDSOURCED_TRAINING)
        cnt = T + 1;
#endif
        coor[0] = snap_shot();
        for (auto &p: coor[0])
            initial.push_back(p.clone());
    }

    ~Agent() {
        if (is_training)
            if (trainThread.joinable()) {
                std::cout << "Agent Network is updating...\nthis might take a few seconds" << std::endl;
                trainThread.join();
                std::cout << "done!" << std::endl;
            }
        delete optimizer;
        delete reward_net;
        coor[0].clear();
        for (auto &p: initial)
            coor[0].push_back(p.clone());
        counter << "A total dist: ";
        calc_diff();
        counter << "====\n";
        counter.flush();
        if (training)
            saveProgress();
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
        log_file.close();
    }

    int predict(const std::vector<float>& obs) {
        if (cnt <= T)
            return 0;
        if (is_training) {
#if !defined(CROWDSOURCED_TRAINING)
            if (done_training) {
                is_training = false;
                cv.notify_all();
                if (trainThread.joinable())
                    trainThread.join();
                h_state = torch::zeros({2, 1, hidden_size}, device);
                action_input = torch::zeros({num_actions}, device);
            }
            else
               return 0;
#else
            return 0;
#endif
        }
        auto state = torch::tensor(obs, torch::dtype(torch::kFloat32).device(device)).view({1, num_channels, grid_x, grid_y});
        states.push_back(state.clone());
        auto output = forward(state);
        values.push_back(output[1].detach().clone());
        probs = output[0];
        std::vector<float> v(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        return dist(gen);
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T)
            return;
        rewards.push_back(reward_net->get_reward(action, imitate, states.back()));
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        if (rewards.back() == -2 && training) {
            rewards.clear();
            states.clear();
            values.clear();
            return;
        }
        actions.push_back(action);
        log_probs.push_back(torch::log(probs.detach().clone()[action] + 1e-8));
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
                cv.notify_all();
                if (trainThread.joinable())
                    trainThread.join();
                h_state = torch::zeros({2, 1, hidden_size}, device);
                action_input = torch::zeros({num_actions}, device);
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
        if (actions.size() == T / 2)
            manual = !manual;
        return manual;
    }
#endif

    bool in_training(){
        return is_training;
    }
};