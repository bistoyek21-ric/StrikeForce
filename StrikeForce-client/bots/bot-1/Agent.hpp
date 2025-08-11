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

const std::string SERVER_URL = "http://bistoyek21.org";
const std::string bot_code = "bot-1";

// Function to escape paths for system commands (handles spaces and special chars)
std::string escape_path(const std::string& path) {
    std::string escaped;
    for (char c : path) {
        if (std::isspace(c) || c == '"' || c == '\\') {
            escaped += '\\';
        }
        escaped += c;
    }
    return "\"" + escaped + "\"";
}

// Function to request and extract backup: delete dir if exists, download backup.zip, extract to dir
int request_and_extract_backup(const std::string& dir) {
    // Sanitize inputs (basic validation)
    if (bot_code.empty() || dir.empty()) {
        std::cerr << "Error: bot_code or dir cannot be empty" << std::endl;
        return 1;
    }
    // Normalize path
    fs::path dir_path = dir;
    // Delete dir if exists
    if (fs::exists(dir_path)) {
        try {
            fs::remove_all(dir_path);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Failed to delete directory " << dir << ": " << e.what() << std::endl;
            return 1;
        }
    }
    // Download backup.zip using curl
    std::string download_cmd = "curl -o backup.zip \"" + SERVER_URL + "/StrikeForce/api/request_backup?code=" + bot_code + "\"";
    if (system(download_cmd.c_str()) != 0) {
        std::cerr << "Failed to download backup.zip for bot_code: " << bot_code << std::endl;
        return 1;
    }
    // Create dir
    try {
        fs::create_directories(dir_path);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Failed to create directory " << dir << ": " << e.what() << std::endl;
        return 1;
    }
    // Extract backup.zip to dir using 7z
    std::string extract_cmd = "7z x -y -o" + escape_path(dir) + " backup.zip";
    if (system(extract_cmd.c_str()) != 0) {
        std::cerr << "Failed to extract backup.zip to " << dir << std::endl;
        return 1;
    }
    // Delete the downloaded zip
    try {
        fs::remove("backup.zip");
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Failed to delete backup.zip: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Backup requested and extracted to " << dir << std::endl;
    return 0;
}

// Function to zip and return backup: zip dir, send to server, delete zip
int zip_and_return_backup(const std::string& dir) {
    // Sanitize input
    if (dir.empty()) {
        std::cerr << "Error: dir cannot be empty" << std::endl;
        return 1;
    }
    // Normalize path
    fs::path dir_path = dir;
    if (!fs::exists(dir_path)) {
        std::cerr << "Error: Directory " << dir << " does not exist" << std::endl;
        return 1;
    }
    // Generate zip name
    std::string zip_name = dir_path.filename().string() + ".zip";
    // Zip the directory using 7z
    std::string zip_cmd = "7z a -tzip " + escape_path(zip_name) + " " + escape_path(dir) + "/*";
    if (system(zip_cmd.c_str()) != 0) {
        std::cerr << "Failed to zip directory: " << dir << std::endl;
        return 1;
    }
    // Send the zip using curl
    std::string send_cmd = "curl -X POST --data-binary @" + escape_path(zip_name) + " \"" + SERVER_URL + "/StrikeForce/api/return_backup\"";
    if (system(send_cmd.c_str()) != 0) {
        std::cerr << "Failed to send zip Golgi file to server" << std::endl;
        // Continue to delete zip
    }
    // Delete the zip
    try {
        fs::remove(zip_name);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Failed to delete zip: " << zip_name << ": " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Backup zipped and returned from " << dir << std::endl;
    return 0;
}

class Agent {
private:
    bool manual;
    int cnt = 0;

    std::mutex mtx;
    std::condition_variable cv;
    bool is_training = false, not_training;
    std::thread trainThread;

    bool training, logging = false;
    int hidden_size, num_actions, T, num_epochs, num_channels, grid_size;
    float gamma, learning_rate, ppo_clip, alpha, c_v;
    
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

    template<typename T>
    void log(const T& message) {
        if (!logging)
            return;
        log_file << message << std::endl;
        log_file.flush();
    }

    void saveProgress() {
        if (backup_dir.empty())
            return;
        if (!std::filesystem::exists(backup_dir))
            std::filesystem::create_directories(backup_dir);
        std::ofstream dim(backup_dir + "/dim.txt");
        dim << num_channels << " " << grid_size << " " << num_actions << " ";
        dim.close();
        std::ofstream coef(backup_dir + "/c_v.txt");
        coef << c_v;
        coef.close();
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
        dim >> num_channels >> grid_size >> num_actions;
        log("Loaded dimensions: num_channels=" + std::to_string(num_channels) +
            ", grid_size=" + std::to_string(grid_size) +
            ", num_actions=" + std::to_string(num_actions));
        dim.close();
        std::ifstream coef(backup_dir + "/c_v.txt");
        coef >> c_v;
        coef.close();
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
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 32, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, hidden_size, 3).stride(1).padding(1)),
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
            torch::nn::Linear(hidden_size, num_actions), torch::nn::ReLU()
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
        torch::Tensor returns = torch::zeros({(int)rewards.size()}, device);
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
                auto adv = adv_tensor[i];
                p_loss -= torch::min(ratio * adv, clipped * adv);
                v_loss += torch::mse_loss(output[1], returns[i]);
            }
            v_loss /= T;
            auto loss = p_loss + c_v * v_loss;
            log("Epoch " + std::to_string(epoch) + ": loss=" + std::to_string(loss.item<float>()));
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            log("*");
        }
        log("Training completed");
        actions.clear();
        rewards.clear();
        log_probs.clear();
        states.clear();
        values.clear();
        not_training = true;
    }

public:
    Agent(bool _training, int _T, int _num_epochs, float _gamma, float _learning_rate, float _ppo_clip,
          float _alpha, const std::string& _backup_dir = "bots/bot-1/backup/agent_backup", int _num_channels = 1,
          int _grid_size = 1, int _num_actions = 1)
        : training(_training), T(_T), num_epochs(_num_epochs), gamma(_gamma), learning_rate(_learning_rate),
          ppo_clip(_ppo_clip), alpha(_alpha), backup_dir(_backup_dir),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          num_channels(_num_channels), grid_size(_grid_size), num_actions(_num_actions){
        c_v = 0.05;
#if defined(COWRDSOURCED_TRAINING)
        request_and_extract_backup(backup_dir + "/..");
#endif
        if (!backup_dir.empty() && std::filesystem::exists(backup_dir)) {
            try {
                load_progress();
            } catch (const std::exception& e) {
                log("Failed to load progress: " + std::string(e.what()));
                initializeNetwork();
            }
        }
        else {
            if (!backup_dir.empty())
                std::filesystem::create_directories(backup_dir);
            initializeNetwork();
        }
        log_file.open(backup_dir + "/agent_log.log");
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
        reward_net = new RewardNet(true, T, learning_rate, alpha, backup_dir + "/../reward_backup", num_actions, num_channels, grid_size);
        log("--------------------------------------------------------");
        auto t = time(nullptr);
        log(std::string(ctime(&t)));
        log("--------------------------------------------------------");
        log("Agent initialized with T=" + std::to_string(T) + ", num_epochs=" + std::to_string(num_epochs) +
            ", gamma=" + std::to_string(gamma) + ", alpha=" + std::to_string(alpha) + ", learning_rate=" + std::to_string(learning_rate) +
            ", ppo_clip=" + std::to_string(ppo_clip));
        h_state = torch::zeros({2, 1, hidden_size}, device);
        action_input = torch::zeros({num_actions}, device);
#if !defined(COWRDSOURCED_TRAINING)
        cnt = T + 1;
#endif
    }

    ~Agent() {
        restore_input_buffering();
        if (is_training)
            if (trainThread.joinable()) {
                std::cout << "Agent Network is updating...\nthis might take a few seconds" << std::endl;
                trainThread.join();
                std::cout << "done!" << std::endl;
            }
        if (training)
            saveProgress();
        delete optimizer;
        delete reward_net;
#if defined(COWRDSOURCED_TRAINING)
        std::cout << "sending backup to server" << std::endl;
        zip_and_return_backup(backup_dir + "/..");
        std::cout << "done!" << std::endl;
#endif
        log_file.close();
        disable_input_buffering();
    }

    int predict(const std::vector<float>& obs) {
        if (cnt <= T)
            return 0;
        if (is_training) {
#if !defined(COWRDSOURCED_TRAINING)
            if (not_training) {
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
        auto state = torch::tensor(obs, torch::dtype(torch::kFloat32).device(device)).view({1, num_channels, grid_size, grid_size});
        states.push_back(state);
        auto output = forward(state);
        values.push_back(output[1].detach());
        probs = output[0];
        std::vector<float> v(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(v.begin(), v.end());
        return dist(gen);
    }

    void update(int action, bool imitate) {
        if (is_training || cnt <= T)
            return;
        actions.push_back(action);
        log_probs.push_back(torch::log(probs.detach()[action] + 1e-8));
        rewards.push_back(reward_net->get_reward(action, imitate, states.back()));
        action_input = action_input * alpha;
        action_input[action] += 1 - alpha;
        if (actions.size() == T) {
            if (training) {
                is_training = true;
                not_training = false;
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

#if defined(COWRDSOURCED_TRAINING)
    bool is_manual() {
        if (!is_training && cnt <= T)
            ++cnt;
        if (is_training) {
            if (not_training) {
                is_training = false;
                cv.notify_all();
                if (trainThread.joinable())
                    trainThread.join();
                h_state = torch::zeros({2, 1, hidden_size}, device);
                action_input = torch::zeros({num_actions}, device);
            }
            else
               return true;
        }
        if (cnt <= T)
            manual = true;
        else if (actions.size() == T / 2)
            manual = !manual;
        return manual;
    }
#endif
};