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
//g++ -std=c++17 AgentServer.cpp -o app_agent_server -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lpthread
// Add -lws2_32 on Windows
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sstream>
#include <set>
#include <string>

#include "../Modules.hpp"

#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close

const int BUFFER_SIZE = 2048;

// Global synchronization
std::mutex mtx;
std::condition_variable cv_update_ready;
std::atomic<bool> server_running{true};
std::atomic<int> update_version{0};

struct ClientHandler {
    SOCKET sock;
    int client_id;
    std::vector<torch::Tensor> gradients;
    bool gradient_ready = false;
    bool is_active = true;
    std::chrono::steady_clock::time_point last_seen;
    int last_update_version = -1;
    
    ClientHandler(SOCKET s, int id) 
        : sock(s), client_id(id), gradient_ready(false), is_active(true),
          last_seen(std::chrono::steady_clock::now()), last_update_version(-1) {}
};

class DynamicAgentServer {
public:
    DynamicAgentServer(int port, const std::string& password, int max_clients, 
                       const std::string& checkpoint_path, double lr = 1e-3,
                       int min_clients_for_update = 1)
        : port(port), password(password), max_clients(max_clients), 
          checkpoint_path(checkpoint_path), min_clients_for_update(min_clients_for_update) {
        
        // Load or initialize checkpoint
        std::string model_path = checkpoint_path + "/model.pt";
        model = AgentModel();
        if (std::filesystem::exists(model_path)) {
            std::cout << "Loading checkpoint from: " << model_path << std::endl;
            torch::load(model, model_path);
            std::cout << "Loaded" << std::endl;
        } else {
            std::cerr << "No checkpoint found at " << model_path << std::endl;
            std::cerr << "Initializing..." << std::endl;
            std::filesystem::create_directories(checkpoint_path);
            std::cerr << "done!" << std::endl;
        }
        
        // Setup optimizer
        optimizer = std::make_unique<torch::optim::AdamW>(
            model->parameters(), 
            torch::optim::AdamWOptions(lr)
        );

        std::string optim_path = checkpoint_path + "/optimizer.pt";
        if (std::filesystem::exists(optim_path)) {
            try {
                torch::load(*optimizer, optim_path);
                std::cout << "Loaded optimizer state" << std::endl;
            } catch (...) {
                std::cout << "Failed to load optimizer state, using fresh" << std::endl;
            }
        }
        
        std::cout << "Server initialized:" << std::endl;
        std::cout << "  Max clients: " << max_clients << std::endl;
        std::cout << "  Min clients for update: " << min_clients_for_update << std::endl;
        std::cout << "  Learning rate: " << lr << std::endl;
    }

    void start() {
#if !defined(__unix__) && !defined(__APPLE__)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return;
        }
#endif

        std::cout << "Is it a global server or local?\n(G:global/any thing else:local)\n";
    	std::string is_global;
    	std::cin >> is_global;
	    std::cout << "Server IP: ";
    	if(is_global == "G"){
	    	std::cout.flush();
		    system("curl -s https://api.ipify.org");
    	}
	    else{
		    char host[256];
    		gethostname(host, sizeof(host));
	    	std::cout << inet_ntoa(*((struct in_addr*)gethostbyname(host)->h_addr_list[0]));
	    }
        std::cout << std::endl;

        SOCKET server_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (server_sock == INVALID_SOCKET) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }

        int opt = 1;
        setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);

        if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
            std::cerr << "Bind failed" << std::endl;
            closesocket(server_sock);
            return;
        }

        if (listen(server_sock, max_clients) == SOCKET_ERROR) {
            std::cerr << "Listen failed" << std::endl;
            closesocket(server_sock);
            return;
        }

        std::cout << "\n=== Dynamic Agent Hub Started ===" << std::endl;
        std::cout << "Listening on port " << port << std::endl;
        std::cout << "Waiting for clients to connect...\n" << std::endl;

        // Start training loop in separate thread
        std::thread training_thread([this]() {
            this->training_loop();
        });

        // Start client cleanup thread
        std::thread cleanup_thread([this]() {
            this->cleanup_inactive_clients();
        });

        // Accept clients continuously
        int next_client_id = 0;
        while (server_running) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                if(kbhit()){
                    if(getch() == '~')
                        server_running = false;
                    if(getch() == '_') {
                        std::cout << "Enter id of the agent you want to eleminate:" << std::endl;
                        try{
                            int id, idx = -1;
                            std::cin >> id;
                            for(int i = 0; i < clients.size(); ++i)
                                if(clients[i]->client_id == id)
                                    idx = i;
                            if(idx != -1) {
                                delete clients[idx];
                                clients.erase(clients.begin() + idx);
                                std::cout << "Agent " << idx << " eleminated.\n";
                            }
                            else
                                std::cout << "No such active agent.\n";
                        } catch(...){}
                        std::cout << "Continue!" << std::endl;
                    }
                }
            }

            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            SOCKET client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_sock == INVALID_SOCKET) {
                if (server_running) {
                    std::cerr << "Accept failed" << std::endl;
                }
                continue;
            }

            // Check if hub is full
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (get_active_client_count() >= max_clients) {
                    std::cout << "Hub full, rejecting client" << std::endl;
                    send(client_sock, "HUB_FULL", 9, 0);
                    closesocket(client_sock);
                    continue;
                }
            }

            // Authenticate
            char recv_password[256] = {0};
            int bytes = recv(client_sock, recv_password, sizeof(recv_password) - 1, 0);
            if (bytes <= 0) {
                closesocket(client_sock);
                continue;
            }
            
            if (std::string(recv_password) != password) {
                std::cout << "Wrong password from connecting client" << std::endl;
                send(client_sock, "REJECT", 7, 0);
                closesocket(client_sock);
                continue;
            }

            send(client_sock, "ACCEPT", 7, 0);
            
            // Create client handler
            ClientHandler* handler = new ClientHandler(client_sock, next_client_id++);
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                clients.push_back(handler);
            }
            
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
            std::cout << "[+] Client " << handler->client_id 
                      << " connected from " << client_ip 
                      << " (active: " << get_active_client_count() << "/" << max_clients << ")" 
                      << std::endl;
            
            // Handle client in separate thread
            std::thread([this, handler]() {
                this->handle_client(handler);
            }).detach();
        }

        training_thread.join();
        cleanup_thread.join();
        
        closesocket(server_sock);
#if !defined(__unix__) && !defined(__APPLE__)
        WSACleanup();
#endif
    }

    ~DynamicAgentServer() {
        // Save final state
        std::cout << "\nShutting down server..." << std::endl;
        server_running = false;
        
        std::string model_path = checkpoint_path + "/model.pt";
        std::string optim_path = checkpoint_path + "/optimizer.pt";
        
        torch::save(model, model_path);
        torch::save(*optimizer, optim_path);
        
        std::cout << "Final checkpoint saved to " << checkpoint_path << std::endl;
        
        // Cleanup clients
        std::lock_guard<std::mutex> lock(mtx);
        for (auto* client : clients) {
            if (client->is_active) {
                closesocket(client->sock);
            }
            delete client;
        }
    }

private:
    AgentModel model{nullptr};
    int port, max_clients, min_clients_for_update;
    std::string password, checkpoint_path;
    std::vector<torch::Tensor> theta_new, theta_old, update_vector;
    std::unique_ptr<torch::optim::AdamW> optimizer;
    std::vector<ClientHandler*> clients;

    int get_active_client_count() {
        int count = 0;
        for (auto* client : clients)
            if (client->is_active)
                count++;
        return count;
    }

    void handle_client(ClientHandler* handler) {
        // Send current checkpoint to new client
        send_checkpoint(handler);
        
        while (server_running && handler->is_active) {
            std::cout << "[-] Agent " << handler->client_id << " entered the room." << std::endl;
            try {
                char type;
                if (recv_block(handler->sock, &type, 1) <= 0) {
                    std::cout << "[-] Client " << handler->client_id << " disconnected" << std::endl;
                    break;  // Connection closed
                }

                handler->last_seen = std::chrono::steady_clock::now();

                if (type == 'Q') {
                    // Client wants to disconnect gracefully
                    std::cout << "[-] Client " << handler->client_id << " disconnecting gracefully" << std::endl;
                    break;
                }
                else if (type == 'G') {
                    // Receive gradients
                    handler->gradients = recv_tensor_vector(handler->sock);
                    
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        handler->gradient_ready = true;
                        cv_update_ready.notify_all();
                    }
                    
                    std::cout << "[G] Client " << handler->client_id 
                              << " sent gradients" << std::endl;
                }
                else if (type == 'U') {
                    // Client requests update
                    std::unique_lock<std::mutex> lock(mtx);
                    
                    // Wait for new update if client is up-to-date
                    int current_version = update_version.load();
                    if (handler->last_update_version >= current_version) {
                        cv_update_ready.wait(lock, [this, current_version]() {
                            return update_version.load() > current_version || !server_running;
                        });
                    }
                    
                    if (!server_running) break;
                    
                    // Send update
                    char response_type = 'U';
                    send_block(handler->sock, &response_type, 1);
                    send_tensor_vector(handler->sock, update_vector);
                    
                    handler->last_update_version = update_version.load();
                    
                    std::cout << "[U] Client " << handler->client_id 
                              << " received update v" << handler->last_update_version << std::endl;
                }
                else {
                    std::cerr << "Unknown message type from client " << handler->client_id 
                              << ": " << type << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error handling client " << handler->client_id 
                          << ": " << e.what() << std::endl;
                break;
            }
        }
        
        // Client disconnected
        {
            std::lock_guard<std::mutex> lock(mtx);
            handler->is_active = false;
            handler->gradient_ready = false;
        }
        
        closesocket(handler->sock);
        std::cout << "[-] Agent " << handler->client_id << " job is done." << std::endl;
    }

    void send_checkpoint(ClientHandler* handler) {
        try {
            std::lock_guard<std::mutex> lock(mtx);
            
            char type = 'C';
            send_block(handler->sock, &type, 1);
            send_tensor_vector(handler->sock, model->parameters());
            
            std::cout << "[C] Sent checkpoint to client " << handler->client_id 
                      << " (version " << update_version.load() << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to send checkpoint to client " << handler->client_id 
                      << ": " << e.what() << std::endl;
        }
    }

    void training_loop() {
        int round = 0;
        
        while (server_running) {
            // Wait for at least min_clients_for_update to have gradients ready
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_update_ready.wait_for(lock, std::chrono::seconds(10), [this]() {
                    int ready_count = 0;
                    for (auto* client : clients) {
                        if (client->is_active && client->gradient_ready) {
                            ready_count++;
                        }
                    }
                    return ready_count >= min_clients_for_update || !server_running;
                });
                
                if (!server_running) break;
            }
            
            // Check if we have enough clients
            int ready_count = 0;
            {
                std::lock_guard<std::mutex> lock(mtx);
                for (auto* client : clients) {
                    if (client->is_active && client->gradient_ready) {
                        ready_count++;
                    }
                }
            }
            
            if (ready_count < min_clients_for_update) {
                continue;  // Not enough clients yet
            }
            
            std::cout << "\n=== Update Round " << round << " ===" << std::endl;
            std::cout << "Active clients: " << get_active_client_count() << "/" << max_clients << std::endl;
            std::cout << "Clients with gradients: " << ready_count << std::endl;
            
            // Aggregate and update
            aggregate_and_update();
            compute_update_vector();

            // Increment version
            update_version++;
            std::cout << "Model version: " << update_version.load() << std::endl;
            
            
            // Reset gradient flags
            {
                std::lock_guard<std::mutex> lock(mtx);
                for (auto* client : clients) {
                    client->gradient_ready = false;
                    client->gradients.clear();
                }
            }
            
            // Notify all clients that update is ready
            cv_update_ready.notify_all();
            
            // Save checkpoint
            std::string model_path = checkpoint_path + "/model.pt";
            torch::save(model, model_path);
            std::cout << "Checkpoint saved (round " << round << ")" << std::endl;

            round++;
        }
        
        std::cout << "Training loop ended after " << round << " rounds" << std::endl;
    }

    void aggregate_and_update() {
        int sz = model->parameters().size();
        int num_groups = 3;   // must match client
        std::vector<std::vector<torch::Tensor>> weighted_grad_sums(num_groups);  // list of parameter vectors
        std::vector<int> total_counts(num_groups, 0);
        for (int g = 0; g < num_groups; ++g) {
            weighted_grad_sums[g] = std::vector<torch::Tensor>(sz);
            for (int p = 0; p < sz; ++p)
                weighted_grad_sums[g][p] = torch::zeros_like(model->parameters()[p]);
        }

        int contributing_clients = 0;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (auto* client : clients) {
                if (client->is_active && client->gradient_ready) {
                    // client->gradients now stores the packed tensor vector
                    // Parse: first tensor is counts (int), then per-action gradient blocks
                    auto& tensors = client->gradients;
                    auto counts_tensor = tensors[0].to(torch::kInt32);
                    auto counts_acc = counts_tensor.accessor<int,1>();
                    std::vector<int> client_counts(num_groups);

                    for (int g = 0; g < num_groups; ++g)
                        client_counts[g] = counts_acc[g];

                    size_t idx = 1;
                    for (int g = 0; g < num_groups; ++g) {
                        if (client_counts[g] > 0) {
                            // read sz tensors (the gradient for that group)
                            for (int p = 0; p < sz; ++p) {
                                weighted_grad_sums[g][p] += tensors[idx] * client_counts[g];
                                idx++;
                            }
                            total_counts[g] += client_counts[g];
                        }
                    }
                    contributing_clients++;
                }
            }
        }

        if (contributing_clients == 0) return;

        // Compute per-group average gradients from accumulated sums
        std::vector<std::vector<torch::Tensor>> avg_grads;
        for (int g = 0; g < num_groups; ++g) {
            if (total_counts[g] > 0) {
                std::vector<torch::Tensor> avg(sz);
                for (int p = 0; p < sz; ++p)
                    if(total_counts[g] > 0)
                        avg[p] = weighted_grad_sums[g][p] / total_counts[g];
                avg_grads.push_back(avg);
            }
        }

        // MGDA on server side
        auto alphas = solve_mgda(avg_grads);

        // Build final combined gradient
        std::vector<torch::Tensor> final_grad(sz);
        for (int p = 0; p < sz; ++p)
            final_grad[p] = torch::zeros_like(model->parameters()[p]);
        
        int active_idx = 0;
        for (int g = 0; g < num_groups; ++g)
            if (total_counts[g] > 0) {
                for (int p = 0; p < sz; ++p)
                    final_grad[p] += alphas[active_idx] * avg_grads[active_idx][p];
                ++active_idx;
            }

        // Store old parameters
        theta_old.clear();
        for (int p = 0; p < sz; ++p)
            theta_old.push_back(model->parameters()[p].clone().detach());

        // Apply optimizer step using final gradient
        optimizer->zero_grad();
        for (int p = 0; p < sz; ++p)
            if (model->parameters()[p].grad().defined())
                model->parameters()[p].mutable_grad() = final_grad[p];
            else
                model->parameters()[p].mutable_grad() = final_grad[p].clone();
        optimizer->step();

        // Compute update vector for clients
        theta_new.clear();
        for (int p = 0; p < sz; ++p)
            theta_new.push_back(model->parameters()[p].clone().detach());

        std::cout << "Aggregated " << avg_grads.size() << " objectives from "
              << contributing_clients << " clients, MGDA alphas applied" << std::endl;
    }

    void compute_update_vector() {
        update_vector.clear();
        float norm = 0.0f;
        for (size_t i = 0; i < theta_new.size(); ++i) {
            auto v = (theta_new[i] - theta_old[i]).detach().clone();
            update_vector.push_back(v);
            norm += v.pow(2).sum().item<float>();
        }
        
        std::cout << "Update L2 norm: " << std::sqrt(norm) << std::endl;
    }

    void cleanup_inactive_clients() {
        const int timeout_seconds = 600;  // 10 minutes
        
        while (server_running) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            auto now = std::chrono::steady_clock::now();
            std::vector<int> removed_ids;
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                
                auto it = clients.begin();
                while (it != clients.end()) {
                    auto* client = *it;
                    
                    if (!client->is_active) {
                        auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
                            now - client->last_seen
                        ).count();
                        
                        if (idle_time > timeout_seconds) {
                            removed_ids.push_back(client->client_id);
                            delete client;
                            it = clients.erase(it);
                            continue;
                        }
                    }
                    
                    ++it;
                }
            }
            
            if (!removed_ids.empty()) {
                std::cout << "[Cleanup] Removed " << removed_ids.size() 
                          << " inactive clients from memory" << std::endl;
            }
        }
    }

    // Communication helpers
    void send_block(SOCKET sock, const char* data, size_t size) {
        size_t sent = 0;
        while (sent < size) {
            ssize_t n = send(sock, data + sent, std::min(size - sent, (size_t)BUFFER_SIZE), 0);
            if (n <= 0) throw std::runtime_error("send failed");
            sent += n;
        }
    }

    ssize_t recv_block(SOCKET sock, char* buffer, size_t size) {
        size_t received = 0;
        while (received < size) {
            ssize_t n = recv(sock, buffer + received, std::min(size - received, (size_t)BUFFER_SIZE), 0);
            if (n <= 0) return n;
            received += n;
        }
        return received;
    }

    void send_tensor_vector(SOCKET sock, const std::vector<torch::Tensor>& tensors) {
        std::stringstream ss;
        torch::save(tensors, ss);
        std::string data = ss.str();
        uint64_t size = data.size();
        send_block(sock, reinterpret_cast<const char*>(&size), sizeof(size));
        send_block(sock, data.data(), size);
    }

    std::vector<torch::Tensor> recv_tensor_vector(SOCKET sock) {
        uint64_t size;
        recv_block(sock, reinterpret_cast<char*>(&size), sizeof(size));
        std::string data(size, '\0');
        recv_block(sock, &data[0], size);
        std::stringstream ss(data);
        std::vector<torch::Tensor> tensors;
        torch::load(tensors, ss);
        return tensors;
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

int main() {
    std::cout << "=== Dynamic Agent Training Hub ===" << std::endl;
    
    int port = 8081;
    std::string password, checkpoint_path = "server_checkpoint";
    int max_clients = 10;
    int min_clients_for_update = 1;
    double learning_rate = 1e-3;
    
    std::cout << "Port [8081]: ";
    std::string s;
    std::getline(std::cin, s);
    if (!s.empty()) port = std::stoi(s);
    
    std::cout << "Password: ";
    std::getline(std::cin, password);
    
    std::cout << "Max clients (hub capacity) [10]: ";
    std::getline(std::cin, s);
    if (!s.empty()) max_clients = std::stoi(s);
    
    std::cout << "Min clients for update [1]: ";
    std::getline(std::cin, s);
    if (!s.empty()) min_clients_for_update = std::stoi(s);
    
    std::cout << "Learning rate [1e-3]: ";
    std::getline(std::cin, s);
    if (!s.empty()) learning_rate = std::stod(s);
    
    std::cout << "Checkpoint directory [server_checkpoint]: ";
    std::getline(std::cin, s);
    if (!s.empty()) checkpoint_path = s;
    
    DynamicAgentServer* server = new DynamicAgentServer(
        port, password, max_clients, checkpoint_path, learning_rate, min_clients_for_update
    );
    
    server->start();
    delete server;
    
    return 0;
}