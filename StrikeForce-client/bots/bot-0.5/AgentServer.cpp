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
//g++ -std=c++17 AgentServer.cpp -o app_agnet_server -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lpthread
// Add -lws2_32 on Windows

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <sstream>
#include <fstream>
#include <filesystem>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <torch/torch.h>

const int BUFFER_SIZE = 2048;

std::mutex mtx;
std::condition_variable cv;
std::atomic<int> connected_clients{0};
std::atomic<int> gradients_received{0};
std::atomic<bool> server_running{true};

struct ClientHandler {
    SOCKET sock;
    int client_id;
    std::vector<torch::Tensor> gradients;
    bool gradient_ready = false;
};

class AgentServer {
public:
    AgentServer(int port, const std::string& password, int num_clients, const std::string& checkpoint_path, double lr = 1e-3)
        : port(port), password(password), num_clients(num_clients), checkpoint_path(checkpoint_path) {
        
        if (std::filesystem::exists(checkpoint_path)) {
            if (std::filesystem::exists(checkpoint_path + "/model.pt")) {
                std::cout << "Loading checkpoint from: " << checkpoint_path << std::endl;
                torch::load(server_params, checkpoint_path);
                std::cout << "Loaded " << server_params.size() << " parameter tensors" << std::endl;
            }
        } else {
            std::filesystem::create_directories(checkpoint_path);
        }
        
        std::vector<torch::Tensor> params_for_optim;
        for (auto& p : server_params) {
            auto param = p.clone().set_requires_grad(true);
            params_for_optim.push_back(param);
        }
        optimizer = std::make_unique<torch::optim::AdamW>(
            params_for_optim, 
            torch::optim::AdamWOptions(lr)
        );

        if (std::filesystem::exists(checkpoint_path + "/optimizer.pt")) {
            try {
                torch::load(*optimizer, checkpoint_path + "/optimizer.pt");
            } catch (...) {}
        }
        
        for (auto& p : server_params) {
            theta_old.push_back(p.clone().detach());
        }
    }

    void start() {
#if !defined(__unix__) && !defined(__APPLE__)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return;
        }
#endif

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

        if (listen(server_sock, num_clients) == SOCKET_ERROR) {
            std::cerr << "Listen failed" << std::endl;
            closesocket(server_sock);
            return;
        }

        std::cout << "Agent Server listening on port " << port << std::endl;
        std::cout << "Waiting for " << num_clients << " clients..." << std::endl;

        for (int i = 0; i < num_clients; ++i) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            SOCKET client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_sock == INVALID_SOCKET) {
                std::cerr << "Accept failed for client " << i << std::endl;
                continue;
            }

            char recv_password[256] = {0};
            recv(client_sock, recv_password, sizeof(recv_password), 0);
            
            if (std::string(recv_password) != password) {
                std::cout << "Client " << i << " - Wrong password" << std::endl;
                send(client_sock, "REJECT", 7, 0);
                closesocket(client_sock);
                --i;
                continue;
            }

            send(client_sock, "ACCEPT", 7, 0);
            
            ClientHandler* handler = new ClientHandler{client_sock, i, {}, false};
            clients.push_back(handler);
            
            std::cout << "Client " << i << " connected" << std::endl;
            
            std::thread([this, handler]() {
                this->send_checkpoint(handler);
                this->handle_client(handler);
            }).detach();
            
            connected_clients++;
        }

        std::cout << "All " << num_clients << " clients connected!" << std::endl;
        training_loop();

        closesocket(server_sock);
#if !defined(__unix__) && !defined(__APPLE__)
        WSACleanup();
#endif
    }

    ~AgentServer() {
        torch::save(optimizer, checkpoint_path + "/optimizer.pt");
    }

private:
    int port, num_clients;
    std::string password, checkpoint_path;
    std::vector<torch::Tensor> server_params, theta_old, update_vector;
    std::unique_ptr<torch::optim::AdamW> optimizer;
    std::vector<ClientHandler*> clients;

    void send_checkpoint(ClientHandler* handler) {
        char type = 'C';
        send_block(handler->sock, &type, 1);
        send_tensor_vector(handler->sock, server_params);
        std::cout << "Sent checkpoint to client " << handler->client_id << std::endl;
    }

    void handle_client(ClientHandler* handler) {
        while (server_running) {
            try {
                char type;
                if (recv_block(handler->sock, &type, 1) <= 0) break;
                if (type == 'Q') {
                    std::cout << "Agent " << handler->client_id << " disconnected." << std::endl;
                    // update clients
                    break;
                }
                if (type == 'G') {
                    handler->gradients = recv_tensor_vector(handler->sock);
                    
                    std::lock_guard<std::mutex> lock(mtx);
                    handler->gradient_ready = true;
                    gradients_received++;
                    cv.notify_all();
                    
                    std::cout << "Received gradients from client " << handler->client_id 
                              << " (" << gradients_received.load() << "/" << num_clients << ")" << std::endl;
                }
                else if (type == 'U') {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this]() { return gradients_received.load() == num_clients; });
                    
                    char response_type = 'U';
                    send_block(handler->sock, &response_type, 1);
                    send_tensor_vector(handler->sock, update_vector);
                    
                    std::cout << "Sent update to client " << handler->client_id << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error handling client " << handler->client_id << ": " << e.what() << std::endl;
                break;
            }
        }
        closesocket(handler->sock);
    }

    void training_loop() {
        int round = 0;
        while (server_running) {
            std::cout << "\n=== Round " << round << " ===" << std::endl;
            
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() { return gradients_received.load() == num_clients; });
            }
            
            std::cout << "Aggregating gradients..." << std::endl;
            aggregate_and_update();
            compute_update_vector();
                        
            torch::save(server_params, checkpoint_path + "/model.pt");
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                gradients_received = 0;
                for (auto* client : clients) {
                    client->gradient_ready = false;
                    client->gradients.clear();
                }
            }
            
            round++;
        }
    }

    void aggregate_and_update() {
        std::vector<torch::Tensor> avg_gradients;
        
        for (size_t i = 0; i < server_params.size(); ++i) {
            torch::Tensor sum_grad = torch::zeros_like(server_params[i]);
            for (auto* client : clients) {
                if (client->gradient_ready && i < client->gradients.size()) {
                    sum_grad += client->gradients[i];
                }
            }
            avg_gradients.push_back(sum_grad / num_clients);
        }
        
        for (size_t i = 0; i < server_params.size(); ++i) {
            theta_old[i] = server_params[i].clone().detach();
        }
        
        optimizer->zero_grad();
        auto params = optimizer->param_groups()[0].params();
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].grad().defined()) {
                params[i].mutable_grad() = avg_gradients[i];
            } else {
                params[i].mutable_grad() = avg_gradients[i].clone();
            }
        }
        
        optimizer->step();
        
        for (size_t i = 0; i < params.size(); ++i) {
            server_params[i] = params[i].detach().clone();
        }
    }

    void compute_update_vector() {
        update_vector.clear();
        float norm = 0.0f;
        for (size_t i = 0; i < server_params.size(); ++i) {
            auto v = (server_params[i] - theta_old[i]).detach().clone();
            update_vector.push_back(v);
            norm += v.pow(2).sum().item<float>();
        }
        std::cout << "Update L2 norm: " << std::sqrt(norm) << std::endl;
    }

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
};

int main() {
    std::cout << "=== Agent Training Server ===" << std::endl;
    
    int port = 8081;
    std::string password, checkpoint_path = "server_checkpoint";
    int num_clients = 2;
    
    std::cout << "Port [8081]: ";
    std::string s;
    std::getline(std::cin, s);
    if (!s.empty()) port = std::stoi(s);
    
    std::cout << "Password: ";
    std::getline(std::cin, password);
    
    std::cout << "Number of clients: ";
    std::getline(std::cin, s);
    num_clients = std::stoi(s);
    
    std::cout << "Checkpoint path [server_checkpoint]: ";
    std::getline(std::cin, s);
    if (!s.empty()) checkpoint_path = s;
    
    AgentServer* server = new AgentServer(port, password, num_clients, checkpoint_path);
    server->start();
    delete server;
    return 0;
}