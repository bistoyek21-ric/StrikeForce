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
#include "../../basic.hpp"
#include <torch/torch.h>
#include <sstream>
#include <filesystem>

const int BUFFER_SIZE = 2048;
char AgentBuffer[BUFFER_SIZE];

class AgentClient {
public:
    AgentClient(std::string dir) : dir(dir) {
        while (true) {
            head();
            std::string server_ip, server_port_s, server_password;
            int server_port = 0;
            
            std::cout << "=== JOINING AGENT SERVER ===" << std::endl;
            std::cout << "Server IP: ";
            std::cout.flush();
            std::getline(std::cin, server_ip);
            
            std::cout << "Server port: ";
            std::cout.flush();
            std::getline(std::cin, server_port_s);
            
            std::cout << "Password: ";
            std::cout.flush();
            std::getline(std::cin, server_password);
            
            for (auto e : server_port_s)
                server_port = 10 * server_port + (e - '0');
            server_port = std::max(std::min(server_port, (1 << 16) - 1), 0);
            
            bool is_connected = connectAgent(server_ip, server_port, server_password);
            
            std::cout << "Press space to continue" << std::endl;
            while (getch() != ' ');
            
            if (is_connected)
                break;
        }
    }

    ~AgentClient() {
        disconnectAgent();
    }

    void send_gradient(const std::vector<torch::Tensor>& grad) {
        try {
            char type = 'G';
            send_block(&type, 1);
            send_tensor_vector(grad);
            std::cout << "Gradients sent successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error sending gradients: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<torch::Tensor> get_update_vector() {
        try {
            char type = 'U';
            send_block(&type, 1);
            
            char response_type;
            recv_block(&response_type, 1);
            
            if (response_type != 'U') {
                throw std::runtime_error("Expected update vector message, got: " + std::to_string(response_type));
            }
            
            auto update = recv_tensor_vector();
            std::cout << "Received update vector with " << update.size() << " tensors" << std::endl;
            return update;
        } catch (const std::exception& e) {
            std::cerr << "Error getting update vector: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void disconnectAgent() {
        if (sock != -1) {
            send(sock, "Q", 2, 0);
            close(sock);
            sock = -1;
        }
#if !defined(__unix__) && !defined(__APPLE__)
        WSACleanup();
#endif
    }

    bool connectAgent(const std::string& server_ip, int server_port, const std::string& server_password) {
#if !defined(__unix__) && !defined(__APPLE__)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cout << "WSAStartup failed" << std::endl;
            return false;
        }
#endif

        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);
        
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            std::cout << "Failed to create socket" << std::endl;
            return false;
        }
        
        std::cout << "Connecting to " << server_ip << ":" << server_port << "..." << std::endl;
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            std::cout << "Failed to connect to server" << std::endl;
            close(sock);
            sock = -1;
            return false;
        }
        
        // Send password
        send(sock, server_password.c_str(), server_password.size() + 1, 0);
        
        // Wait for response
        char buffer[7] = {};
        int bytes_received = recv(sock, buffer, 7, 0);
        
        if (bytes_received <= 0 || buffer[0] == 'R') {
            std::cout << "Authentication failed" << std::endl;
            close(sock);
            sock = -1;
            return false;
        }
        
        std::cout << "Authentication successful" << std::endl;
        
        // Receive initial checkpoint
        return get_checkpoint();
    }

    bool get_checkpoint() {
        try {
            std::cout << "Waiting for checkpoint from server..." << std::endl;
            
            char type;
            if (recv_block(&type, 1) <= 0) {
                std::cout << "Failed to receive checkpoint type" << std::endl;
                return false;
            }
            
            if (type != 'C') {
                std::cout << "Expected checkpoint message, got: " << type << std::endl;
                return false;
            }
            
            auto checkpoint = recv_tensor_vector();
            
            // Save checkpoint to file
            std::filesystem::create_directories(dir);
            std::string checkpoint_path = dir + "/checkpoint.pt";
            torch::save(checkpoint, checkpoint_path);
            
            std::cout << "Checkpoint received and saved: " << checkpoint.size() << " tensors" << std::endl;
            std::cout << "Saved to: " << checkpoint_path << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error receiving checkpoint: " << e.what() << std::endl;
            return false;
        }
    }

    void send_block(const char* data, size_t size) {
        size_t sent = 0;
        while (sent < size) {
            size_t chunk_size = std::min(static_cast<size_t>(BUFFER_SIZE), size - sent);
            std::memcpy(AgentBuffer, data + sent, chunk_size);
            ssize_t n = send(sock, AgentBuffer, chunk_size, 0);
            if (n <= 0) {
                throw std::runtime_error("send failed: connection closed");
            }
            sent += n;
        }
    }

    ssize_t recv_block(char* buffer, size_t size) {
        size_t received = 0;
        while (received < size) {
            size_t chunk_size = std::min(static_cast<size_t>(BUFFER_SIZE), size - received);
            ssize_t n = recv(sock, AgentBuffer, chunk_size, 0);
            if (n <= 0) {
                throw std::runtime_error("recv failed: connection closed");
            }
            std::memcpy(buffer + received, AgentBuffer, n);
            received += n;
        }
        return received;
    }

    void send_tensor_vector(const std::vector<torch::Tensor>& tensors) {
        std::stringstream ss;
        torch::save(tensors, ss);
        std::string data = ss.str();
        uint64_t size = data.size();
        
        // Send size first
        send_block(reinterpret_cast<const char*>(&size), sizeof(size));
        // Send data
        send_block(data.data(), size);
    }

    std::vector<torch::Tensor> recv_tensor_vector() {
        uint64_t size;
        // Receive size first
        recv_block(reinterpret_cast<char*>(&size), sizeof(size));
        
        // Sanity check
        if (size == 0 || size > 1e9) {  // 1GB max
            throw std::runtime_error("Invalid tensor vector size: " + std::to_string(size));
        }
        
        // Receive data
        std::string data(size, '\0');
        recv_block(&data[0], size);
        
        // Deserialize
        std::stringstream ss(data);
        std::vector<torch::Tensor> tensors;
        torch::load(tensors, ss);
        
        return tensors;
    }

    int sock = -1;
    struct sockaddr_in server_addr;
    std::string dir;
};