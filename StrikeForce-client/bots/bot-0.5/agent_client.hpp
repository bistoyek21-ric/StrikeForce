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

const int BUFFER_SIZE = 2048;

char AgentBuffer[BUFFER_SIZE];

class AgentClient{
public:

    AgentClient(std::string dir): dir(dir) {
        while (true) {
            head();
            std::string server_ip, server_port_s, server_password;
			int server_port = 0;
			std::cout << "JOINIGN INTO THE AGENTS SERVER:" << std::endl;
			std::cout << "Enter the server IP: ";
			std::cout.flush();
			getline(std::cin, server_ip);
			std::cout << "Enter the server port: ";
			std::cout.flush();
			getline(std::cin, server_port_s);
			std::cout << "Enter the server's password: ";
			std::cout.flush();
			getline(std::cin, server_password);
			for(auto e: server_port_s)
				server_port = 10 * server_port + (e - '0');
			server_port = std::max(std::min(server_port, (1 << 16) - 1), 0);
            bool is_connected = connectAgnet(server_ip, server_port, server_password);
            std::cout << "press space button to continue" << std::endl;
			while(getch() != ' ');
            if (is_connected)
                break;
        }
    }

    ~AgentClient(){
        disconnectAgent();
    }

    void send_gradient(std::vector<torch::Tensor> grad) {
        char type = 'G';
        send_block(&type, 1);
        send_tensor_vector(grad);
    }

    std::vector<torch::Tensor> get_update_vector() {
        char type = 'U';
        send_block(&type, 1);
        char response_type;
        recv_block(&response_type, 1);
        if (response_type != 'U') {
            throw std::runtime_error("Expected update vector message");
        }
        return recv_tensor_vector();
    }

    private:
   	
    void disconnectAgent(){
        close(sock);
		#if !defined(__unix__) && !defined(__APPLE__)
		WSACleanup();
		#endif
		return;
    }

    bool connectAgnet(const std::string& server_ip, int server_port, const std::string& server_password){
		#if !defined(__unix__) && !defined(__APPLE__)
		WSADATA wsaData;
		if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0){
			std::cout << "WSAStartup failed" << std::endl;
			return false;
        }
		#endif
		server_addr.sin_family = AF_INET;
		server_addr.sin_port = htons(server_port);
		inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);
		sock = socket(AF_INET, SOCK_STREAM, 0);
		if(sock == -1){
			std::cout << "Failed to create socket" << std::endl;
			return false;
		}
		if(connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1){
			std::cout << "Failed to connect to server" << std::endl;
			return false;
		}
		send(sock, server_password.c_str(), server_password.size() + 1, 0);
		char buffer[7] = {};
	    my_recv(sock, buffer, 7, 0);
		if(buffer[0] == 'R'){
			std::cout << "Wrong password entered" << std::endl;
			return false;
		}
		return get_checkpoint();
	}

    bool get_checkpoint() {
        char type;
        recv_block(&type, 1);
        if (type != 'C') {
            std::cout << "Expected checkpoint message" << std::endl;
            return false;
        }
        auto checkpoint = recv_tensor_vector();
        torch::save(checkpoint, dir + "/checkpoint.pt");
        return true;
    }

    void send_block(const char* data, size_t size) {
        size_t sent = 0;
        while (sent < size) {
            size_t chunk_size = std::min(static_cast<size_t>(BUFFER_SIZE), size - sent);
            std::memcpy(AgentBuffer, data + sent, chunk_size);
            ssize_t n = send(sock, AgentBuffer, chunk_size, 0);
            if (n <= 0)
                throw std::runtime_error("send failed");
            sent += n;
        }
    }

    void recv_block(char* buffer, size_t size) {
        size_t received = 0;
        while (received < size) {
            size_t chunk_size = std::min(static_cast<size_t>(BUFFER_SIZE), size - received);
            ssize_t n = recv(sock, AgentBuffer, chunk_size, 0);
            if (n <= 0)
                throw std::runtime_error("recv failed");
            std::memcpy(buffer + received, AgentBuffer, n);
            received += n;
        }
    }

    void send_tensor_vector(const std::vector<torch::Tensor>& tensors) {
        std::stringstream ss;
        torch::save(tensors, ss);
        std::string data = ss.str();
        uint64_t size = data.size();
        send_block(reinterpret_cast<const char*>(&size), sizeof(size));
        send_block(data.data(), size);
    }

    std::vector<torch::Tensor> recv_tensor_vector() {
        uint64_t size;
        recv_block(reinterpret_cast<char*>(&size), sizeof(size));
        std::string data(size, '\0');
        recv_block(&data[0], size);
        std::stringstream ss(data);
        std::vector<torch::Tensor> tensors;
        torch::load(tensors, ss);
        return tensors;
    }

    int sock;
	struct sockaddr_in server_addr;
    std::string dir;
}; 