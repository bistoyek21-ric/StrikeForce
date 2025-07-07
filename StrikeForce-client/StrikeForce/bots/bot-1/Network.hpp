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

class Network{

public:
    bool disconnected;

    int BUFFER_SIZE;

	Network(int BUFFER_SIZE): BUFFER_SIZE(BUFFER_SIZE) {
		disconnected = false;
		std::string server_ip, server_port_s, server_password;
		int server_port = 0;
		#if !defined(__unix__) && !defined(__APPLE__)
		WSADATA wsaData;
		if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0){
			std::cout << "WSAStartup failed" << std::endl;
			return;
        }
		#endif
        while(true){
			server_port = 0;
			std::cout << "JOINIGN INTO AGENTS SERVER:" << std::endl;
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
		    server_addr.sin_family = AF_INET;
		    server_addr.sin_port = htons(server_port);
		    inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);
		    sock = socket(AF_INET, SOCK_STREAM, 0);
		    if(sock == -1){
			    std::cout << "Failed to create socket" << std::endl;
			    return;
		    }
		    if(connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1){
			    std::cout << "Failed to connect to server" << std::endl;
			    return;
		    }
		    send(sock, server_password.c_str(), server_password.size() + 1, 0);
		    char buffer[32] = {};
	    	my_recv(sock, buffer, 32, 0);
	        if(buffer[0] == 'R'){
		        std::cout << "Wrong password entered" << std::endl;
		        continue;
		    }
			else
				std::cout << "Password verified" << std::endl;
            break;
        }
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 200000;
        #if defined(__unix__) || defined(__APPLE__)
        auto t = &timeout;
        #else
        int ms = 200;
        char* t = (char*)&ms;
        #endif
        if(setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, t, sizeof(timeout)) < 0){
            std::cout << "Error setting socket options" << '\n';
            exit(1);
        }
		return;
	}

	void give_h(const std::vector<double> &v){
		if(disconnected)
			return;
		send(sock, reinterpret_cast<const char*>(v.data()), BUFFER_SIZE, 0);
		return;
	}

	std::vector<double> get_c(){
        std::vector<double> result(BUFFER_SIZE / sizeof(double), 0);
		if(disconnected)
			result;
		if(recv(sock, reinterpret_cast<char*>(result.data()), BUFFER_SIZE, 0) <= 0){
            disconnected = true;
            std::cout << "Failed to receive vector data or connection closed" << std::endl;
        }
        return result;
    }

    ~ Network(){
        close(sock);
		#if !defined(__unix__) && !defined(__APPLE__)
		WSACleanup();
		#endif
    }

private:
    int sock;
    struct sockaddr_in server_addr;
};