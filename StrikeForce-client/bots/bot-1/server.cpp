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

#pragma once

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <unistd.h>
#include <time.h>
#include <execution>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netdb.h>
#else
#include "../../inet_for_windows.hpp"
#endif

int BUFFER_SIZE;
static int PORT;
static std::string PASS;

std::vector<bool> disconnect;
std::vector<int> clients;
std::vector<double> h[1024], c[1024];

char buffer[200000];

int n, m, cnt;

void rcv_h(){
    for(int i = 0; i < n; ++i){
        if(disconnect[i])
            continue;
        h[i].assign(BUFFER_SIZE / sizeof(double), 0);
		if(recv(clients[i], reinterpret_cast<char*>(h[i].data()), BUFFER_SIZE, 0) <= 0)
            disconnect[i] = true;
    }
	return;
}

void send_c(){
    for(int i = 0; i < n; ++i)
        if(!disconnect[i]){
            c[i].assign(BUFFER_SIZE / sizeof(double), 0);
            for(int j = 0; j < n; ++j)
                if(i != j && !disconnect[j])
                    for(int k = 0; k < c[j].size(); ++k)
                        c[i][k] += h[j][k];
            for(int j = 0; j < c[i].size(); ++j)
                c[i][j] /= std::max(n - 1, 1);
        }
	for(int i = 0; i < n; ++i)
        if(!disconnect[i])
		    send(clients[i], reinterpret_cast<const char*>(c[i].data()), BUFFER_SIZE, 0);
	return;
}

int main(){
	std::cout << "StrikeForce-CommNet-server\n";
	std::cout << "Created by: 21\n";
	std::cout << "____________________________________________________\n\n";
    #if !defined(__unix__) && !defined(__APPLE__)
    WSADATA wsaData;
    if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0){
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }
    #endif
	std::cout << "Is it a global server or local?\n(G:global/any thing else:local)\n";
	std::string s;
	std::cin >> s;
	std::cout << "Server IP: ";
	if(s == "G"){
		std::cout.flush();
		system("curl -s https://api.ipify.org");
	}
	else{
		char host[256];
		gethostname(host, sizeof(host));
		std::cout << inet_ntoa(*((struct in_addr*)gethostbyname(host)->h_addr_list[0]));
	}
    std::cout << "\nSize of communication (enter a number): ";
	std::cin >> BUFFER_SIZE;
    BUFFER_SIZE *= sizeof(double);
	std::cout << "\nListening on port (enter a port): ";
	std::cin >> PORT;
	std::cout << "-------------\n";
	std::cout << "choose a password: ";
	std::getline(std::cin, PASS);
	std::getline(std::cin, PASS);
	std::cout << "-------------\n";
	std::cout << "Please enter n (the number of the players)\n";
	std::cin >> n;
	cnt = n;
	disconnect.assign(n, false);
	int server_socket, client_socket;
	struct sockaddr_in server_addr, client_addr;
	socklen_t addr_len = sizeof(client_addr);
	server_socket = socket(AF_INET, SOCK_STREAM, 0);
	if(server_socket == -1){
		std::cerr << "Failed to create socket" << std::endl;
		return 1;
	}
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = INADDR_ANY;
	server_addr.sin_port = htons(PORT);
	if(bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1){
		std::cerr << "Failed to bind socket" << std::endl;
		return 1;
	}
	if(listen(server_socket, SOMAXCONN) == -1){
		std::cerr << "Failed to listen on socket" << std::endl;
		return 1;
	}
	std::cout << "Server is running..." << std::endl;
	while(clients.size() != n){
		client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &addr_len);
		if(client_socket == -1){
			std::cerr << "Failed to accept client" << std::endl;
			continue;
		}
		std::cout << "Connection from " << inet_ntoa(client_addr.sin_addr) << std::endl;
		char password[32] = {};
		recv(client_socket, password, 32, 0);
		disconnect[0] = false;
		if(std::string(password) == PASS){
			send(client_socket, "A", 2, 0);
			std::cout << "Password ACCEPTED" << std::endl;
			clients.push_back(client_socket);
		}
		else{
			send(client_socket, "R", 2, 0);
			std::cout << "Password REJETED" << std::endl;
		}
	}
    for(int client_socket: clients){
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000;
        #if defined(__unix__) || defined(__APPLE__)
        auto t = &timeout;
        #else
        int ms = 100;
        char* t = (char*)&ms;
        #endif
        if(setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, t, sizeof(timeout)) < 0){
            std::cout << "Error setting socket options" << '\n';
            return 1;
        }
    }
	while(cnt){
		rcv_h();
		send_c();
	}
	close(server_socket);
	#if !defined(__unix__) && !defined(__APPLE__)
    WSACleanup();
    #endif
	std::cout << "GAME OVER\n";
	std::cout << "Enter \"done!\" to end the program.\n";
    std::cout << "___________________________\n";
	std::string str = "";
	while(str != "done!")
        std::cin >> str;
	return 0;
}