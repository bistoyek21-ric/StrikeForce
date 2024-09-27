/*
MIT License

Copyright (c) 2024 bistoyek(21)

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
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <unistd.h>
#include <time.h>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netdb.h>

#else
#include "win_arpa_inet.hpp"
#endif

int constexpr PORT = 10000, BUFFER_SIZE = 2048, BS = 2;

std::vector<int> clients;
std::vector<bool> alive;
std::vector<char> command;

int n, cnt;

void my_recv(int sock, char *buffer, const int buffer_size, int flag){
    int len = 0;
	while(true){
    	recv(sock, buffer, 1, 0);
        if(!buffer[len])
			return;
		++buffer;
	}
	return;
}

void get_commands(){
	char buffer[BS], c;
	for(int i = 0; i < n; ++i)
		if(alive[i]){
			memset(buffer, 0, BS);
			my_recv(clients[i], buffer, BS, 0);
			sscanf(buffer, "%c", &command[i]);
			if(command[i] == '_')
				alive[i] = false, --cnt, close(clients[i]);
		}
	return;
}

void give_commands(){
	for(int i = 0; i < n; ++i)
		if(alive[i]){
			for(int j  = 0; j < i; ++j)
				if(alive[j]){
					char msg[2] = {command[j]};
					send(clients[i], msg, 2, 0);
				}
			for(int j  = i + 1; j < n; ++j)
				if(alive[j]){
					char msg[2] = {command[j]};
					send(clients[i], msg, 2, 0);
				}
		}
	return;
}

int main(){
	std::cout << "StrikeForce\n";
	std::cout << "Created by: 21\n";
	std::cout << "____________________________________________________\n\n";
    #if !defined(__unix__) && !defined(__APLLE__)
    WSADATA wsaData;
    if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0){
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }
    #endif
	char host[256];
	gethostname(host, sizeof(host));
	std::cout << "Server IP: " << inet_ntoa(*((struct in_addr*)gethostbyname(host)->h_addr_list[0])) << '\n';
	std::cout << "Running on port: " << PORT << '\n';
	std::vector<int> team;
	time_t tb = time(nullptr);
	srand(tb);
	long long serial_number = ((rand() & 1023) << 20) + ((rand() & 1023) << 10) + (rand() & 1023);
	int m;
	std::cout << "-------------\n";
	std::cout << "start: " << tb << '\n';
	std::cout << "serial_number: " << serial_number << '\n';
	std::cout << "-------------\n";
	std::cout << '\n';
	std::cout << "Please enter n (the number of the players) and  m (number of teams respectively)\n";
	std::cout << "And enter the team of i-th player (they should be in range of [1, m])\n";
	std::cin >> n >> m;
	for(int i = 0; i < n; ++i){
		int num;
		std::cin >> num;
		team.push_back(num);
	}
	cnt = n;
	command.assign(n, '+');
	alive.assign(n, true);
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
	if(listen(server_socket, n) == -1){
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
		clients.push_back(client_socket);
	}
	for(int i = 0; i < n; ++i){
		std::string msg = std::to_string(tb) + " " + std::to_string(serial_number);
		send(clients[i], msg.c_str(), msg.size() + 1, 0);
	}
	for(int i = 0; i < n; ++i){
		std::string msg = std::to_string(n) + " " + std::to_string(i) + " " + std::to_string(team[i]);
		send(clients[i], msg.c_str(), msg.size() + 1, 0);
	}
	for(int i = 0; i < n; ++i){
		char buffer[BUFFER_SIZE];
		memset(buffer, 0, BUFFER_SIZE);
		my_recv(clients[i], buffer, BUFFER_SIZE, 0);
		for(int j = 0; j < n; ++j)
			if(i != j){
				send(clients[j], buffer, strlen(buffer) + 1, 0);
				std::string msg = std::to_string(team[i]);
				send(clients[j], msg.c_str(), msg.size() + 1, 0);
			}
    }
	while(cnt){
		get_commands();
		give_commands();
	}
	close(server_socket);
	#if !defined(__unix__) && !defined(__APPLE__)
    WSACleanup();
    #endif
	return 0;
}