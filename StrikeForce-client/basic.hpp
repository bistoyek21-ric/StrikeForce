/*
MIT License

Copyright (c) 2024 bistoyek21 R.I.C.

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
#include <ctime>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <bitset>
#include <random>
#include <filesystem>

#include "GraphicPrinter.hpp"

#define CROWDSOURCED_TRAINING

bool during_battle = false;

void usleep(int x){
	sf::sleep(sf::microseconds(x));
	return;
}

#if defined(__unix__) || defined(__APPLE__)
#include <sys/select.h>
#include <termios.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int constexpr BK = 127, EN = 10;

void disable_input_buffering(){
	struct termios t;
	tcgetattr(STDIN_FILENO, &t);
	t.c_lflag &= ~ICANON;
	t.c_lflag &= ~ECHO;
	tcsetattr(STDIN_FILENO, TCSANOW, &t);
}

void restore_input_buffering(){
	struct termios t;
	tcgetattr(STDIN_FILENO, &t);
	t.c_lflag |= ICANON | ECHO;
	tcsetattr(STDIN_FILENO, TCSANOW, &t);
}

int kbhit(){
	if(!during_battle)
		disable_input_buffering();
	struct timeval tv = {0, 0};
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(STDIN_FILENO, &fds);
	int res = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
	if(!res)
		tcflush(STDIN_FILENO, TCIFLUSH);
	if(!during_battle)
		restore_input_buffering();
	return res;
}

int getch(){
	if(!during_battle)
		disable_input_buffering();
	int res = getchar();
	while(kbhit())
		res = getchar();
	tcflush(STDIN_FILENO, TCIFLUSH);
	if(!during_battle)
		restore_input_buffering();
	return res;
}

#else
#include <conio.h>

#include "inet_for_windows.hpp"

int constexpr BK = 8, EN = 13;

void disable_input_buffering(){
	return;
}

void restore_input_buffering(){
	return;
}
#endif

const std::string SERVER_URL = "http://89.106.206.119:8080";

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

int request_and_extract_backup(const std::string& dir, const std::string& bot_code) {
    if (std::filesystem::exists(dir))
        return 2;
    if (bot_code.empty() || dir.empty()) {
        std::cerr << "Error: bot_code or dir cannot be empty" << std::endl;
        return 1;
    }
    std::filesystem::path dir_path = dir;
    std::string download_cmd = "curl --noproxy \"*\" -o backup.zip \"" + SERVER_URL + "/StrikeForce/api/request_backup?bot=" + bot_code + "\"";
    if (system(download_cmd.c_str()) != 0) {
        std::cerr << "Failed to download backup.zip for bot_code: " << bot_code << std::endl;
        return 1;
    }
	if (!std::filesystem::exists(dir_path)) {
	    try {
    	    std::filesystem::create_directories(dir_path);
	    } catch (const std::filesystem::filesystem_error& e) {
    	    std::cerr << "Failed to create directory " << dir << ": " << e.what() << std::endl;
        	return 1;
    	}
	}
    std::string extract_cmd = "7z x -y -o" + escape_path(dir) + " backup.zip";
    int ret = 0;
    if (system(extract_cmd.c_str()) != 0) {
        std::cerr << "Failed to extract backup.zip to " << dir << std::endl;
        ret = 1;
    }
	if (!ret)
    	try {
        	std::filesystem::remove("backup.zip");
    	} catch (const std::filesystem::filesystem_error& e) {
        	std::cerr << "Failed to delete backup.zip: " << e.what() << std::endl;
        	return 1;
    	}
    std::cout << "Backup requested and extracted to " << dir << std::endl;
    return ret;
}

int zip_and_return_backup(const std::string dir) {
    if (dir.empty()) {
        std::cerr << "Error: dir cannot be empty" << std::endl;
        return 1;
    }
    std::filesystem::path dir_path = dir;
    if (!std::filesystem::exists(dir_path)) {
        std::cerr << "Error: Directory " << dir << " does not exist" << std::endl;
        return 1;
    }
	std::cout << " >> zipping the backup ...." << std::endl;
    std::string zip_name = "backup.zip";
    std::string zip_cmd = "cd " + escape_path(dir) + " && 7z a -tzip " + escape_path(zip_name) + " ./*";
    if (system(zip_cmd.c_str()) != 0) {
        std::cerr << "Failed to zip directory: " << dir << std::endl;
        return 1;
    }
	std::cout << " >> submitting into the server ...." << std::endl;
    std::string send_cmd = "cd " + escape_path(dir) + " && curl --noproxy \"*\" -X POST -F \"file=@" + escape_path(zip_name) + "\" \"" + SERVER_URL + "/StrikeForce/api/return_backup\"";
	if (system(send_cmd.c_str()) != 0)
        std::cerr << "\nFailed to send zip file to server" << std::endl;
	std::cout << std::endl;
    try {
        std::filesystem::remove(dir + "/" + zip_name);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Failed to delete zip: \"" << dir + "/" + zip_name << "\": " << e.what() << std::endl;
        return 1;
    }
	std::cout << "Do you want to delete the former backup? (press 'y' for yes and any other key for no)" << std::endl;
    if (getch() == 'y') {
        try {
            std::filesystem::remove_all(dir_path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to delete directory " << dir << ": " << e.what() << std::endl;
            return 1;
        }
    }
	else
    	std::cout << "backup directory remained" << std::endl;
    return 0;
}

std::string user;

std::string c_col(int c1, int c2){
	if(!c1)
		c1 = 37;
	if(!c2)
		c2 = 40;
	std::string res = "\033[" + std::to_string(c1) + "m\033[" + std::to_string(c2) + "m";
	return res;
}

void cls(){
	#if defined(__unix__) || defined(__APPLE__)
	system("clear");
	#else
	system("cls");
	#endif
	return;
}

char* date(){
	time_t t = time(0);
	char* dt = ctime(&t);
	return dt;
}

std::string head(bool ingame = false, bool dont = false){
	if(!dont){
		if(!ingame)
			cls();
		else
			printer.cls();
	}
	std::string res = c_col(32, 40);
	res += "StrikeForce\n";
	res += "Created by: 21\n" + c_col(37, 40);
	res += "____________________________________________________\n";
	if(user.size()){
		res += c_col(34, 40) + "~ " + user + "\n";
		res += c_col(37, 40) + "____________________________________________________\n";
	}
	res += c_col(36, 40) + "Local time: " + date() + "\n";
	res += c_col(37, 40);
	return res;
}

int my_recv(int sock, char* buffer, const int buffer_size, int flags){
	int len = 0;
	while(len < buffer_size){
		if(recv(sock, buffer, 1, 0) < 0)
            return -1;
		if(!(*buffer))
			return len;
		++buffer, ++len;
	}
	return len;
}

namespace Environment{

	namespace Random{

	}

	namespace Item{

	}

	namespace Character{

	}

	namespace Field{

	}
}