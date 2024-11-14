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
#pragma GCC optimize("Ofast")

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

#if defined(__unix__) || defined(__APPLE__)
#include <sys/select.h>
#include <termios.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>

int constexpr BK = 127, EN = 10;

bool kbhit(){
	struct termios oldt, newt;
	int ch, oldf;
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
	ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);
	if(ch != EOF){
		ungetc(ch, stdin);
		return true;
	}
	return false;
}

int getch(){
	struct termios oldt, newt;
	int ch;
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	return ch;
}

#else
#include <conio.h>

#include "win_arpa_inet.hpp"

void usleep(int x){
	Sleep((x + 500) / 1000);
	return;
}

int constexpr BK = 8, EN = 13;
#endif

std::string user;

std::string c_col(int c1, int c2, bool b = true){
	std::string res = "\033[" + std::to_string(c1) + "m\033[" + std::to_string(c2) + "m";
	if(b)
		std::cout << res;
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

std::string head(bool b = true){
	cls();
	std::string res = c_col(32, 40, false);
	res += "StrikeForce\n";
	res += "Created by: 21\n" + c_col(37, 40, false);
	res += "____________________________________________________\n";
	if(user.size()){
		res += c_col(34, 40, false) + "~ " + user + "\n";
		res += c_col(37, 40, false) + "____________________________________________________\n";
	}
	res += c_col(36, 40, false) + "Local time: " + date() + "\n";
	res += c_col(37, 40, false);
	if(b)
		std::cout << res;
	return res;
}

int damage(int x, int y){
    int l = 0, r = x + 1, z = 2;
    while(1 < y){
        y >>= 1;
        ++z;
    }
    while(r - l > 1){
        int mid = (l + r) >> 1, tmp = x;
        for(int i = 0; i < z && mid; ++i)
            tmp /= mid;
        if(tmp)
            l = mid;
        else
            r = mid;
    }
    return l;
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
