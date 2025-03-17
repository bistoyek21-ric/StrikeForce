#include<bits/stdc++.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
using namespace std;

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
	disable_input_buffering();
	struct timeval tv = {0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    int res = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
	restore_input_buffering();
	return res;
}

int getch(){
	disable_input_buffering();
    int res = getchar();
	restore_input_buffering();
	return res;
}

int main(){
    while (1) {
        cout << getch() << '\n';
    }
    return 0;
}