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
#include "menu.hpp"

namespace Environment::Character{

	void init(){
		std::ofstream info("./accounts/game/" + user + "/info, " + user + ".txt");
		std::ifstream human("./character/human.txt");
	    std::string s;
	    info << user << '\n';
		while(human >> s)
			info << s << '\n';
	    info.close();
	    info.open("./accounts/game/" + user + "/" + user + ".txt");
	    info.close();
		return;
	}

	void recovery(){
        me.build();
        return;
	}
}

using ll = long long;

void sign_in(std::vector<std::pair<std::string, std::string>> &users);
void sign_up(std::vector<std::pair<std::string, std::string>> &users);

std::string hs(std::string &s){
	ll bs[5] = {259, 258, 257, 256, 263};
	ll md[5] = {1000000021, 1000000009, 1000000007, 998244353, 2000000011};
	ll sum = 0;
	std::string res = "", st;
	for(int k = 0; k < 5; ++k, sum = 0, res += st){
		for(int i = 0; i < s.size(); ++i){
			sum = (sum * bs[k]) % md[k];
			sum += s[i];
			sum %= md[k];
			st = std::to_string((int)sum);
		}
		for(int j = 0; j < 11 - st.size(); ++j)
			res.push_back('0');
	}
	return res;
}

int get_ind(std::vector<std::pair<std::string, std::string>> &users, std::string &s){
	for(int i = 0; i < users.size(); ++i)
		if(users[i].first == s)
			return i;
	return -1;
}

void psw(std::string &s){
	s = "";
	for(bool b = false; true; b = true){
		char c = getch();
		if(c == EN)
			return;
		else if(c == BK){
			if(s.size()){
				s.pop_back();
				std::cout << "\b \b";
				std::cout.flush();
			}
		}
		else{
			std::cout << c;
			std::cout.flush();
			usleep(100000);
			std::cout << "\b*";
			std::cout.flush();
			s.push_back(c);
		}
	}
	return;
}

void sign_up(std::vector<std::pair<std::string, std::string>> &users){
	std::string s1, s2, s3;
	for(bool b = false; true; b = true){
		head();
		std::cout << "If you have an account and want to sing in write \"sign in\" and press enter" << '\n';
		std::cout << "-------------------" << std::endl;
		if(b && s1 != "")
			std::cout << "username is taken" << std::endl;
		if(b && s1 == "")
			std::cout << "username can't be null" << std::endl;
        if(b && s1 == "sign up")
			std::cout << "username can't be sign up or sign in" << std::endl;
		std::cout << "username: ";
		std::cout.flush();
		getline(std::cin, s1);
		if(s1 == "sign in"){
			sign_in(users);
			return;
		}
		int ind = get_ind(users, s1);
		if(ind == -1 && s1 != "")
			break;
	}
	for(bool b = false; true; b = true){
		head();
		std::cout << "If you have an account and want to sing in write \"sing in\" and press enter" << '\n';
		std::cout << "-------------------" << std::endl;
		if(b){
			std::cout << "password isn't match" << '\n';
			std::cout << "-------------------" << std::endl;
		}
		std::cout << "username: " << s1 << std::endl;
		std::cout << "password: ";
		std::cout.flush();
		psw(s2);
		if(s2 == "sign in"){
			sign_in(users);
			return;
		}
		std::cout << std::endl;
		std::cout << "confirm passwaord: ";
		std::cout.flush();
		psw(s3);
		if(s3 == "sign in"){
			sign_in(users);
			return;
		}
		if(s2 == s3)
			break;
	}
	std::ofstream us("./accounts/users.txt", std::ios::app);
	std::ofstream ps("./accounts/pass.txt", std::ios::app);
	user = s1;
	us << user << '\n';
	ps << hs(s2) << '\n';
	us.close();
	ps.close();
	#if defined(__unix__) || defined(__APPLE__)
        system(("mkdir ./accounts/game/" + user).c_str());
	#else
        system(("mkdir .\\accounts\\game\\" + user).c_str());
	#endif
	Environment::Character::init();
	std::string modes[4] = {"", "solo", "timer", "squad"};
	for(int i = 0; i < 4; ++i){
		std::ifstream rank("./accounts/ranking" + modes[i] + ".txt");
		std::vector<std::string> rnk;
		while(getline(rank, s3))
			rnk.push_back(s3);
		rank.close();
		std::ofstream rank1("./accounts/ranking" + modes[i] + ".txt");
		rnk.push_back(s1);
		rnk.push_back("0");
		for(auto &e: rnk)
			rank1 << e << '\n';
		rank1.close();
	}
	return;
}

void sign_in(std::vector<std::pair<std::string, std::string>> &users){
	std::string s1, s2;
	for(bool b = false; true; b = true){
		head();
		std::cout << "If want to create account write \"sign up\" and press enter" << '\n';
		std::cout << "-------------------" << std::endl;
		if(b)
			std::cout << "username or password is in correct" << std::endl;
		std::cout << "username: ";
		std::cout.flush();
		getline(std::cin, s1);
		if(s1 == "sign up"){
			sign_up(users);
			return;
		}
		std::cout << "password: ";
		std::cout.flush();
		psw(s2);
		if(s2 == "sign up"){
			sign_up(users);
			return;
		}
		int ind = get_ind(users, s1);
		if(ind != -1 && users[ind].first == s1 && users[ind].second == hs(s2)){
			user = s1;
			return;
		}
	}
	return;
}

void enter(){
	while(true){
		std::ifstream sli("./accounts/saved_login.txt");
		std::string l1;
		int l2;
		getline(sli, l1);
		sli >> l2;
		if(l1.size() && time(0) - l2 <= 7 * 60 * 24 * 60){
			user = l1;
			Environment::Character::recovery();
			Environment::Character::menu();
			continue;
		}
		sli.close();
		std::ifstream us("./accounts/users.txt");
		std::ifstream ps("./accounts/pass.txt");
		std::vector<std::pair<std::string, std::string>> users;
		std::string s;
		while(getline(us, s)){
			users.push_back({});
			users.back().first = s;
			getline(ps, s);
			users.back().second = s;
		}
		us.close();
		ps.close();
		for(bool b = false; true; b = true){
			head();
			std::cout << "Have an account? (y/n)" << std::endl;
			if(b)
				std::cout << "invalid input, try again" << std::endl;
			getline(std::cin, s);
			if(s == "y" || s == "n")
				break;
		}
		if(s[0] == 'y')
			sign_in(users);
		else
			sign_up(users);
    	std::cout << "\nDo you want to save your login for a week? (1:yes, any other key:no)" << std::endl;
    	char c = getch();
	    if(c == '1'){
    		std::ofstream slo("./accounts/saved_login.txt");
	    	slo << user << '\n';
    		slo << time(0) << '\n';
		}
		Environment::Character::recovery();
		Environment::Character::menu();
	}
	return;
}
