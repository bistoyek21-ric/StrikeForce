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

class SHA256 {
public:
    static const size_t HASH_SIZE = 32;

    std::string hash(std::string s){
		reset();
        auto input = s.c_str();
        uint8_t hash[HASH_SIZE] = {};
        update(reinterpret_cast<const uint8_t*>(input), strlen(input));
        finalize(hash);
        std::string hashed = "";
        for(int i = 0; i < HASH_SIZE; ++i)
            hashed += (char)hash[i];
        return hashed;
    }

    void update(const uint8_t* data, size_t length) {
        size_t i = 0;
        while (i < length) {
            size_t chunk_size = std::min(BLOCK_SIZE - buffer_size, length - i);
            std::memcpy(buffer + buffer_size, data + i, chunk_size);
            buffer_size += chunk_size;
            i += chunk_size;
            if (buffer_size == BLOCK_SIZE) {
                process_block(buffer);
                buffer_size = 0;
            }
        }
    }

    void finalize(uint8_t* hash) {
        buffer[buffer_size] = 0x80;
        if (buffer_size >= BLOCK_SIZE - 8) {
            std::memset(buffer + buffer_size + 1, 0, BLOCK_SIZE - buffer_size - 1);
            process_block(buffer);
            buffer_size = 0;
        }
        std::memset(buffer + buffer_size + 1, 0, BLOCK_SIZE - buffer_size - 9);
        uint64_t total_bits = total_length * 8;
        for (int i = 0; i < 8; ++i)
            buffer[BLOCK_SIZE - 1 - i] = total_bits >> (8 * i);
        process_block(buffer);
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                hash[i * 4 + j] = (state[i] >> (24 - j * 8)) & 0xFF;
            }
        }
    }

private:
    static const size_t BLOCK_SIZE = 64;

    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t state[8];
    uint8_t buffer[BLOCK_SIZE];
    size_t buffer_size;
    uint64_t total_length;

    void reset() {
        state[0] = 0x6a09e667;
        state[1] = 0xbb67ae85;
        state[2] = 0x3c6ef372;
        state[3] = 0xa54ff53a;
        state[4] = 0x510e527f;
        state[5] = 0x9b05688c;
        state[6] = 0x1f83d9ab;
        state[7] = 0x5be0cd19;
        buffer_size = 0;
        total_length = 0;
    }

    void process_block(const uint8_t* block) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | block[i * 4 + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = right_rotate(w[i - 15], 7) ^ right_rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = right_rotate(w[i - 2], 17) ^ right_rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g;
			g = f;
			f = e;
            e = d + temp1;
            d = c;
			c = b;
			b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
        total_length += BLOCK_SIZE;
    }

    uint32_t right_rotate(uint32_t value, uint32_t count) {
        return (value >> count) | (value << (32 - count));
    }

} sha256;

std::string hs(std::string &s){
	return sha256.hash(s);
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
		std::cout << head();
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
		if(s1 == "sign up")
			continue;
		for(auto &e: s1)
			if(e == ' ')
				e = '_';
		int ind = get_ind(users, s1);
		if(ind == -1 && s1 != "")
			break;
	}
	for(bool b = false; true; b = true){
		std::cout << head();
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
		std::cout << std::endl;
		std::cout << "confirm passwaord: ";
		std::cout.flush();
		psw(s3);
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
		std::cout << head();
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
		for(auto &e: s1)
			if(e == ' ')
				e = '_';
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
			std::cout << head();
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
