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
#include "Character.hpp"

int rand_(){
	return rand();
}

namespace Environment::Field{

	int rand(){
		return Environment::Random::_rand();
	}

	int constexpr F = 3, N = 29, M = 100, H = 9000, Z = 9000, B = 9000, C = 9000, lim_portal = 1000, lim_block = 1100;

	int ind, countdown;

	bool disconnect;

	char command[H], symbol[8][4] = {{'V', '>', 'A', '<'}, {'z','Z'}, {'*'}, {'#'}, {'?'}, {'^'}, {'v'}, {'O'}};

	int agent[H];

	const std::string valid_commands = "`13upxawsdfghjkl;'cvbnm,./[]+";

	Environment::Item::Bullet bull[B];
	Environment::Character::Zombie zomb[Z];
	Environment::Character::Human hum[H];

	std::vector<int> portal[B];

	std::bitset<B> mb, active;
	std::bitset<Z> mz;
	std::bitset<H> mh, remote;

	std::chrono::time_point<std::chrono::steady_clock> start;

	class Client{

	public:
		bool open = false;

		const int BUFFER_SIZE = 2048, BS = 2;
		int n, team;
		long long tb, serial_number;

		void start(const std::string& server_ip, int server_port, const std::string& server_password){
			open = true;
			#if !defined(__unix__) && !defined(__APLLE__)
			WSADATA wsaData;
			if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0){
				std::cout << "WSAStartup failed" << std::endl;
				disconnect = true;
				return;
            }
			#endif
			server_addr.sin_family = AF_INET;
			server_addr.sin_port = htons(server_port);
			inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);
			sock = socket(AF_INET, SOCK_STREAM, 0);
			if(sock == -1){
				std::cout << "Failed to create socket" << std::endl;
				disconnect = true;
				return;
			}
			if(connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1){
				std::cout << "Failed to connect to server" << std::endl;
				disconnect = true;
				return;
			}
			send(sock, server_password.c_str(), server_password.size() + 1, 0);
			char buffer[BUFFER_SIZE];
			memset(buffer, 0, BUFFER_SIZE);
			my_recv(sock, buffer, BUFFER_SIZE, 0);
			if(buffer[0] == 'R'){
				std::cout << "Wrong password entered" << std::endl;
				disconnect = true;
				return;
			}
			memset(buffer, 0, BUFFER_SIZE);
			my_recv(sock, buffer, BUFFER_SIZE, 0);
			sscanf(buffer, "%lld %lld", &tb, &serial_number);
			memset(buffer, 0, BUFFER_SIZE);
			my_recv(sock, buffer, BUFFER_SIZE, 0);
			sscanf(buffer, "%d %d %d", &n, &ind, &team);
			Environment::Character::me.backpack.vec = -1;
			hum[ind] = Environment::Character::me;
			hum[ind].set_team(team);
			mh[ind] = true;
			remote[ind] = false;
			return;
		}

		void send_it(){
			std::string msg;
			msg.push_back(command[ind]);
			send(sock, msg.c_str(), 2, 0);
			return;
		}

		void give_info(){
			std::string data = "", s;
			bool b = false;
			std::ifstream f("./accounts/game/" + user + "/info, " + user + ".txt");
			while(getline(f, s)){
				if(b)
					data += '\n';
				data += s;
				b = true;
			}
			send(sock, data.c_str(), data.size() + 1, 0);
			return;
		}

		void get_info(){
			for(int i = 0; i < n; ++i){
				if(i == ind)
					continue;
				char buffer[BUFFER_SIZE];
				memset(buffer, 0, BUFFER_SIZE);
				my_recv(sock, buffer, BUFFER_SIZE, 0);
				hum[i].scan(buffer);
				memset(buffer, 0, BUFFER_SIZE);
				my_recv(sock, buffer, BUFFER_SIZE, 0);
				int t;
				sscanf(buffer, "%d", &t);
				hum[i].set_team(t);
				mh[i] = true;
				remote[i] = true;
			}
			return;
		}

		void prepare(){
		    struct timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            #if defined(__unix__) || defined(__APPLE__)
            auto t = &timeout;
            #else
            int ms = 1000;
            char* t = (char*)&ms;
            #endif
            if(setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, t, sizeof(timeout)) < 0){
                std::cout << "Error setting socket options" << '\n';
                exit(1);
            }
            return;
		}

		void recieve(){
			char buffer[BS];
			for(int i = 0; i < ind; ++i){
				if(!mh[i])
					continue;
				memset(buffer, 0, BS);
				if(my_recv(sock, buffer, BS, 0) < 0){
                    disconnect = true;
                    return;
                }
				sscanf(buffer, "%c", &command[i]);
			}
			for(int i = ind + 1; i < n; ++i){
				if(!mh[i])
					continue;
				memset(buffer, 0, BS);
				if(my_recv(sock, buffer, BS, 0) < 0){
                    disconnect = true;
                    return;
                }
				sscanf(buffer, "%c", &command[i]);
			}
			return;
		}

        void end_it(){
        	close(sock);
			#if !defined(__unix__) && !defined(__APPLE__)
			WSACleanup();
			#endif
			open = false;
			return;
        }

    	private:
       		int sock;
		struct sockaddr_in server_addr;
	} client;

	int p_ind(){
		for(int i = 0; i < B; ++i)
			if(!active[i])
				return i;
		return -1;
	}

	int h_ind(){
		for(int i = 0; i < H; ++i)
			if(i != ind && !mh[i] && !remote[i] && agent[ind] == -1)
				return i;
		return -1;
	}

	int z_ind(){
		for(int i = 0; i < Z; ++i)
			if(!mz[i])
				return i;
		return -1;
	}

	int b_ind(){
		for(int i = 0; i < B; ++i)
			if(!mb[i])
				return i;
		return -1;
	}

	struct node{
		std::bitset<11> s;
		int dmg = 0, portal_ind = -1;
		Environment::Character::Human* human = nullptr;
		Environment::Character::Zombie* zombie = nullptr;
		Environment::Item::Bullet* bullet = nullptr;
		Environment::Item::ConsumableItem* cons = nullptr;

		std::string showit_() const{
			std::string ans = "";
			if(s[3]){
				if(s[10]){
					if(s[9])
						ans += c_col(35, 47) + "#";
					else
						ans += c_col(35, 40) + "#";
				}
				else
					ans += c_col(0, 0) + "#";
				return ans;
			}
			if(s[0]){
				if(!s[9]){
					if(human == &hum[ind])
						ans += c_col(32, 40);
					else if(human->is_rnpc())
						ans += c_col(31, 40);
					else if(human->get_team() != hum[ind].get_team())
						ans += c_col(35, 40);
					else
						ans += c_col(34, 40);
					ans += symbol[0][human->get_way() - 1];
					return ans;
				}
				if(human == &hum[ind])
					ans += c_col(32, 47);
				else if(human->is_rnpc())
					ans += c_col(31, 47);
				else if(human->get_team() != hum[ind].get_team())
					ans += c_col(35, 47);
				else
					ans += c_col(34, 47);
				ans += symbol[0][human->get_way() - 1];
				ans += c_col(0, 0);
				return ans;
			}
			if(s[1]){
				if(!s[9]){
					ans += c_col(31, 40);
					ans += symbol[1][zombie->is_super()];
					return ans;
				}
				ans += c_col(31, 47);
				ans += symbol[1][zombie->is_super()];
				ans += c_col(0, 0);
				return ans;
			}
			if(s[5]){
				if(s[10]){
					if(s[9])
						ans += c_col(35, 47) + "^";
					else
						ans += c_col(35, 40) + "^";
				}
				else
					ans += c_col(0, 0) + "^";
				return ans;
			}
			if(s[6])
				return c_col(0, 0) + "v";
			if(s[2])
				return c_col(35, 40) + "*";
			if(s[4])
				return c_col(33, 40) + "?";
			if(s[8])
				return c_col(0, 0) + "X";
			if(s[7]){
				if(s[10])
					return c_col(35, 40) + "O";
				return c_col(32, 40) + "O";
			}
			return c_col(0, 0) + ".";
		}

		char showit() const{
			if(s[3])
				return '#';
			if(s[0])
				return symbol[0][human->get_way() - 1];
			if(s[1])
				return symbol[1][zombie->is_super()];
			if(s[5])
				return '^';
			if(s[6])
				return 'v';
			if(s[2])
				return '*';
			if(s[4])
				return '?';
			if(s[8])
				return 'X';
			if(s[7])
				return 'O';
			return '.';
		}

		void update(){
			s[8] = s[9] = 0;
			return;
		}
	};

	const node nd;

	struct gameplay{
		bool is_human, online, silent, quit, full, manual;
		Environment::Character::Zombie* recomZ;
		Environment::Character::Human* recomH;

		const int L = 9, pc = 30, pz = 40, ph = 50, wdx[4] = {1, 0, -1, 0}, wdy[4] = {0, 1, 0, -1};
		long long loot, level, teams_kills, kills, chest, frame, serial_number;
		int W, _H;

		std::vector<node*> temp;

		std::vector<int> place[2 * B];

		std::string mode;
		time_t tb;

		node themap[F][N][M], themap1[F][N][M];

		void view() const;

		char bot(Environment::Character::Human& player) const;

		void print_game() const;

		char human_rnpc_bot(Environment::Character::Human& player) const;

		void updmap(){
			for(int i = 0; i < F; ++i)
                for(int j = 0; j < N; ++j)
                    for(int k = 0; k < M; ++k)
                        themap[i][j][k].update();
            return;
        }

		void load_data(){
			if(!manual)
				agent[ind] = 0;
			srand(tb);
			serial_number = (rand_() & 1023) + ((rand_() & 1023) << 10) + ((rand_() & 1023) << 20);
			Environment::Random::_srand(tb, serial_number);
			if(online){
				disconnect = false;
				std::string server_ip;
				int server_port = 0;
				std::cout << "Enter the server IP: ";
				std::cout.flush();
				getline(std::cin, server_ip);
				std::cout << "Enter the server port: ";
				std::cout.flush();
				std::string server_port_s;
				getline(std::cin, server_port_s);
				std::cout << "Enter the server's password: ";
				std::cout.flush();
				std::string server_password;
				getline(std::cin, server_password);
				for(auto e: server_port_s)
					server_port = 10 * server_port + (e - '0');
				server_port = std::max(std::min(server_port, (1 << 16) - 1), 0);
				client.start(server_ip, server_port, server_password);
				if(disconnect){
					std::cout << "press space button to continue" << std::endl;
					while(getch() != ' ');
					return;
				}
				client.give_info();
				client.get_info();
				serial_number = client.serial_number;
				tb = time(nullptr);
				Environment::Random::_srand(client.tb, serial_number);
				for(int i = 0; i < client.n; ++i){
					hum[i].set_way(rand() % 4 + 1);
					while(true){
						std::vector<int> v = {rand() % F, rand() % N, rand() % M};
						if(themap[v[0]][v[1]][v[2]].showit() == '.'){
							themap[v[0]][v[1]][v[2]].human = &hum[i];
							themap[v[0]][v[1]][v[2]].s[0] = 1;
							hum[i].set_cor(v);
							break;
						}
					}
				}
				return;
			}
			if(mode == "Squad"){
				ind = 0;
				mh[ind] = true;
				hum[ind] = Environment::Character::me;
				remote[ind] = false;
				themap[0][3][1].human = &hum[ind];
				themap[0][3][1].s[0] = 1;
				hum[ind].set_way(1);
				hum[ind].set_cor(std::vector<int>{0, 3, 1});
				hum[ind].set_team(1);
				for(int i = 1; i < 5; ++i){
					mh[i] = true;
					remote[i] = false;
					std::string s = "team mate ";
					s += (char)('0' + i);
					gen_human(false, hum[i], level, std::vector<int>{0, 1, i + 1}, s);
					themap[0][1][i + 1].human = &hum[i];
					themap[0][1][i + 1].s[0] = 1;
					hum[i].set_team(1);
				}
				for(int i = 5; i < 10; ++i){
					mh[i] = true;
					remote[i] = false;
					std::string s = "opponent ";
					s += (char)('0' + i - 4);
					gen_human(false, hum[i], level, std::vector<int>{2, 1, i + 1}, s);
					themap[2][1][i + 1].human = &hum[i];
					themap[2][1][i + 1].s[0] = 1;
					hum[i].set_team(2);
				}
				return;
			}
			if(mode == "Solo" || mode == "Timer"){
				ind = 0;
				mh[ind] = true;
				remote[ind] = false;
				hum[ind] = Environment::Character::me;
				themap[0][1][1].human = &hum[ind];
				themap[0][1][1].s[0] = 1;
				hum[ind].set_way(1);
				hum[ind].set_cor(std::vector<int>{0, 1, 1});
				hum[ind].set_team(1);
				return;
			}
			return;
		}

		bool rivals_are_dead(){
			for(int i = 0; i < H; ++i)
				if(mh[i]){
					int team = hum[i].get_team();
					if(team && team != hum[ind].get_team())
						return false;
                }
			return true;
        }

		void claim_chest(Environment::Character::Human& player){
			std::vector<int> v = player.get_cor();
			if(themap[v[0]][v[1]][v[2]].s[4]){
        		player.claim_chest(*(themap[v[0]][v[1]][v[2]].cons));
				themap[v[0]][v[1]][v[2]].s[4] = 0;
				--chest;
			}
			return;
		}

		void teleport(Environment::Character::Human& player){
			std::vector<int> v = player.get_cor();
			int index = themap[v[0]][v[1]][v[2]].portal_ind;
			if(index == -1)
				return;
			char sit = themap[portal[index][0]][portal[index][1]][portal[index][2]].showit();
			if(sit != 'O')
				return;
			themap[portal[index][0]][portal[index][1]][portal[index][2]].s[0] = 1;
			themap[portal[index][0]][portal[index][1]][portal[index][2]].human = &player;
			themap[v[0]][v[1]][v[2]].s[0] = 0;
			player.set_cor(portal[index]);
			return;
		}

		void spawn_chest(){
			if(C <= chest)
				return;
			int i = rand() % F, j = rand() % N, k = rand() % M;
			if(themap[i][j][k].showit() != '.')
				return;
			themap[i][j][k].cons = Environment::Item::gen_item(rand() % 4);
			themap[i][j][k].s[4] = 1;
			++chest;
			return;
		}

		void spawn_zombie_npc(){
			int i = rand() % F, j = rand() % N, k = rand() % M;
			if(themap[i][j][k].showit() != '.')
				return;
			int index = z_ind();
			if(index == -1)
				return;
			bool super = (rand() % 4 == 0);
			Environment::Character::gen_zombie(zomb[index], super, std::vector<int>{i, j, k}, (super ? "SZ" : "Z") + std::to_string(frame));
			themap[i][j][k].zombie = &zomb[index];
			themap[i][j][k].s[1] = 1;
			mz[index] = true;
			return;
		}

		void spawn_human_npc(){
			int i = rand() % F, j = rand() % N, k = rand() % M;
			if(themap[i][j][k].showit() != '.')
				return;
			int index = h_ind();
			if(index == -1)
				return;
			Environment::Character::gen_human(true, hum[index], level, std::vector<int>{i, j, k}, "H" + std::to_string(frame));
			themap[i][j][k].human = &hum[index];
			themap[i][j][k].s[0] = 1;
			remote[index] = false;
			mh[index] = true;
			return;
		}

		void zombie_damage(node* pix){
			pix->s[9] = 1;
			pix->zombie->hit(*(pix->bullet));
			pix->s[2] = 0;
			Environment::Character::Human* owner = reinterpret_cast<Environment::Character::Human*>(pix->bullet->get_owner());
			mb[pix->bullet - bull] = false;
			if(pix->zombie->get_Hp() <= 0){
				mz[pix->zombie - zomb] = false;
				pix->s[8] = 1;
				pix->s[1] = 0;
				if(owner && owner->get_team() == hum[ind].get_team()){
					int pts = 500 + 250 * (pix->zombie->is_super());
					++teams_kills, loot += pts / 10;
					if(owner == &hum[ind])
						loot += pts * 9 / 10, ++kills;
				}
			}
			return;
		}

		void hit_zombie(){
			for(int i = 0; i < Z; ++i)
				if(mz[i]){
					std::vector<int> v = zomb[i].get_cor();
					auto pix = &themap[v[0]][v[1]][v[2]];
					if(pix->s[2])
						zombie_damage(pix);
				}
			return;
		}

		void human_damage(node* pix){
			pix->s[9] = 1;
			pix->human->hit(*(pix->bullet));
			pix->s[2] = 0;
			Environment::Character::Human* owner = reinterpret_cast<Environment::Character::Human*>(pix->bullet->get_owner());
			mb[pix->bullet - bull] = false;
			if(pix->human->get_Hp() <= 0 && pix->human != &hum[ind]){
				mh[pix->human - hum] = false;
				agent[pix->human - hum] = -1;
				pix->s[8] = 1;
				pix->s[0] = 0;
				if(owner && owner->get_team() == hum[ind].get_team() && pix->human->get_team() != hum[ind].get_team()){
					++teams_kills, loot += 100;
					if(owner == &hum[ind])
						loot += 900, ++kills;
				}
			}
			return;
		}

		void hit_human(){
			for(int i = 0; i < H; ++i)
		        if(mh[i]){
	        		std::vector<int> v = hum[i].get_cor();
	       			auto pix = &themap[v[0]][v[1]][v[2]];
					if(hum[i].get_Hp() <= 0){
						mh[i] = false;
						agent[i] = -1;
						pix->s[8] = 1;
						pix->s[0] = 0;
					}
	       			else if(pix->s[2])
    	       			human_damage(pix);
				}
        	return;
		}

		void zombie_action(){
			node* pix;
			for(int _ = 0; _ < Z; ++_)
				if(mz[_]){
					std::vector<int> v = zomb[_].get_cor();
					int i = v[0], j = v[1], k = v[2];
					if(themap[i][j][k].s[2])
						continue;
					bool b = false;
					for(int i1 = 0; i1 < 4; ++i1){
						if(themap[i][wdx[i1] + j][wdy[i1] + k].s[0]){
							pix = &themap[i][wdx[i1] + j][wdy[i1] + k];
							int index = b_ind();
							std::vector<int> v = {i, j + wdx[i1], k + wdy[i1]};
							if(!pix->s[2] && index != -1){
								themap[i][j][k].zombie->punch(bull[index], i1);
								pix->bullet = &bull[index];
								pix->s[2] = 1;
								mb[index] = true;
							}
							b = true;
						}
					}
					if(b == false){
						if(rand() % 5 < 2)
							continue;
						for(int i1 = 0; i1 < 2; ++i1){
							int i2 = rand() % 4;
							if(wdx[i2] + j != 1 && themap[i][wdx[i2] + j][wdy[i2] + k].showit() == '.'){
								themap[i][j][k].zombie->set_cor({i, wdx[i2] + j, wdy[i2] + k});
								std::swap(themap[i][j][k], themap[i][wdx[i2] + j][wdy[i2] + k]);
								break;
							}
						}
					}
				}
			return;
		}

		void obey(const char c, Environment::Character::Human &player){
		    if(c == '_'){
                player.set_Hp(0);
                return;
		    }
		    if(c == '[' || c == ']'){
                std::vector<int> v = player.get_cor();
                int d = player.get_way() - 1;
				v[1] += wdx[d], v[2] += wdy[d];
				if(v[1] >= N || 0 > v[1] || v[2] >= M || 0 > v[2])
					return;
                if(themap[v[0]][v[1]][v[2]].showit() != '.')
                    return;
                if(c == '['){
                    if(player.backpack.get_blocks()){
                        themap[v[0]][v[1]][v[2]].s[10] = themap[v[0]][v[1]][v[2]].s[3] = 1;
                        player.backpack.use_block();
                        temp.push_back(&themap[v[0]][v[1]][v[2]]);
                    }
                    return;
                }
                else{
                    if(~player.backpack.get_portal_ind()){
                    	themap[v[0]][v[1]][v[2]].s[10] = themap[v[0]][v[1]][v[2]].s[5] = 1;
                        themap[v[0]][v[1]][v[2]].portal_ind = player.backpack.get_portal_ind();
                        player.backpack.set_portal_ind(-1);
                        temp.push_back(&themap[v[0]][v[1]][v[2]]);
                    }
                    else if(player.backpack.get_portals()){
                    	int index = p_ind();
                    	if(index == -1)
                    		return;
                    	themap[v[0]][v[1]][v[2]].s[10] = themap[v[0]][v[1]][v[2]].s[7] = 1;
                    	player.backpack.use_portal();
                    	player.backpack.set_portal_ind(index);
                    	portal[index] = v;
                    	active[index] = 1;
                    	temp.push_back(&themap[v[0]][v[1]][v[2]]);
                    }
                    return;
                }
                return;
		    }
			if(c == '1' || c == '`'){
				(c == '1' ? player.turn_r() : player.turn_l());
				return;
			}
			if(c == 'a' || c == 's' || c == 'd' || c == 'w'){
				int i = 0;
				char s[4] = {'s', 'd', 'w', 'a'};
				while(c != s[i])
					++i;
				std::vector<int> v = player.get_cor();
				if(v[1] + wdx[i] >= N || 0 > v[1] + wdx[i] || v[2] + wdy[i] >= M || 0 > v[2] + wdy[i])
					return;
				char sit = themap[v[0]][v[1] + wdx[i]][v[2] + wdy[i]].showit();
				if(sit == '?' || sit == '^' || sit == 'v' || sit == '.' || sit == 'X' || sit == '*'){
					themap[v[0]][v[1] + wdx[i]][v[2] + wdy[i]].s[0] = 1;
					themap[v[0]][v[1] + wdx[i]][v[2] + wdy[i]].human = &player;
					themap[v[0]][v[1]][v[2]].s[0] = 0;
					player.set_cor(std::vector<int>{v[0], v[1] + wdx[i], v[2] + wdy[i]});
				}
				return;
			}
			if(c == 'f' || c == 'g' || c == 'h' || c == 'j'){
				int i = 0;
				char s[4] = {'f', 'g', 'h', 'j'};
				while(c != s[i])
					++i;
				if(!player.backpack.list_cons[i].second)
					return;
				player.backpack.vec = 0;
				player.backpack.ind = i;
				return;
			}
			if(c == 'k' || c == 'l' || c == ';' || c == '\''){
				int i = 0;
				char s[4] = {'k', 'l', ';', '\''};
				while(c != s[i])
					++i;
				if(!player.backpack.list_throw[i].second.second)
					return;
				player.backpack.vec = 1;
				player.backpack.ind = i;
				return;
			}
			if(c == 'c' || c == 'v' || c == 'b' || c == 'n' || c == 'm' || c == ',' || c == '.' || c == '/'){
				int i = 0;
				char s[8] = {'c', 'v', 'b', 'n', 'm', ',', '.', '/'};
				while(c != s[i])
					++i;
				if(!player.backpack.list_w[i].second)
					return;
				player.backpack.vec = 2;
				player.backpack.ind = i;
				return;
			}
			if(c == 'u'){
				player.use(player.backpack.list_cons[player.backpack.ind].first);
				return;
			}
			if(c == 'p' || c == 'x'){
				int bway = player.get_way() - 1;
				std::vector<int> v = player.get_cor();
				v[1] += wdx[bway], v[2] += wdy[bway];
				int index = b_ind();
				if(index == -1 || v[1] >= N || 0 > v[1] || v[2] >= M || 0 > v[2])
					return;
				bool can;
				if(c == 'p')
					can = player.punch(bull[index]);
				else if(player.backpack.vec == 1)
					can = player.throw_it(bull[index]);
				else if(player.backpack.vec == 2)
					can = player.shot_it(bull[index]);
				else
					return;
				char sit = themap[v[0]][v[1]][v[2]].showit();
				if(can && ((sit != '#' && sit != 'v' && sit != '^') || themap[v[0]][v[1]][v[2]].s[10])){
					themap[v[0]][v[1]][v[2]].bullet = &bull[index];
					themap[v[0]][v[1]][v[2]].s[2] = 1;
					mb[index] = true;
				}
				return;
			}
			return;
		}

		void my_command(){
			if(kbhit()){
				command[ind] = getch();
				if(command[ind] == '8')
					command[ind] = 'w';
				else if(command[ind] == '4')
					command[ind] = 'a';
				else if(command[ind] == '6')
					command[ind] = 'd';
				else if(command[ind] == '2')
					command[ind] = 's';
				if(agent[ind] != -1){
					if(!manual && command[ind] == ' ')
						silent = !silent;
					if(command[ind] == '3')
						manual = !manual;
					else if(!manual){
						command[ind] = '+';
						return;
					}
				}
				silent = false;
			}
			else{
				command[ind] = '+';
				return;
			}
			if(command[ind] == 'Q'){
				quit = true;
				return;
			}
	        if(command[ind] == '0'){
        		silent = online;
				command_list(online);
				command[ind] = '+';
				return;
			}
	        if(command[ind] == '-'){
				silent = online;
				hum[ind].show_backpack(silent, true);
				command[ind] = '+';
				return;
	        }
        	if(command[ind] == 'F'){
				full = true;
				command[ind] = '+';
				return;
        	}
			if(command[ind] == 'O'){
				full = false;
				command[ind] = '+';
				return;
			}
			if(command[ind] == 'W' || command[ind] == 'E'){
				if(command[ind] == 'W')
					++W;
				else
					--W;
				W = std::min(M / 2 - 1, std::max(W, 0));
                command[ind] = '+';
				return;
			}
			if(command[ind] == 'R' || command[ind] == 'T'){
				if(command[ind] == 'R')
					++_H;
				else
					--_H;
				_H = std::min(N / 2 - 1, std::max(_H, 0));
				command[ind] = '+';
				return;
			}
			for(const char &e: valid_commands)
				if(e == command[ind])
					return;
			command[ind] = '+';
			return;
		}

		void get_command(int i){
			if(hum[i].is_rnpc())
				command[i] = human_rnpc_bot(hum[i]);
			else
				command[i] = bot(hum[i]);
			return;
		}

		void human_action(){
			my_command();
			if(quit){
				command[ind] = '_';
				if(online){
					client.send_it();
					client.end_it();
				}
				printer.print("You quitted, press space button to continue\n");
				while(getch() != ' ');
				return;
			}
			if(agent[ind] != -1 && !manual && command[ind] != '3')
				command[ind] = bot(hum[ind]); 
			if(online){
				client.send_it();
				client.recieve();
			}
			for(int i = 0; i < ind; ++i)
				if(mh[i] && !remote[i])
					get_command(i);
			for(int i = ind + 1; i < H; ++i)
				if(mh[i] && !remote[i])
					get_command(i);
			int r = rand() & 1, st = (1 - r) * (H - 1), dif = 2 * r - 1;
			for(int i = st; i < H && (~i); i += dif)
				if(mh[i]){
					std::vector<int> v = hum[i].get_cor();
					obey(command[i], hum[i]);
					teleport(hum[i]);
					claim_chest(hum[i]);
					command[i] = '+';
				}
			return;
		}

		void command_list(bool b = false){
			printer.print(head(true));
			printer.print("Command list:\n");
			printer.print(" - : show backpack\n");
			printer.print(" 0 : command list\n");
			printer.print(" ` : turn to left [1]\n");
			printer.print(" 1 : turn to right\n");
			printer.print(" 3 : switch bitween Manual and Automate\n");
			printer.print(" a or 4 : move to left [2]\n");
			printer.print(" d or 6: move to right\n");
			printer.print(" w or 8: move to up\n");
			printer.print(" s or 2: move to down\n");
			printer.print(" p : punch\n");
			printer.print(" [ : add block\n");
			printer.print(" ] : add portal\n");
			printer.print(" (Item's sign)* : change item\n");
			printer.print(" u : use item(for consumables)\n");
			printer.print(" x : attack\n");
			printer.print(" Q : quit\n");
			printer.print("-------------------------------------\n");
			printer.print("Item signes:\n");
			printer.print(" energy_drink: <f>, first_aid_box: <g>, food_package: <h>, zombie_vaccine: <j>\n");
			printer.print(" gas: <k>, flash_bang: <l>, acid_bomb: <;>, stinger: <'>\n");
			printer.print(" push_dagger: <c>, wing_tactic: <v>, F_898: <b>, lochabrask: <n>\n");
			printer.print(" AK_47: <m>, M416: <,>, MOSSBERG: <.>, AWM: </>\n");
			printer.print("-------------------------------------\n");
			printer.print("How to turn on and off full mode and change width and hight:\n");
			printer.print(" F: on, O: off\n");
			printer.print(" W : increase width, E : decrease width\n");
			printer.print(" R : increase hight, T : decrease hight\n");
			printer.print("-------------------------------------\n");
			printer.print("[1]: its location on the standardized keyboards is the key below Esc.\n");
			printer.print("[2]: you can enable NumLock and then use the arrows!.\n");
			printer.print("note: if you do an invalid move nothing will happen\n");
			printer.print("press any key to continue\n");
			if(!b)
				getch();
			return;
		}

		void update_bull(){
			int cnt = 0;
			for(int _ = 0; _ < B; ++_)
				if(mb[_]){
					std::vector<int> v = bull[_].get_cor();
					int i = v[0], j = v[1], k = v[2];
					int d = bull[_].get_way() - 1;
					themap1[i][j][k].s = themap[i][j][k].s;
					themap1[i][j][k].s[2] = 0;
					place[cnt++] = std::vector<int>{i, j, k};
					themap1[i][j + wdx[d]][k + wdy[d]].s = themap[i][j + wdx[d]][k + wdy[d]].s;
					themap1[i][j + wdx[d]][k + wdy[d]].s[2] = 0;
					place[cnt++] = std::vector<int>{i, j + wdx[d], k + wdy[d]};
				}
			int r = rand() & 1, st = (1 - r) * (B - 1), dif = 2 * r - 1;
			if(r)
				reverse(place, place + cnt);
			for(int _ = st; _ < B && (~_); _ += dif)
				if(mb[_]){
					std::vector<int> v = bull[_].get_cor();
					int i = v[0], j = v[1], k = v[2];
					if(bull[_].expire()){
						mb[_] = false;
						continue;
					}
					int d = bull[_].get_way() - 1;
					char sit = themap[i][j + wdx[d]][k + wdy[d]].showit();
					if((sit != '#' && sit != 'v' && sit != '^') || themap[i][j + wdx[d]][k + wdy[d]].s[10]){
						themap1[i][j + wdx[d]][k + wdy[d]].bullet = &bull[_];
						bull[_].set_cor(std::vector<int>{i, j + wdx[d], k + wdy[d]});
						themap1[i][j + wdx[d]][k + wdy[d]].s[2] = 1;
					}
					else
						mb[_] = false;
				}
			for(int _ = 0; _ < cnt; ++_){
				int i = place[_][0], j = place[_][1], k = place[_][2];
				themap[i][j][k].s[2] = themap1[i][j][k].s[2];
				themap[i][j][k].bullet = themap1[i][j][k].bullet;
			}
			return;
		}

		bool check_end(){
		    if(online && disconnect){
                printer.print("You're disconnected :(\npress space button to continue\n");
				hum[ind].back_Hp();
				hum[ind].back_mindamage();
				hum[ind].back_stamina();
				hum[ind].backpack.back_tmp();
				client.end_it();
				while(getch() != ' ');
				return true;
		    }
			if(hum[ind].get_Hp() <= 0){
                if(online){
					command[ind] = '~';
					client.send_it();
					client.end_it();
				}
                printer.print("You Died :(\npress space button to continue\n");
				hum[ind].back_Hp();
				hum[ind].back_mindamage();
				hum[ind].back_stamina();
				hum[ind].backpack.back_tmp();
				while(getch() != ' ');
				return true;
			}
			if(mode == "Timer"){
				if(time(0) - tb >= level * 60 * 5){
					if(kills < level * 5){
						printer.print("You Lost :(\npress space button to continue\n");
						hum[ind].back_Hp();
						hum[ind].back_mindamage();
						hum[ind].back_stamina();
						hum[ind].backpack.back_tmp();
						while(getch() != ' ');
						return true;
					}
					else{
						std::string s = "You won :)\nreward: ";
						s += std::to_string((int)(hum[ind].get_level_solo() == level) * level * 1000 + loot);
						s += "$\nlevel ";
						s += std::to_string(level);
						s += " has done successfully!\npress space button to continue\n";
						printer.print(s);
						hum[ind].set_money(hum[ind].get_money() + loot + (int)(hum[ind].get_level_timer() == level) * level * 1000);
						if(hum[ind].get_level_timer() == level)
							hum[ind].level_timer_up();
						hum[ind].back_Hp();
						hum[ind].back_mindamage();
						hum[ind].back_stamina();
						hum[ind].backpack.back_tmp();
						while(getch() != ' ');
						return true;
					}
				}
				return false;
			}
			if(level * 5 <= kills && mode == "Solo"){
				std::string s = "You won :)\nreward: ";
				s += std::to_string((int)(hum[ind].get_level_solo() == level) * level * 1000 + loot);
				s += "$\nlevel ";
				s += std::to_string(level);
				s += " has done successfully!\npress space button to continue\n";
				printer.print(s);
				hum[ind].set_money(hum[ind].get_money() + loot + (int)(hum[ind].get_level_solo() == level) * level * 1000);
				if(hum[ind].get_level_solo() == level)
					hum[ind].level_solo_up();
				hum[ind].back_Hp();
				hum[ind].back_mindamage();
				hum[ind].back_stamina();
				hum[ind].backpack.back_tmp();
				while(getch() != ' ');
				return true;
			}
			if(level * 10 <= teams_kills && rivals_are_dead() && mode == "Squad"){
				std::string s = "You won :)\nreward: ";
				s += std::to_string((int)(hum[ind].get_level_squad() == level) * level * 1000 + loot);
				s += "$\nlevel ";
				s += std::to_string(level);
				s += " has done successfully!\npress space button to continue\n";
				printer.print(s);
				if(hum[ind].get_level_squad() == level)
					hum[ind].level_squad_up();
				hum[ind].back_Hp();
				hum[ind].back_mindamage();
				hum[ind].back_stamina();
                hum[ind].backpack.back_tmp();
				hum[ind].set_money(hum[ind].get_money() + loot + (int)(hum[ind].get_level_squad() == level) * level * 1000);
				while(getch() != ' ');
				return true;
			}
			if(online && rivals_are_dead()){
                if(countdown > 0){
                    --countdown;
                    return false;
                }
				std::string s = c_col(32, 40);
				s += "*** Congratulations! You won the match :) ***\n";
				s += c_col(0, 0);
				s += "press space button to continue\n";
				printer.print(s);
				while(getch() != ' ');
				return true;
			}
			return false;
		}

		void setup(){
			countdown = 1;
			Environment::Character::me.backpack.vec = -1;
			tb = time(nullptr);
			loot = teams_kills = kills = frame = 0;
			online = (mode == "AI Battle Royal" || mode == "Battle Royal");
			silent = quit = is_human = false;
			recomZ = nullptr;
			recomH = nullptr;
			temp.clear();
			for(int i = 0; i < B; ++i)
				active[i] = mb[i] = false;
			for(int i = 0; i < Z; ++i)
				mz[i] = false;
			for(int i = 0; i < H; ++i){
				mh[i] = remote[i] = false;
				agent[i] = -1, command[i] = '+';
            }
			for(int k = 0; k < F; ++k){
				std::ifstream f("./map/floor" + std::to_string(k + 1) + ".txt");
				for(int i = 0; i < N; ++i)
					for(int j = 0; j < M; ++j){
						themap[k][i][j] = nd;
						char c;
						f >> c;
						if(c == '#')
							themap[k][i][j].s[3] = 1;
						else if(c == '^'){
							themap[k][i][j].s[5] = 1;
							f >> themap[k][i][j].portal_ind;
						}
						else if(c == 'v'){
							themap[k][i][j].s[6] = 1;
							f >> themap[k][i][j].portal_ind;
						}
						else if(c == 'O'){
							themap[k][i][j].s[7] = 1;
							int index = p_ind();
							portal[index] = std::vector<int>{k, i, j};
							active[index] = 1;
						}
					}
				f.close();
			}
			load_data();
			return;
		}

		void portal_damage(){
			for(int i = 0; i < B; ++i){
				if(!active[i])
					continue;
				std::vector<int> v = portal[i];
				if(themap[v[0]][v[1]][v[2]].showit() != 'O'){
					Environment::Item::Weapon radiation;
					radiation.ready(20, -10, 1);
					int index = b_ind();
					if(index == -1)
						return;
					bull[index].shot(v, 3, radiation, 0);
					themap[v[0]][v[1]][v[2]].bullet = &bull[index];
					themap[v[0]][v[1]][v[2]].s[2] = 1;
					mb[index] = true;
				}
			}
			return;
		}

		void find_recom(){
			is_human = false;
			recomH = nullptr;
			recomZ = nullptr;
			int mn = 1000000021;
			for(int i = 0; i < ind; ++i)
				if(mh[i]){
					std::vector<int> v1 = hum[i].get_cor();
					std::vector<int> v = hum[ind].get_cor();
					int dist = abs(v[1] - v1[1]) + abs(v[2] - v1[2]);
					dist += 60 * (v[0] != v1[0]) + 5 * abs(v[0] - v1[0]);
					if(dist < mn && hum[i].get_team() != hum[ind].get_team()){
						mn = dist;
						is_human = true;
						recomH = &hum[i];
					}
				}
			for(int i = ind + 1; i < H; ++i)
				if(mh[i]){
					std::vector<int> v1 = hum[i].get_cor();
					std::vector<int> v = hum[ind].get_cor();
					int dist = abs(v[1] - v1[1]) + abs(v[2] - v1[2]);
					dist += 60 * (v[0] != v1[0]) + 5 * abs(v[0] - v1[0]);
					if(dist < mn && hum[i].get_team() != hum[ind].get_team()){
						mn = dist;
						is_human = true;
						recomH = &hum[i];
					}
				}
			for(int i = 0; i < Z; ++i)
				if(mz[i]){
					std::vector<int> v1 = zomb[i].get_cor();
					std::vector<int> v = hum[ind].get_cor();
					int dist = abs(v[1] - v1[1]) + abs(v[2] - v1[2]);
					dist += 60 * (v[0] != v1[0]) + 5 * abs(v[0] - v1[0]);
					if(dist < mn){
						mn = dist;
						is_human = false;
						recomZ = &zomb[i];
					}
				}
			return;
		}

        void update_tmp(){
        	for(int _ = 0; _ < B; ++_){
        		if(!mb[_])
        			continue;
        		std::vector<int> v = bull[_].get_cor();
        		int i = v[0], j = v[1], k = v[2];
				char sit = themap[i][j][k].showit();
				if((sit == '^' || sit == '#') && themap[i][j][k].s[10]){
					themap[i][j][k].dmg += bull[_].get_damage();
					themap[i][j][k].s[9] = 1;
					themap[i][j][k].s[2] = 0;
					mb[_] = false;
				}
			}
            for(auto e: temp){
                char c = e->showit();
                int dmg = e->dmg;
                if(c == '^' && dmg >= lim_portal){
					int i = e->portal_ind;
                    auto e1 = &themap[portal[i][0]][portal[i][1]][portal[i][2]];
                    e1->s[7] = e1->s[10] = 0;
                    e->s[5] = e->s[10] = 0;
                    e->portal_ind = -1;
                    e->dmg = 0;
                    active[i] = 0;
                }
                else if(c == '#' && dmg >= lim_block){
                    e->s[3] = e->s[10] = 0;
                    e->dmg = 0;
                }
            }
            for(int i = 0; i < (int)temp.size(); ++i)
                if(!(temp[i]->s[10])){
                    std::swap(temp[i], temp.back());
                    temp.pop_back();
                    --i;
                }
            return;
		}

		void play(){
			setup();
			if(disconnect && online)
				return;
            if(online)
                client.prepare();
			cls();
			std::cout << "* Please keep this terminal\nwindow active while playing :)" << std::endl;
			during_battle = true;
			#if defined(__unix__) || defined(__APPLE__)
			disable_input_buffering();
			#endif
			printer.start();
			start = std::chrono::steady_clock::now();
			++frame, find_recom(), view(), print_game();
			start = std::chrono::steady_clock::now();
			while(true){
				if(frame % pc == 1)
					spawn_chest();
				if(frame % pz == 1)
					spawn_zombie_npc();
				if(frame % ph == 1)
					spawn_human_npc();
				if(check_end())
					break;
				human_action();
				if(quit){
					printer.stop();
					break;
				}
				zombie_action();
				portal_damage();
				view();
				update_tmp();
				hit_human(), hit_zombie();
				++frame, find_recom(), print_game();
				start = std::chrono::steady_clock::now();
				updmap();
				update_bull();
				view();
				update_tmp();
				hit_human(), hit_zombie();
				++frame, find_recom(), print_game();
				start = std::chrono::steady_clock::now();
				updmap();
				update_bull();
			}
			printer.stop();
			if(!online && !quit)
				Environment::Character::me = hum[ind];
			if(!quit)
				update();
			during_battle = false;
			#if defined(__unix__) || defined(__APPLE__)
			restore_input_buffering();
			#endif
			return;
		}

		void open(){
			full = false;
			_H = 7, W = 24;
			for(bool b = false; true;){
				std::cout << head();
				std::cout << "Game Modes:" << '\n';
				std::cout << "  1. Solo" << '\n';
				std::cout << "  2. Timer" << '\n';
				std::cout << "  3. Squad [The other players are bots 5v5]" << '\n';
				std::cout << "  4. Join into a Battle Royal room [Solo/Squad]" << '\n';
				std::cout << "  5. AI Battle Royal!" << '\n';
				std::cout << "  6. Back to menu" << '\n';
				std::cout << "\n----------------------------\n";
				std::cout << "to choose an option write it's section number" << '\n';
				std::cout << "----------------------------" << std::endl;
				if(b)
					std::cout << "invalid input, try again" << std::endl;
				b = false;
				char c = getch();
				if(c == '6')
					return;
				if(c == '1'){
					mode = "Solo";
					while(true){
						std::cout << head();
						std::cout << "Game Mode: Solo\nChoose the level which you want to play:\n";
						for(int i = 0; i < L; ++i){
							std::cout << "  " << i + 1 << ". level " << i + 1 << ", (" << (i ? "" : "0") << 5 * (i + 1) << " kills)";
							if(Environment::Character::me.get_level_solo() < i + 1)
								std::cout << " (Locked)";
							std::cout << '\n';
						}
						std::cout << "------------------\npress any other key to back\n------------------" << std::endl;
						level = getch() - '0';
						if(Environment::Character::me.get_level_solo() < level && level <= L && level > 0){
							std::cout << "you can't choose this level" << std::endl;
							usleep(1500000);
							continue;
						}
						break;
					}
					if(level >= L || level <= 0)
						continue;
					std::cout << "do you want to use your AI agent? (y: yes/any other key: no)" << std::endl;
					manual = (getch() != 'y');
					play();
					continue;
				}
				if(c == '2'){
					mode = "Timer";
					while(true){
						std::cout << head();
						std::cout << "Game Mode: Timer\n(You have to stay alive in all of the time)\nChoose the level which you want to play:\n";
						for(int i = 0; i < L; ++i){
							std::cout << "- level " << i + 1 << ", (" << (i ? "" : "0") << 5 * (i + 1) << " kills, in " << (i ? "" : "0") << 5 * (i + 1) << " minutes)";
							if(Environment::Character::me.get_level_timer() < i + 1)
								std::cout << " (Locked)";
							std::cout << '\n';
                    				}
						std::cout << "------------------\npress any other key to back\n------------------" << std::endl;
						level = getch() - '0';
						if(Environment::Character::me.get_level_timer() < level && level <= L && 0 < level){
							std::cout << "you can't choose this level" << std::endl;
							usleep(1500000);
							continue;
						}
						break;
					}
					if(level >= L || level <= 0)
						continue;
					std::cout << "do you want to use your AI agent? (y: yes/any other key: no)" << std::endl;
					manual = (getch() != 'y');
					play();
					continue;
				}
				if(c == '3'){
					mode = "Squad";
					while(true){
						std::cout << head();
						std::cout << "Game Mode: Squad\nChoose the level which you want to play:\n";
						for(int i = 0; i < L; ++i){
							std::cout << "- level " << i + 1 << ", (" << 10 * (i + 1) << " kills and all of the rival's team death)";
							if(Environment::Character::me.get_level_squad() < i + 1)
								std::cout << " (Locked)";
							std::cout << '\n';
						}
						std::cout << "------------------\npress any other key to back\n------------------" << std::endl;
						level = getch() - '0';
						if(Environment::Character::me.get_level_squad() < level && level <= L && 0 < level){
							std::cout << "you can't choose this level" << std::endl;
							usleep(1500000);
							continue;
						}
						break;
					}
					if(level >= L || level <= 0)
						continue;
					std::cout << "do you want to use your AI agent? (y: yes/any other key: no)" << std::endl;
					manual = (getch() != 'y');
					play();
					continue;
				}
				if(c == '4'){
					level = 1;
					mode = "Battle Royal";
					std::cout << "do you want to use your AI agent? (y: yes/any other key: no)" << std::endl;
					manual = (getch() != 'y');
					play();
					online = false;
					continue;
                }
                if(c == '5'){
                	level = 1;
                    mode = "AI Battle Royal";
					manual = false;
                    play();
                    online = false;
                    continue;
				}
				b = true;
            }
            return;
		}

		void update(){
			if(online || agent[ind] != -1)
				return;
			int r_changes = (kills * 100 * level) / (time(0) - tb + 1);
			if(mode == "Timer")
				Environment::Character::me.set_rate_timer(Environment::Character::me.get_rate_timer() + r_changes);
			else if(mode == "Solo")
				Environment::Character::me.set_rate_solo(Environment::Character::me.get_rate_solo() + r_changes);
			else if(mode == "Squad")
				Environment::Character::me.set_rate_squad(Environment::Character::me.get_rate_squad() + r_changes);
			std::string s = ctime(&tb), ln;
			std::ifstream hs("./accounts/game/" + user + "/" + user + ".txt");
			std::vector<std::string> vec;
			while(getline(hs, ln))
				vec.push_back(ln);
			hs.close();
			std::ofstream histo("./accounts/game/" + user + "/" + user + ".txt");
			histo << s;
			histo << mode << '\n';
			histo << kills << '\n';
			if(r_changes > 0)
				histo << '+';
			histo << r_changes << '\n';
			for(std::string &e: vec)
				histo << e << '\n';
			histo.close();
			vec.clear();
			std::ifstream rnk("./accounts/ranking.txt");
			while(getline(rnk, ln))
				vec.push_back(ln);
			for(int i = 0; i < vec.size(); i += 2)
				if(user == vec[i]){
					vec[i + 1] = std::to_string(stoi(vec[i + 1]) + r_changes);
					break;
				}
			std::ofstream rank("./accounts/ranking.txt");
			for(int i = 0; i < vec.size(); ++i)
				rank << vec[i] << '\n';
			rank.close();
			vec.clear();
			std::string tmp = mode;
			mode[0] = tolower(mode[0]);
			std::ifstream rnk1("./accounts/ranking" + mode + ".txt");
			while(getline(rnk1, ln))
				vec.push_back(ln);
			for(int i = 0; i < vec.size(); i += 2)
				if(user == vec[i]){
					vec[i + 1] = std::to_string(stoi(vec[i + 1]) + r_changes);
					break;
				}
			rnk1.close();
			std::ofstream rank1("./accounts/ranking" + mode + ".txt");
			for(int i = 0; i < vec.size(); ++i)
				rank1 << vec[i] << '\n';
			rank1.close();
			mode = tmp;
			Environment::Character::me.save_progress();
			return;
		}
	} g;

	auto lim = std::chrono::duration<long long, std::ratio<1, 1000000000LL>>(40000000LL);

	void gameplay::print_game() const{
		if(silent){
			if(agent[ind] != -1 && !manual)
			       	return;
			auto end_ = std::chrono::steady_clock::now();
			int k = (lim.count() - (end_ - start).count()) / 1000;
			usleep(std::max(k, 0));
			return;
		}
		std::string res = "";
		if(!full){
			res += head(true, true) + "Mode: " + mode;
			if(agent[ind] != -1){
				if(manual)
					res += " (Manual)";
				else
					res += " (Automate)";
			}
			if(online){
                res += " | index: " + std::to_string(ind);
                res += ", team: " + std::to_string(hum[ind].get_team());
            }
            res += "\n_____________________\n";
    		res += c_col(33, 40);
            res += "Frame: " + std::to_string(frame) + "\n";
			res += "Timer: " + std::to_string(time(nullptr) - tb) + "s\n\n";
			res += c_col(34, 40);
			res += "Your teams' kills: " + std::to_string(teams_kills) + " (yours': " + std::to_string(kills) + ")";
			if(!online)
				res += ", level: " + std::to_string(level);
			res += "\n";
			if(mode == "Timer")
				res += "Your' reward (If you win): " + std::to_string(loot + (int)(hum[ind].get_level_timer() == level) * 1000 * level) + "\n";
			else if(mode == "Solo")
				res += "Your' reward (If you win): " + std::to_string(loot + (int)(hum[ind].get_level_solo() == level) * 1000 * level) + "\n";
			else if(mode == "Squad")
				res += "Your' reward (If you win): " + std::to_string(loot + (int)(hum[ind].get_level_squad() == level) * 1000 * level) + "\n";
			res += "\nYou:\n";
			res += hum[ind].subtitle();
			res += c_col(31, 40) + "\n";
			if(!is_human && recomZ != nullptr){
				res += "Enemy:\n";
				res += (*recomZ).subtitle() + '\n';
			}
			else if(recomH != nullptr){
				res += "Enemy:\n";
				res += (*recomH).subtitle();
			}
			else
				res += "\n\n\n\n\n";
			res += c_col(0, 0);
			if(agent[ind] == -1){
				res += "to see the command list";
				res += (!online ? " or pause the game" : "");
				res += " press 0\n";
			}
			else
				res += "to not show the situation please press space button\n";
			res += "____________________________________________________\n";
		}
		else
			res += "0: command list\n";
		std::vector<int> v = hum[ind].get_cor();
		std::string last = "", color, cell;
		v[1] = std::max(v[1], _H), v[1] = std::min(v[1], N - _H - 1);
		v[2] = std::max(v[2], W), v[2] = std::min(v[2], M - W - 1);
		for(int i = v[1] - _H; i <= v[1] + _H; ++i, res.push_back('\n'))
			for(int j = v[2] - W; j <= v[2] + W; ++j){
				cell = themap[v[0]][i][j].showit_();
				color = "";
				int cnt = 2;
				for(int k = 0; k < cell.size(); ++k){
					if(cnt < 2)
						color.push_back(cell[k]);
					else if(cell[k] != '\033')
						res.push_back(cell[k]);
					else
						color.push_back(cell[k]), cnt = 0;
					if(cell[k] == 'm')
						++cnt;
					if(cnt == 2 && color != last){
						res += color;
						last = color;
					}
				}
			}
		color = c_col(0, 0);
		if(last != color)
			res += color;
		printer.cls();
		printer.print(res.c_str());
		#if defined(__unix__) || defined(__APPLE__)
		auto end_ = std::chrono::steady_clock::now();
		int k = (lim.count() - (end_ - start).count()) / 1000;
		usleep(std::max(k, 0));
		#endif
		return;
	}

	char gameplay::human_rnpc_bot(Environment::Character::Human& player) const{
		if(frame % 50 == 1){
			char c[8] = {'c', 'v', 'b', 'n', 'm', ',', '.', '/'};
			return c[rand() % 8];
		}
		else if(rand() % 5 < 3)
			return 'x';
		else if(rand() % 5 < 3){
			char c[7] = {'1', '2', 'a', 'w', 's', 'd', 'p'};
			return c[rand() % 7];
		}
		char c[8] = {'+', 'u', 'f', 'g', 'h', 'j', '[', ']'};
		return c[rand() % 8];
	}
}