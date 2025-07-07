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
#include "../../gameplay.hpp"

namespace Environment::Field{
    
    auto lim = std::chrono::duration<long long, std::ratio<1, 1000000000LL>>(40000000LL);

	void gameplay::print_game() const{
		if(silent){
			if(using_an_agent && !manual)
			    return;
			auto end_ = std::chrono::steady_clock::now();
			int k = (lim.count() - (end_ - start).count()) / 1000;
			usleep(std::max(k, 0));
			return;
		}
		std::string res = "";
		if(!full){
			res += head(true, true) + "Mode: " + mode;
			if(using_an_agent){
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
			if(!using_an_agent){
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
					else if(cell[k] != '\033'){
						if(cell[k] == 'V')
							res.push_back((char)1);
						else if(cell[k] == '>')
							res.push_back((char)2);
						else if(cell[k] == 'A')
							res.push_back((char)3);
						else if(cell[k] == '<')
							res.push_back((char)4);
						else
							res.push_back(cell[k]);
					}
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
		auto end_ = std::chrono::steady_clock::now();
		int k = (lim.count() - (end_ - start).count()) / 1000;
		if(!using_an_agent)
			usleep(std::max(k, 0));
		return;
	}

	char gameplay::human_rnpc_bot(Environment::Character::Human& player) const{
		if(frame % 50 == 1){
			char c[8] = {'c', 'v', 'b', 'n', 'm', ',', '.', '/'};
			return c[rand() % 8];
		}
		else if(rand() % 5 < 3){
			char c[7] = {'1', '2', 'a', 'w', 's', 'd', 'p'};
			return c[rand() % 7];
		}
		else if(rand() % 5 < 3)
			return 'x';
		char c[8] = {'+', 'u', 'f', 'g', 'h', 'j', '[', ']'};
		return c[rand() % 6];
	}

	std::vector<double> describe(const node &cell, const Environment::Character::Human &player){
		std::vector<double> res;
		if(cell.s[0]){
			double team[3] = {};
			team[0] = (player.get_team() == cell.human->get_team());
			team[1] = (player.get_team() != cell.human->get_team() && cell.human->get_team());
			team[2] = (!cell.human->get_team());
			for(int i = 0; i < 3; ++i)
				res.push_back(team[i]);
			double dir[4] = {};
			dir[cell.human->get_way() - 1] += 1;
			for(int i = 0; i < 4; ++i)
				res.push_back(dir[i]);
		}
		else{
			for(int i = 0; i < 7; ++i)
				res.push_back(0);
		}
		if(cell.s[1]){
			res.push_back(cell.zombie->is_super());
			res.push_back(1 - res.back());
		}
		else{
			for(int i = 0; i < 2; ++i)
				res.push_back(0);
		}
		if(cell.s[2]){
			double dir[4] = {};
			auto dc = cell.bullet->get_dcor();
			auto c =  cell.bullet->get_cor();
			dir[cell.bullet->get_way() - 1] += (cell.bullet->get_range() - abs(c[1] - dc[1]) - abs(c[2] - dc[2])) / 100.0;
			for(int i = 0; i < 4; ++i)
				res.push_back(dir[i]);
		}
		else{
			for(int i = 0; i < 4; ++i)
				res.push_back(0);
		}
		for(int i = 3; i < 11; ++i){
			if(i == 6)
				continue;
			if(i == 5){
				res.push_back(cell.s[5] || cell.s[6]);
				continue;
			}
			res.push_back(cell.s[i]);
		}
		return res;
	}

	void gameplay::prepare(){
		action = "+`1awsdxpm";
		Environment::Character::me.agent = new Agent(online, true, 128, 0.99, 1e-3, 121 * 22, action.size());
	}

	void gameplay::view() const {
        return;
    }

    char gameplay::bot(Environment::Character::Human& player) const {
		if(&player != &hum[ind])
			return '+';
		std::vector<int> v = player.get_cor();
		std::vector<double> obs;
		for(int i = v[1] - 10; i <= v[1] + 10; ++i)
			for(int j = v[2] - 10; j <= v[2] + 10; ++j){
				if(10 < abs(v[1] - i) + abs(v[2] - j))
					continue;
				std::vector<double> vec;
				if(std::min(i, j) < 0 || N <= i || M <= j)
					vec = describe(themap[0][0][0], player);
				else
					vec = describe(themap[v[0]][i][j], player);
				for(auto &e: vec)
					obs.push_back(e);
			}
		return action[player.agent->predict(obs)];
    }
}