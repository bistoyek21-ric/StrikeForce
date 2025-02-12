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
#include "../../gameplay.hpp"

bool training_mode = true;

namespace Environment::Field{

    auto lim = std::chrono::duration<long long, std::ratio<1, 1000000000LL>>(40000000LL);

	void gameplay::print_game() const{
		if(silent){
            if(agent[ind] != -1)
                return;
            auto end_ = std::chrono::steady_clock::now();
            int k = (lim.count() - (end_ - start).count() + 999) / 1000;
            usleep(std::max(k, 0));
            return;
        }
		std::string res = "";
		if(!full){
		    res += head(false) + "Mode: " + mode;
            if(online){
                res += " | index: " + std::to_string(ind);
                res += ", team: " + std::to_string(hum[ind].get_team());
            }
            res += "\n_____________________\n";
            res += c_col(33, 40, false);
            res += "Frame: " + std::to_string(frame) + "\n";
			res += "Timer: " + std::to_string(time(nullptr) - tb) + "s\n\n";
			res += c_col(34, 40, false);
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
			res += c_col(31, 40, false) + "\n";
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
			res += c_col(0, 0, false);
            if(agent[ind] == -1){
                res += "to see the command list";
                res += (!online ? " or pause the game" : "");
                res += " press 0\n";
            }
            else
                res += "to not show the situation please press space button\n";
            res += "____________________________________________________\n";
        }
        else{
            cls();
            res += "0: command list\n";
        }
        std::vector<int> v = hum[ind].get_cor();
        std::string last = "", color, cell;
        v[1] = std::max(v[1], _H), v[1] = std::min(v[1], N - _H - 1);
        v[2] = std::max(v[2], W), v[2] = std::min(v[2], M - W - 1);
        for(int i = v[1] - _H; i <= v[1] + _H; ++i, res.push_back('\n'))
            for(int j = v[2] - W; j <= v[2] + W; ++j){
                cell = themap[v[0]][i][j].showit_();
                color = "";
                int cnt = 0, k = 0;
                for(; k < cell.size(); ++k){
                    if(cnt < 2)
                        color.push_back(cell[k]);
                    else
                        res.push_back(cell[k]);
                    if(cell[k] == 'm')
                        ++cnt;
                    if(cnt == 2 && color != last){
                        res += color;
                        last = color;
                    }
                }
            }
        color = c_col(0, 0, false);
        if(last != color)
            res += color;
        puts(res.c_str());
        return;
    }

	void gameplay::view() const{
        return;
    }

    char gameplay::bot(Environment::Character::Human& player) const{
        return '+';
    }
}
