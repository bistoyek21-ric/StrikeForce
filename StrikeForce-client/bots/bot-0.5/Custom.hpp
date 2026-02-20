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

	std::vector<float> describe(const node &cell, const Environment::Character::Human &player){
		std::vector<float> res;
		// object type |Char bullet wall chest portal-in portal-out tmp| {0, 1}^7      | 7
		res.push_back(cell.s[0] || cell.s[1]);
		res.push_back(cell.s[2]); res.push_back(cell.s[3]); res.push_back(cell.s[4]);
		res.push_back(cell.s[5] || cell.s[6]);
		res.push_back(cell.s[7]); res.push_back(cell.s[10]);
		// character situation |khoodie, doshmane, npc, zombie| {0, 1}^4 N^3 {0, 1}    | 8
		std::vector<float> sit = {0, 0, 0, 0};
		if(cell.s[0]){
			int t = cell.human->get_team();
			if(!t)
				sit[2] = 1;
			else if(t == player.get_team())
				sit[0] = 1;
			else
				sit[1] = 1;
		}
		if(cell.s[1])
			sit[3] = 1;
		for(int i = 0; i < 4; ++i)
			res.push_back(sit[i]);
		if(cell.s[0]){
			res.push_back(cell.human->get_kills());
			res.push_back(cell.human->backpack.get_blocks());
			res.push_back(cell.human->backpack.get_portals());
			res.push_back(cell.human->backpack.get_portal_ind() != -1);
		}
		else{
			for(int i = 0; i < 4; ++i)
				res.push_back(0);
		}
		// it can't be passed through by human, bullet; can it destroyed by shooting, Hp {0, 1}^3 [0, inf)| 4
		sit = {0, 0, 0};
		float hp = 0;
		if(cell.s[3] || cell.s[5] || cell.s[6] || cell.s[0] || cell.s[1]){
			sit[0] = sit[1] = 1;
			sit[2] = cell.s[10] || cell.s[0] || cell.s[1];
			if(cell.s[0])
				hp = cell.human->get_Hp() / 1000.0;
			else if(cell.s[1])
				hp = cell.zombie->get_Hp() / 1000.0;
			else if(cell.s[10]){
				if(cell.s[3])
					hp = (lim_block - cell.dmg) / 1000.0;
				else
					hp = (lim_portal - cell.dmg) / 1000.0;
			}
		}
		else if(cell.s[7]){
			sit[0] = 1;
			sit[1] = sit[2] = 0;
		}
		for(int i = 0; i < 3; ++i)
			res.push_back(sit[i]);
		res.push_back(hp);
		// is it a bullet, attack vector, damage effect stamina {0, 1} [0, 1]^4 [0, inf)^3 | 8
		sit = {0, 0, 0, 0};
		float damage = 0, effect = 0, is_bull = 0, estamina = 0;
		if(cell.s[0]){
			sit[cell.human->get_way() - 1] = 1;
			auto v = cell.human->get_damage_effect();
			damage = v[0] / 1000.0;
			effect = -v[1] / 1000.0;
			estamina = cell.human->get_stamina() / 1000.0;
		}
		else if(cell.s[1]){
			sit = {0.01, 0.01, 0.01, 0.01};
			damage = cell.zombie->get_mindamage() / 1000.0;
		}
		else if(cell.s[2]){
			is_bull = 1;
			auto dc = cell.bullet->get_dcor();
        	auto c = cell.bullet->get_cor();
			int dist_traveled = abs(c[1] - dc[1]) + abs(c[2] - dc[2]);
			sit[cell.bullet->get_way() - 1] = (cell.bullet->get_range() - dist_traveled) / 100.0;
			damage = cell.bullet->get_damage() / 1000.0;
			effect = -cell.bullet->get_effect() / 1000.0;
		}
		else if(cell.s[7]){
			damage = 20 / 1000.0;
			effect = 10 / 1000.0;
		}
		res.push_back(is_bull);
		for(int i = 0; i < 4; ++i)
			res.push_back(sit[i]);
		res.push_back(damage), res.push_back(effect), res.push_back(estamina);
		// Consumable items [0, inf)^3                                                     | 3
		sit = {0, 0, 0};
		if(cell.s[4]){
			sit[0] = cell.cons->get_stamina() / 1000.0;
			sit[1] = cell.cons->get_effect() / 1000.0;
			sit[2] = cell.cons->get_Hp() / 1000.0;
		}
		for(int i = 0; i < 3; ++i)
			res.push_back(sit[i]);
		// damage effect   [0, inf)^2                                                     | 2
		if(cell.s[0]){
			res.push_back(cell.human->get_damage() / 1000.0);
			res.push_back(-cell.human->get_effect() / 1000.0);
		}
		else{
			res.push_back(0);
			res.push_back(0);
		}
		return res;
	}

	char gameplay::bot(Environment::Character::Human& player) const {
		if(!player.get_active_agent())
			return '+';
		std::vector<int> v = player.get_cor();
		std::vector<float> obs;
		std::vector<std::vector<float>> ch(32);
		int r = 31 / 2;
		for(int i = v[1] - r; i <= v[1] + r; ++i)
			for(int j = v[2] - r; j <= v[2] + r; ++j){
				std::vector<float> vec;
				if(i < 0 || j < 0 || N <= i || M <= j)
					vec = describe(nd, player);
				else
					vec = describe(themap[v[0]][i][j], player);
				for(int k = 0; k < vec.size(); ++k)
					ch[k].push_back(vec[k]);
			}
		for(int k = 0; k < 32; ++k)
			for(int i = 0; i < 31; ++i)
				for(int j = 0; j < 31; ++j)
					if (ch[k][i * 31 + j] >= 0.5)
						obs.push_back(10 / (1 + std::exp(5.0 - ch[k][i * 31 + j])));
					else
						obs.push_back(ch[k][i * 31 + j]);	
		return action[player.agent->predict(obs)];
    }

	void gameplay::prepare(Environment::Character::Human& player){
		action = "+xzqeawsd";
		player.agent = new Agent(/*training=*/true);
		player.set_agent_active();
	}

	void gameplay::view() const {
        return;
    }
}