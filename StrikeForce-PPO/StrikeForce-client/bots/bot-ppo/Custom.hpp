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

	std::vector<float> describe_full(const node &cell, const Environment::Character::Human &player);

	// Fast path: a cell whose 11 flags are all zero always yields the same 32 numbers (most of the 961 cells
	// of every observation are like that), so it is computed once and copied.
	const std::vector<float>& describe(const node &cell, const Environment::Character::Human &player){
		static thread_local std::vector<float> empty_cell; static thread_local bool have = false;
		static thread_local std::vector<float> scratch;
		if(cell.s.none()){
			if(!have)
				empty_cell = describe_full(cell, player), have = true;
			return empty_cell;
		}
		scratch = describe_full(cell, player);
		return scratch;
	}

	std::vector<float> describe_full(const node &cell, const Environment::Character::Human &player){
		std::vector<float> res;
		res.reserve(32);
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
			res.push_back(cell.human->get_kills() / 20);
			res.push_back(cell.human->backpack.get_blocks() / 20);
			res.push_back(cell.human->backpack.get_portals() / 20);
			res.push_back(cell.human->backpack.get_portal_ind() != -1);
		}
		else{
			for(int i = 0; i < 4; ++i)
				res.push_back(0);
		}
		// it can  be passed through by human, bullet; can it destroyed by shooting, Hp {0, 1}^3 [0, inf)| 4
		sit = {1, 1, 0};
		float hp = 0;
		if(cell.s[3] || cell.s[5] || cell.s[6] || cell.s[0] || cell.s[1]){
			sit[0] = sit[1] = 0;
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
			sit[0] = 0;
			sit[1] = 1, sit[2] = 0;
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
		// damage effect   [0, inf)^2                                                      | 2
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
		// Observation: 32 channels x 31 x 31, channel-major.  Written straight into one preallocated buffer
		// (identical values/order to the original push_back version, ~10x fewer allocations).
		constexpr int R_ = 31 / 2, S_ = 31, CELLS = S_ * S_, CH = 32;
		std::vector<float> obs(CH * CELLS);
		const double beta = 1;
		int target_dist = -1;     // Manhattan distance to the nearest hostile (zombie / other-team human) in the window, -1 = none
		int tgt_i = -1, tgt_j = -1;   // its coordinates (used by the scripted expert below)
		for(int i = v[1] - R_; i <= v[1] + R_; ++i)
			for(int j = v[2] - R_; j <= v[2] + R_; ++j){
				const int idx = (i - (v[1] - R_)) * S_ + (j - (v[2] - R_));
				if(0 <= i && 0 <= j && i < N && j < M){
					const node& c_ = themap[v[0]][i][j];
					if(c_.s[1] || (c_.s[0] && c_.human && c_.human->get_team() != player.get_team())){
						const int d_ = std::abs(i - v[1]) + std::abs(j - v[2]);
						if(target_dist < 0 || d_ < target_dist)
							target_dist = d_, tgt_i = i, tgt_j = j;
					}
				}
				const std::vector<float>& vec = (i < 0 || j < 0 || N <= i || M <= j)
					? describe(nd, player) : describe(themap[v[0]][i][j], player);
				for(int k = 0; k < CH; ++k){
					const float x = vec[k];
					obs[k * CELLS + idx] = (5.0 < x) ? (float)(10 - (10.0 / (beta * (x - 5) + 2))) : x;
				}
			}
		// "aligned" = a hostile stands in the cell(s) the agent's attack reaches.  The action set "+xzqeawsd" has no key that
		// selects a weapon (backpack.vec stays -1, so 'x' does nothing); the only damage is the melee punch 'z', whose
		// weapon range is 1.  So the useful alignment is: enemy adjacent, in the facing direction.
		constexpr int kReach = 1;
		int aligned = 0;
		{
			const int bway = player.get_way() - 1;
			if(0 <= bway && bway < 4)
				for(int step = 1; step <= kReach; ++step){
					const int i = v[1] + wdx[bway] * step, j = v[2] + wdy[bway] * step;
					if(i < 0 || j < 0 || i >= N || j >= M)
						break;
					const node& c_ = themap[v[0]][i][j];
					if(c_.s[1] || (c_.s[0] && c_.human && c_.human->get_team() != player.get_team())){
						aligned = 1;
						break;
					}
					const char sit_ = c_.showit();
					if((sit_ == '#' || sit_ == 'v' || sit_ == '^') && !c_.s[10])
						break;
				}
		}
		// ── Scripted "teacher": walk (shortest path) to the nearest enemy, face it, punch ('z') ──────────────────────
		// Keys: s/d/w/a move +row/+col/-row/-col (facing does NOT change), q/e turn (q: facing index +1, e: -1);
		// facing index f = get_way()-1 indexes wdx/wdy.  Walkable cells = the ones gameplay.hpp lets a move enter.
		// First version walked straight at the enemy: 76 % of its moves hit a wall (10 kills / 1000 steps).
		char expert = '+';
		{
			static unsigned rs = 2463534242u;                 // xorshift: no dependence on the game's rand()
			auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; };
			const char walk[4] = {'s', 'd', 'w', 'a'};
			const int f = player.get_way() - 1;
			auto inside   = [&](int i, int j) { return 0 <= i && 0 <= j && i < N && j < M; };
			auto hostile  = [&](int i, int j) {
				if(!inside(i, j)) return false;
				const node& c = themap[v[0]][i][j];
				return c.s[1] || (c.s[0] && c.human && c.human->get_team() != player.get_team());
			};
			auto passable = [&](int i, int j) {
				if(!inside(i, j)) return false;
				const char t = themap[v[0]][i][j].showit();
				return t == '?' || t == '^' || t == 'v' || t == '.' || t == 'X' || t == '*';
			};
			// 1) an enemy next to me?  punch it if I face it, otherwise turn towards it
			int adj = -1;
			for(int d = 0; d < 4; ++d)
				if(hostile(v[1] + wdx[d], v[2] + wdy[d])){
					if(d == f){ adj = d; break; }
					if(adj < 0) adj = d;
				}
			if(adj >= 0){
				const int diff = ((adj - f) % 4 + 4) % 4;
				expert = diff == 0 ? 'z' : (diff == 3 ? 'e' : 'q');
			}
			else{
				// 2) breadth-first search over the 31x31 window to the nearest cell that touches an enemy
				constexpr int W_ = 31;
				static signed char first[W_][W_]; static bool seen[W_][W_]; static int qi[W_ * W_], qj[W_ * W_];
				for(int a_ = 0; a_ < W_; ++a_) for(int b_ = 0; b_ < W_; ++b_) seen[a_][b_] = false, first[a_][b_] = -1;
				auto loc = [&](int i) { return i - (v[1] - R_); };
				auto locj = [&](int j) { return j - (v[2] - R_); };
				int head = 0, tail = 0, goal = -1;
				seen[loc(v[1])][locj(v[2])] = true;
				qi[tail] = v[1], qj[tail++] = v[2];
				while(head < tail && goal < 0){
					const int ci = qi[head], cj = qj[head++];
					for(int d = 0; d < 4; ++d){
						const int ni = ci + wdx[d], nj = cj + wdy[d];
						if(std::abs(ni - v[1]) > R_ || std::abs(nj - v[2]) > R_ || !passable(ni, nj) || seen[loc(ni)][locj(nj)])
							continue;
						seen[loc(ni)][locj(nj)] = true;
						first[loc(ni)][locj(nj)] = (ci == v[1] && cj == v[2]) ? d : first[loc(ci)][locj(cj)];
						qi[tail] = ni, qj[tail++] = nj;
						for(int e = 0; e < 4; ++e)
							if(hostile(ni + wdx[e], nj + wdy[e])){ goal = first[loc(ni)][locj(nj)]; break; }
						if(goal >= 0) break;
					}
				}
				if(goal >= 0)
					expert = walk[goal];
				else{                                                // nothing reachable in view: wander, keep direction until blocked
					static int dir = 0, lv = -1, lr = -1, lc = -1;
					if(lv == v[0] && lr == v[1] && lc == v[2] || rnd() % 12 == 0)
						dir = rnd() % 4;
					lv = v[0], lr = v[1], lc = v[2];
					expert = walk[dir];
				}
			}
		}
#if defined(EXPERT_POLICY)
		// Build flag: play the scripted policy itself (reference / baseline), no learning involved.
		player.agent->observe(player.get_Hp(), player.get_stamina(), player.get_damage_effect()[0],
		                      player.get_damage(), player.get_kills(), target_dist, aligned);
		return expert;
#endif
		// Reward-shaping signals (raw Human getters + targeting) — consumed by Agent::observe() BEFORE the action is chosen.
		player.agent->observe(player.get_Hp(), player.get_stamina(), player.get_damage_effect()[0],
		                      player.get_damage(), player.get_kills(), target_dist, aligned);
		const int expert_idx = (int)action.find(expert);   // index of the teacher's key in "+xzqeawsd"
		return action[player.agent->predict(obs, expert_idx)];
    }

	void gameplay::prepare(Environment::Character::Human& player){
		action = "+xzqeawsd";
		player.agent = new Agent(/*training=*/true);   // FREEZE_AGENT_BLOCK in macros.hpp switches to evaluation mode
		player.set_agent_active();
	}

	void gameplay::view() const {
        return;
    }
}