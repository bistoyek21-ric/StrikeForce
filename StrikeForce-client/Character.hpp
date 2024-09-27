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
#include"Item.hpp"

namespace Enviorment::Character{

	const int wdx[4] = {1, 0, -1, 0}, wdy[4] = {0, 1, 0, -1};

	class Backpack{

	using pci = std::pair<Enviorment::Item::ConsumableItem, int>;

	using pbii = std::pair<Enviorment::Item::Bullet, std::pair<int, int>>;

	using pii = std::pair<int, int>;

	using pwi = std::pair<Enviorment::Item::Weapon, int>;

	private:
		int capacity, lvl, vol, price;

		int def_blocks, blocks, def_portals, portals, portal_ind;

	public:
		int vec, ind;

		pci list_cons[4];

		pbii list_throw[4];

		pwi list_w[8];

		int get_price() const{
			return price;
		}

		void build(){
		    def_blocks = 8;
		    def_portals = 1;
			capacity = 2000, lvl = 1, vol = 0, price = 100;
			portal_ind = -1;
			vec = ind = -1;
			for(int i = 0; i < 4; ++i)
				list_cons[i] = pci{Enviorment::Item::cons[i], 0};
			for(int i = 0; i < 4; ++i)
				list_throw[i] = pbii{Enviorment::Item::throw_[i], pii{1, 0}};
			for(int i = 0; i < 8; ++i)
				list_w[i] = pwi{Enviorment::Item::w[i], 0};
			return;
		}

		std::string get_select() const{
			if(vec == -1)
				return "punch";
			if(vec == 0)
				return list_cons[ind].first.get_name() + "(" + std::to_string(list_cons[ind].second) + ")";
			if(vec == 1)
				return list_throw[ind].first.get_name() + "(" + std::to_string(list_throw[ind].second.second) + ") lvl" + std::to_string(list_throw[ind].first.get_level());
			if(vec == 2)
				return list_w[ind].first.get_name() + " lvl" + std::to_string(list_w[ind].first.get_level());
			return "";
		}

		int get_capacity() const{
			return capacity;
		}

		int get_vol() const{
			return vol;
		}

		int get_level() const{
			return lvl;
		}

		void upgrade(){
			capacity += 200;
			price += 200;
			++lvl;
			return;
		}

		void set_vol(int vol){
			this->vol = vol;
			return;
		}

		void set_portal_ind(int portal_ind){
		    this->portal_ind = portal_ind;
		    return;
		}

		void use_portal(){
			--portals;
			return;
		}

		int get_portals() const{
			return portals;
		}

		void use_block(){
			--blocks;
			return;
		}

		int get_blocks() const{
			return blocks;
		}

		int get_portal_ind() const{
		    return portal_ind;
		}

		void back_tmp(){
		    blocks = def_blocks;
		    portals = def_portals;
		    portal_ind = -1;
		    return;
		}

		void increase_def(){
		    ++def_blocks;
		    ++def_portals;
		    return;
		}

		void show(int money, bool b = false){
			head();
			std::cout << "Your money: " << money << "$\n";
			std::cout << "Accopied volume: " << vol << " / " << capacity << "\n\n";
			std::cout << "Consumeables:\n";
			std::cout << list_cons[0].first.get_name() << " : x" << list_cons[0].second << ",    ";
			std::cout << list_cons[1].first.get_name() << " : x" << list_cons[1].second << '\n';
			std::cout << list_cons[2].first.get_name() << " : x" << list_cons[2].second << ",    ";
			std::cout << list_cons[3].first.get_name() << " : x" << list_cons[3].second << '\n';
			std::cout << "------------------------------------------" << '\n';
			std::cout << "Throwables:\n";
			std::cout << list_throw[0].first.get_name() << " : lvl" << list_throw[0].second.first << ", x" << list_throw[0].second.second << ",    ";
			std::cout << list_throw[1].first.get_name() << " : lvl" << list_throw[1].second.first << ", x" << list_throw[1].second.second << '\n';
			std::cout << list_throw[2].first.get_name() << " : lvl" << list_throw[2].second.first << ", x" << list_throw[2].second.second << ",    ";
			std::cout << list_throw[3].first.get_name() << " : lvl" << list_throw[3].second.first << ", x" << list_throw[3].second.second << '\n';
			std::cout << "------------------------------------------" << '\n';
			std::cout << "ColdWeapon:\n";
			std::cout << list_w[0].first.get_name() << " : lvl" << list_w[0].second << ",    ";
			std::cout << list_w[1].first.get_name() << " : lvl" << list_w[1].second << '\n';
			std::cout << list_w[2].first.get_name() << " : lvl" << list_w[2].second << ",    ";
			std::cout << list_w[3].first.get_name() << " : lvl" << list_w[3].second << '\n';
			std::cout << "------------------------------------------" << '\n';
			std::cout << "WarmWeapon:\n";
			std::cout << list_w[4].first.get_name() << " : lvl" << list_w[4].second << ",    ";
			std::cout << list_w[5].first.get_name() << " : lvl" << list_w[5].second << '\n';
			std::cout << list_w[6].first.get_name() << " : lvl" << list_w[6].second << ",    ";
			std::cout << list_w[7].first.get_name() << " : lvl" << list_w[7].second << '\n';
			std::cout << "------------------------------------------" << '\n';
			std::cout << "Available Blcocks: (" << blocks << "/" << def_blocks << ")" << '\n';
			std::cout << "Available Portals: (" << portals << "/" << def_portals << ")" << '\n';
			std::cout << "------------------------------------------" << '\n';
			std::cout << "lvl0 means you don't have this item" << '\n';
			std::cout << "press any key to continue" << std::endl;
			if(!b)
				getch();
			return;
		}
	};

	class Character{

	protected:
		std::string name;
		int Hp, mindamage, mindamage_def, def_Hp;
		std::vector<int> cor = {0, 0, 0};

	public:
		void set_Hp(int Hp){
			this->Hp = Hp;
			return;
		}

		int get_Hp() const{
			return Hp;
		}

		void hit(const Enviorment::Item::Bullet &b){
			Hp -= b.get_damage();
			mindamage += b.get_effect();
			return;
		}

		void set_name(std::string s){
			name = s;
			return;
		}

		std::string get_name() const{
			return name;
		}

		std::vector<int> get_cor() const{
			return cor;
		}

		void set_cor(const std::vector<int> &c){
			cor = c;
			return;
		}

		void set_mindamage(int mindamage){
			this->mindamage = mindamage;
			return;
		}

		int get_mindamage() const{
			return mindamage;
		}

		int get_mindamage_def() const{
			return mindamage_def;
		}

		void set_def_Hp(int def_Hp){
			this->def_Hp = def_Hp;
			return;
		}

		virtual std::string subtitle(){
			return "";
		}
	};

	class Human: public Character{

	protected:
		bool rnpc;
		int level_solo, level_timer, level_squad, money, stamina, def_stamina, rate_solo, rate_timer, rate_squad, rate, way, team;

	public:
		Backpack backpack;

		void set_team(int team){
			this->team = team;
			return;
		}

		int get_team() const{
			return team;
		}

		void claim_chest(const Enviorment::Item::ConsumableItem &c){
			stamina += c.get_stamina();
			Hp += c.get_Hp();
			mindamage += c.get_effect();
			return;
		}

		bool use(const Enviorment::Item::ConsumableItem &c){
			if(backpack.vec || backpack.list_cons[backpack.ind].second < 1)
				return false;
			stamina += c.get_stamina();
			Hp += c.get_Hp();
			mindamage += c.get_effect();
			if((--backpack.list_cons[backpack.ind].second) < 1)
				backpack.vec = -1;
            backpack.set_vol(backpack.get_vol() - c.get_vol());
			return true;
		}

		bool punch(Enviorment::Item::Bullet &b){
			Enviorment::Item::Weapon p;
			p.ready(std::max(0, mindamage), 0, 1);
			std::vector<int> cor_ = {cor[0], cor[1] + wdx[way - 1], cor[2] + wdy[way - 1]};
			b.shot(cor_, way, p, (uintptr_t)this);
			return true;
		}

		bool shot_it(Enviorment::Item::Bullet &b){
		    Enviorment::Item::Weapon w = backpack.list_w[backpack.ind].first;
			if(stamina + w.get_stamina() < 0)
				return false;
			stamina += w.get_stamina();
			w.set_damage(std::max(0, w.get_damage() + mindamage));
			std::vector<int> cor_ = {cor[0], cor[1] + wdx[way - 1], cor[2] + wdy[way - 1]};
			b.shot(cor_, way, w, (uintptr_t)this);
			return true;
		}

		bool throw_it(Enviorment::Item::Bullet &b){
			b = backpack.list_throw[backpack.ind].first;
			b.set_damage(std::max(0, b.get_damage() + mindamage));
			if(stamina + b.get_stamina() < 0)
				return false;
			if(backpack.list_throw[backpack.ind].second.second < 1){
				backpack.vec = -1;
				return false;
			}
			stamina += b.get_stamina();
			--backpack.list_throw[backpack.ind].second.second;
			if(backpack.list_throw[backpack.ind].second.second < 1)
				backpack.vec = -1;
			std::vector<int> cor_ = {cor[0], cor[1] + wdx[way - 1], cor[2] + wdy[way - 1]};
			b.shot(cor_, way, b, (uintptr_t)this);
			backpack.set_vol(backpack.get_vol() - b.get_vol());
			return true;
		}

		void set_rate(int rate){
			this->rate = rate;
			return;
		}

		int get_rate() const{
			return rate;
		}

		void set_rate_squad(int rate_squad){
			this->rate_squad = rate_squad;
			return;
		}

		int get_rate_squad() const{
			return rate_squad;
		}

		void set_rate_solo(int rate_solo){
			this->rate_solo = rate_solo;
			return;
		}

		int get_rate_solo() const{
			return rate_solo;
		}

		void set_rate_timer(int ratetimer){
			this->rate_timer = rate_timer;
			return;
		}

		int get_rate_timer() const{
			return rate_timer;
		}

		int get_money() const{
			return money;
		}

		void set_money(int money){
			this->money = money;
			return;
		}

		void set_rnpc(bool rnpc){
			this->rnpc = rnpc;
			return;
		}

		bool is_rnpc() const{
			return rnpc;
		}

		void scan(const char* buffer){
			backpack.build();
			rnpc = false;
			char nm[100] = {};
			sscanf(buffer, "%s", &nm);
			while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
				++buffer;
			name = (std::string)(nm);
			sscanf(buffer, "%d %d %d", &def_Hp, &mindamage_def, &def_stamina);
			for(int _ = 0; _ < 3; ++_)
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
			Hp = def_Hp, mindamage = mindamage_def, stamina = def_stamina;
			sscanf(buffer, "%d %d %d", &level_solo, &level_timer, &level_squad);
			for(int _ = 0; _ < 3; ++_)
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
			sscanf(buffer, "%d", &money);
			while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
				++buffer;
			sscanf(buffer, "%d %d %d %d", &rate_solo, &rate_timer, &rate_squad, &rate);
			for(int _ = 0; _ < 4; ++_)
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
			Hp = def_Hp, mindamage = mindamage_def, stamina = def_stamina;
			for(int i = 0; i < 4; ++i){
				sscanf(buffer, "%d", &backpack.list_cons[i].second);
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
				backpack.set_vol(backpack.get_vol() + backpack.list_cons[i].second * backpack.list_cons[i].first.get_vol());
			}
			for(int i = 0; i < 4; ++i){
				sscanf(buffer, "%d", &backpack.list_throw[i].second.first);
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
				sscanf(buffer, "%d", &backpack.list_throw[i].second.second);
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
				for(int j = 0; j + 1 < backpack.list_throw[i].second.first; ++j)
					backpack.list_throw[i].first.upgrade();
				backpack.set_vol(backpack.get_vol() + backpack.list_throw[i].second.second * backpack.list_throw[i].first.get_vol());
			}
			for(int i = 0; i < 8; ++i){
				sscanf(buffer, "%d", &backpack.list_w[i].second);
				while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
					++buffer;
				for(int j = 0; j < backpack.list_w[i].second; ++j)
					backpack.list_w[i].first.upgrade();
				backpack.set_vol(backpack.get_vol() + backpack.list_w[i].second * backpack.list_w[i].first.get_vol());
			}
			int k, k1;
			sscanf(buffer, "%d", &k);
			while(*buffer != '\t' && *buffer != '\n' && *buffer != ' ' && *buffer != '\0')
				++buffer;
			while(--k)
				backpack.upgrade();
            k1 = k = level_solo;
            level_solo = 1;
			while(--k)
				level_solo_up();
            level_solo = k1;
            k1 = k = level_timer;
            level_timer = 1;
			while(--k)
				level_timer_up();
            level_timer = k1;
            k1 = k = level_squad;
            level_squad = 1;
			while(--k)
				level_squad_up();
            level_squad = k1;
			backpack.back_tmp();
			return;
		}

		void build(bool rnpc = false, std::string _name = "", std::string dir = ""){
			backpack.build();
			std::ifstream f;
			this->rnpc = rnpc;
			if(!dir.empty()){
				f.open(dir);
				f >> name;
			}
			else if(!rnpc){
				f.open("./accounts/game/" + user + "/info, " + user + ".txt");
				f >> name;
			}
			else{
				f.open("./character/human_enemy.txt");
				if(_name.empty())
					name = "H" + std::to_string(time(0));
				else
					name = _name;
			}
			f >> def_Hp >> mindamage_def >> def_stamina >> level_solo >> level_timer >> level_squad >> money >> rate_solo >> rate_timer >> rate_squad >> rate;
			Hp = def_Hp, mindamage = mindamage_def, stamina = def_stamina;
			for(int i = 0; i < 4; ++i){
				f >> backpack.list_cons[i].second;
				backpack.set_vol(backpack.get_vol() + backpack.list_cons[i].second * backpack.list_cons[i].first.get_vol());
			}
			for(int i = 0; i < 4; ++i){
				f >> backpack.list_throw[i].second.first;
				f >> backpack.list_throw[i].second.second;
				for(int j = 0; j + 1 < backpack.list_throw[i].second.first; ++j)
					backpack.list_throw[i].first.upgrade();
				backpack.set_vol(backpack.get_vol() + backpack.list_throw[i].second.second * backpack.list_throw[i].first.get_vol());
			}
			for(int i = 0; i < 8; ++i){
				f >> backpack.list_w[i].second;
				for(int j = 0; j < backpack.list_w[i].second; ++j)
                    backpack.list_w[i].first.upgrade();
				backpack.set_vol(backpack.get_vol() + backpack.list_w[i].second * backpack.list_w[i].first.get_vol());
			}
			int k, k1;
			f >> k;
			while(--k)
				backpack.upgrade();
            k1 = k = level_solo;
            level_solo = 1;
			while(--k)
				level_solo_up();
            level_solo = k1;
            k1 = k = level_timer;
            level_timer = 1;
			while(--k)
				level_timer_up();
            level_timer = k1;
            k1 = k = level_squad;
            level_squad = 1;
			while(--k)
				level_squad_up();
            level_squad = k1;
			backpack.back_tmp();
			return;
		}

		void save_progress(){
			std::ofstream f("./accounts/game/" + user + "/info, " + user + ".txt");
			f << name << '\n';
			f << def_Hp << '\n' << mindamage_def << '\n' << def_stamina << '\n' << level_solo << '\n';
			f << level_timer << '\n' << level_squad << '\n' << money << '\n' << rate_solo << '\n' << rate_timer << '\n' << rate_squad << '\n' << rate << '\n';
			for(int i = 0; i < 4; ++i)
				f << backpack.list_cons[i].second << '\n';
			for(int i = 0; i < 4; ++i)
				f << backpack.list_throw[i].second.first << '\n' << backpack.list_throw[i].second.second << '\n';
			for(int i = 0; i < 8; ++i)
				f << backpack.list_w[i].second << '\n';
			f << backpack.get_level() << '\n';
			f.close();
			return;
		}

		int get_def_stamina() const{
			return def_stamina;
		}

		void back_stamina(){
			stamina = def_stamina;
			return;
		}

		void set_stamina(int stamina){
			this->stamina = stamina;
			return;
		}

		int get_stamina() const{
			return stamina;
		}

		void turn_r(){
			if(way == 1)
				way = 4;
			else
				--way;
			return;
		}

		void turn_l(){
			if(way == 4)
				way = 1;
			else
				++way;
			return;
		}

		int get_level_solo() const{
			return level_solo;
		}

		void level_solo_up(){
			++level_solo;
			mindamage_def += 5;
			def_Hp += 50;
			def_stamina += 50;
			if(level_solo % 2 == 1)
                backpack.increase_def();
			return;
		}

		int get_level_squad() const{
			return level_squad;
		}

		void level_squad_up(){
			++level_squad;
			mindamage_def += 5;
			def_Hp += 50;
			def_stamina += 50;
			if(level_squad % 2 == 1)
                backpack.increase_def();
			return;
		}

		int get_level_timer() const{
			return level_timer;
		}

		void level_timer_up(){
			++level_timer;
			mindamage_def += 5;
			def_Hp += 50;
			def_stamina += 50;
			if(level_timer % 2 == 1)
                backpack.increase_def();
			return;
		}

		void set_way(int way){
			this->way = way;
			return;
		}

		int get_way() const{
			return way;
		}

		void back_Hp(){
			Hp = def_Hp;
			return;
		}

		void back_mindamage(){
			mindamage = mindamage_def;
			return;
		}

		virtual std::string subtitle() override{
			return "username: " + name + ", Hp: " + std::to_string(Hp) + "\n" +
			"stamina: " + std::to_string(stamina) + ", money: " + std::to_string(money) + "\n" +
			"selected item: " + backpack.get_select() + ", mindamage: " + std::to_string(mindamage) +"\n" +
			"coordinate: " + std::to_string(cor[0]) + ", " + std::to_string(cor[1]) + ", " + std::to_string(cor[2]) + "\n";
		}
	};

	class Zombie: public Character{

	private:
		bool super;

	public:
		bool punch(Enviorment::Item::Bullet &b, int way){
			Enviorment::Item::Weapon p;
			p.ready(std::max(0, mindamage), 0, 1);
			std::vector<int> cor_ = {cor[0], cor[1] + wdx[way], cor[2] + wdy[way]};
			b.shot(cor_, way + 1, p, 0);
			return true;
		}

		bool is_super() const{
			return super;
		}

		void gen_npc(bool b){
			super = b;
			name = (b ? "S" : "");
			name += "Z" + std::to_string(time(0));
			mindamage_def = (b + 1) * 100, def_Hp = (b + 1) * 400;
			mindamage = mindamage_def, Hp = def_Hp;
			return;
		}

		virtual std::string subtitle() override{
			return (std::string)(super ? "super " : "") + "zombie: " + name + ", Hp = " + std::to_string(Hp) + "\n" +
			"cordinate: " + std::to_string(cor[0]) + ", " + std::to_string(cor[1]) + ", " + std::to_string(cor[2]) +
			", damage: " + std::to_string(std::max(0, mindamage)) + "\n\n";
		}
	};

	void gen_zombie(Zombie &z, bool super, std::vector<int> cor_, std::string name = ""){
		z.set_cor(cor_);
		z.gen_npc(super);
		z.set_name(name);
		return;
	}

	void gen_human(bool rnpc, Human &h, int lvl, std::vector<int> cor_, std::string name = "", std::string dir = ""){
		h.set_cor(cor_);
		h.set_way(1);
		h.build(true, name, dir);
		h.set_rnpc(rnpc);
		h.set_team(0);
		while(--lvl){
			h.level_solo_up();
			h.level_timer_up();
			h.level_squad_up();
		}
		return;
	}

	Human me;
}
