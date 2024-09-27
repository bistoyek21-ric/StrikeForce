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
#include "random.hpp"

namespace Environment::Item{

	class Item{

	protected:
    	std::string name;
		int price, vol, lvl, stamina;

	public:
		std::string get_name() const{
			return name;
		}

		int get_stamina() const{
			return stamina;
		}

		int get_price() const{
			return price;
		}

		int get_vol() const{
			return vol;
		}

		int get_level() const{
			return lvl;
		}
	};

	class ConsumableItem: public Item{
	protected:
		int Hp, effect;
	public:
		int get_Hp() const{
			return Hp;
		}

		int get_effect() const{
			return effect;
		}

		void build_cons(std::string s){
			std::ifstream f(s);
			f >> name >> price >> vol >> lvl >> stamina;
			f >> Hp >> effect;
			return;
		}
	};

	class Weapon: public Item{
	protected:
		int damage, effect, range;
	public:
		void ready(int damage, int effect, int range){
			this->damage = damage;
			this->effect = effect;
			this->range = range;
			return;
		}

		void set_damage(int damage){
			this->damage = damage;
			return;
		}

		int get_damage() const{
			return damage;
		}

		int get_effect() const{
			return effect;
		}

		int get_range() const{
			return range;
		}

		void upgrade(){
			++lvl;
			price += 500;
			damage += 50;
			effect -= 50;
			return;
		}

		void build_w(std::string s){
			std::ifstream f(s);
			f >> name >> price >> vol >> lvl >> stamina;
			f >> damage >> effect >> range;
			return;
		}
	};

	class Bullet: public Weapon{
	protected:
		int way;
		std::vector<int> cor, dcor;
		uintptr_t owner;
	public:
		uintptr_t get_owner() const{
			return owner;
		}

		void set_way(int way){
			this->way = way;
			return;
		}

		int get_way() const{
			return way;
		}

		std::vector<int> get_cor() const{
			return cor;
		}

		void set_cor(std::vector<int> cor){
			this->cor = cor;
			return;
		}

		void build_throw(std::string s){
			std::ifstream f(s);
			f >> name >> price >> vol >> lvl >> stamina;
			f >> damage >> effect >> range;
			return;
		}

		void shot(std::vector<int> cor_, int way, Weapon &w, uintptr_t owner){
			this->owner = owner;
			this->way = way, this->cor = cor_, this->dcor = cor_;
			name = w.get_name(), price = w.get_price();
			vol = w.get_vol(), lvl = w.get_level(), stamina = w.get_stamina();
			damage = w.get_damage(), effect = w.get_effect(), range = w.get_range();
			return;
		}

		bool expire(){
			int dist = abs(cor[0] - dcor[0]) + abs(cor[1] - dcor[1]) + abs(cor[2] - dcor[2]);
			return (dist + 1 >= range);
		}
	};

	ConsumableItem cons[4];
	Weapon w[8];
	Bullet throw_[4];

	void download_items(){
		std::string s[8] = {"0", "1", "2", "3", "4", "5", "6", "7"};
		for(int i = 0; i < 4; ++i)
			cons[i].build_cons("./Items/cons" + s[i] + ".txt");
		for(int i = 0; i < 4; ++i)
			throw_[i].build_throw("./Items/throw" + s[i] + ".txt");
		for(int i = 0; i < 8; ++i)
			w[i].build_w("./Items/w" + s[i] + ".txt");
		return;
	}


	ConsumableItem* gen_item(int i){
	    return &cons[i];
	}
}
