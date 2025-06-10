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
#include "selected.hpp"

void history(){
	std::string s1;
	std::ifstream games("./accounts/game/" + user + "/" + user + ".txt");
	std::cout << head();
	std::vector<std::string> act;
	while(getline(games, s1))
		act.push_back(s1);
	std::cout << "Your history:" << '\n';
	std::cout << "________________________________________________________________" << '\n';
	std::cout << "|";
	std::cout << c_col(34, 40);
	std::cout << "Time";
	std::cout << c_col(0, 0);
	std::cout << "                    |";
	std::cout << c_col(34, 40);
	std::cout << "Game's Mode";
	std::cout << c_col(0, 0);
	std::cout << "|";
	std::cout << c_col(34, 40);
	std::cout << "Kills     ";
	std::cout << c_col(0, 0);
	std::cout << "|";
	std::cout << c_col(34, 40);
	std::cout << "Rating Changes";
	std::cout << c_col(0, 0);
	std::cout << "|" << '\n';
	for(int i = 0; true; ++i){
		std::cout << "|________________________|___________|__________|______________|" << '\n';
		if(i == act.size())
			break;
		std::cout << "|";
		std::cout << c_col(32, 40);
		std::cout << act[i];
		std::cout << c_col(0, 0);
		std::cout << "|";
		std::cout << c_col(32, 40);
		++i;
		std::cout << (char)toupper(act[i][0]);
		for(int j = 1; j < 11; ++j){
			if(j < act[i].size())
				std::cout << act[i][j];
			else
				std::cout << " ";
		}
		std::cout << c_col(0, 0);
		std::cout << "|";
		std::cout << c_col(32, 40);
		++i;
		for(int j = 0; j < 10; ++j){
			if(j < act[i].size())
				std::cout << act[i][j];
			else if(j == act[i].size())
				std::cout << " ";
			else
				std::cout << " ";
		}
		std::cout << c_col(0, 0);
		std::cout << "|";
		std::cout << c_col(32, 40);
		++i;
		std::cout << act[i];
		for(int j = 0; j < 14 - (int)act[i].size(); ++j)
			std::cout << " ";
		std::cout << c_col(0, 0);
		std::cout << "|" << '\n';
	}
	std::cout << "\nTo back into the menu press any key" << std::endl;
	getch();
	return;
}

void leaderboard(std::string mode){
	std::ifstream ranking("./accounts/ranking" + mode + ".txt");
	std::string s1, s2;
	std::vector<std::pair<std::string, int>> standing;
	while(getline(ranking, s1)){
		int num;
		ranking >> num;
		standing.push_back({s1, num});
		getline(ranking, s1);
	}
	sort(standing.begin(), standing.end(), [&](std::pair<std::string, int> e1, std::pair<std::string, int> e2){return e1.second > e2.second;});
	std::cout << head();
	std::cout << "_______________________________________" << '\n';
	std::cout << "|";
	std::cout << c_col(34, 40);
	std::cout << "Ranking";
	std::cout << c_col(0, 0);
	std::cout << "|";
	std::cout << c_col(34, 40);
	std::cout << "Handle                ";
	std::cout << c_col(0, 0);
	std::cout << "|";
	std::cout << c_col(34, 40);
	std::cout << "Rating";
	std::cout << c_col(0, 0);
	std::cout << "|" << '\n';
	for(int i = 0; true; ++i){
		std::cout << "|_______|______________________|______|" << '\n';
		if(i == standing.size())
			break;
		int l;
		std::cout << "|";
		std::cout << c_col(32, 40);
		std::cout << "#" << i + 1;
		l = std::to_string(i + 1).size();
		for(int j = 0; j < 6 - l; ++j)
			std::cout << " ";
		std::cout << c_col(0, 0);
		std::cout << "|";
		std::cout << c_col(32, 40);
		for(int j = 0; j < 22; ++j)
			if(j < standing[i].first.size())
				std::cout << standing[i].first[j];
			else
				std::cout << " ";
		std::cout << c_col(0, 0);
		std::cout << "|";
		std::cout << c_col(32, 40);
		std::cout << standing[i].second;
		l = std::to_string(standing[i].second).size();
		for(int j = 0; j < 6 - l; ++j)
			std::cout << " ";
		std::cout << c_col(0, 0);
		std::cout << "|" << '\n';
	}
	std::cout << "\n----------------------------------------\n";
	std::cout << "To back see a leaderboards you press:" << '\n';
	std::cout << "  1. total leader board" << '\n';
	std::cout << "  2. leader board in solo mode" << '\n';
	std::cout << "  3. leader board in timer mode" << '\n';
	std::cout << "  4. leader board in squad mode" << '\n';
	std::cout << "\nTo back into the menu press any other key" << std::endl;
	char c = getch();
	if(c == '1')
		leaderboard("");
	else if(c == '2')
		leaderboard("solo");
	else if(c == '3')
		leaderboard("timer");
	else if(c == '4')
		leaderboard("squad");
	return;
}


namespace Environment::Character{

	void shop(){
		while(true){
	        me.save_progress();
			std::cout << head();
			std::cout << "Your money: " << me.get_money() << "$\n";
			std::cout << "Accopied volume of your backpack: " << me.backpack.get_vol() << " / " << me.backpack.get_capacity() << "\n\n";
			std::cout << "Backpack:" << '\n';
			std::cout << "(*)=============(*)" << '\n';
			std::cout << "|*|backpack     |*|" << '\n';
			std::cout << "|*|             |*|" << '\n';
			std::cout << "|*|code:0       |*|" << '\n';
			std::cout << "(*)=============(*)" << '\n';
			std::cout << "Items:" << '\n';
			std::cout << "(*)[Consumable:](*)[Throwable: ](*)[ColdWeapon:](*)[WarmWeapon:](*)" << '\n';
			std::cout << "|*|energy_drink |*|gas          |*|push_dagger  |*|AK_47        |*|" << '\n';
			std::cout << "|*|             |*|             |*|             |*|             |*|" << '\n';
			std::cout << "|*|code:1       |*|code:5       |*|code:9       |*|code:13      |*|" << '\n';
			std::cout << "(*)=============(*)=============(*)=============(*)=============(*)" << '\n';
			std::cout << "|*|first_aid_box|*|flash_bang   |*|wing_tactic  |*|M416         |*|" << '\n';
			std::cout << "|*|             |*|             |*|             |*|             |*|" << '\n';
			std::cout << "|*|code:2       |*|code:6       |*|code:10      |*|code:14      |*|" << '\n';
			std::cout << "(*)=============(*)=============(*)=============(*)=============(*)" << '\n';
			std::cout << "|*|food_package |*|acid_bomb    |*|F_898        |*|MOSSBERG     |*|" << '\n';
			std::cout << "|*|             |*|             |*|             |*|             |*|" << '\n';
			std::cout << "|*|code:3       |*|code:7       |*|code:11      |*|code:15      |*|" << '\n';
			std::cout << "(*)=============(*)=============(*)=============(*)=============(*)" << '\n';
			std::cout << "|*|zombievaccine|*|stinger      |*|lochabreask  |*|AWM          |*|" << '\n';
			std::cout << "|*|             |*|             |*|             |*|             |*|" << '\n';
			std::cout << "|*|code:4       |*|code:8       |*|code:12      |*|code:16      |*|" << '\n';
			std::cout << "(*)=============(*)=============(*)=============(*)=============(*)" << '\n';
			std::cout << "\n-------------------------------------------\n";
			std::cout << "You can write 17 to see your backpack" << '\n';
			std::cout << "Enter the code of any Item you want to buy/upgrade or write -1 for back to menu";
			std::cout << "\n-------------------------------------------" << std::endl;
			std::string code;
			char ans;
			std::cin >> code;
			bool isdone = false;
			if(code == "-1")
				return;
			if(code == "17"){
				me.backpack.show(me.get_money());
				continue;
			}
			if(code == "0"){
				std::cout << "Do you want to upgrade your backpac? (y: yes/any other key: no)\n";
				std::cout << " Backpack level: " << me.backpack.get_level() << " -> " << me.backpack.get_level() + 1 << '\n';
				std::cout << " Backpack capacity: " << me.backpack.get_capacity() << " -> " << me.backpack.get_capacity() + 200 << '\n';
				std::cout << "---------------------------\n";
				std::cout << " price: " << me.backpack.get_price() << std::endl;
				ans = getch();
				if(ans == 'y'){
					if(me.get_money() >= me.backpack.get_price()){
						me.set_money(me.get_money() - me.backpack.get_price());
						me.backpack.upgrade();
						std::cout << "you upgrade back pack successfully!" << std::endl;
					}
					else
						std::cout << "your money isn't enough" << std::endl;
				}
				std::cout << "press any key to continue" << std::endl;
				isdone = true;
				getch();
				continue;
			}
			for(int i = 0; i < 4; ++i)
				if(code == std::to_string(i + 1)){
					std::cout << "Do you want to buy " << me.backpack.list_cons[i].first.get_name() << "(y:yes / any other key:no)?" << '\n';
					std::cout << "volume : " << me.backpack.list_cons[i].first.get_vol() << '\n';
					std::cout << "---------------------------\n";
					std::cout << " price: " << me.backpack.list_cons[i].first.get_price() << std::endl;
					ans = getch();
					if(ans == 'y'){
						if(me.get_money() >= me.backpack.get_price() && me.backpack.get_vol() + me.backpack.list_cons[i].first.get_vol() <= me.backpack.get_capacity()){
							me.set_money(me.get_money() - me.backpack.list_cons[i].first.get_price());
							++me.backpack.list_cons[i].second;
							me.backpack.set_vol(me.backpack.get_vol() + me.backpack.list_cons[i].first.get_vol());
							std::cout << "you bought " << me.backpack.list_cons[i].first.get_name() << " successfully!" << std::endl;
						}
						else
							std::cout << "your money isn't enough or your backpack doesn't have free space :(" << std::endl;
					}
					std::cout << "press any key to continue" << std::endl;
					isdone = true;
					getch();
					break;
				}
			if(isdone)
				continue;
			for(int i = 0; i < 4; ++i)
				if(code == std::to_string(i + 5)){
        	        std::cout << "Do you want to buy or upgrade? (b:buy / u:upgrade / any other key:nothing)\n";
					std::cout << " " << me.backpack.list_throw[i].first.get_name() << " level: "
						<< me.backpack.list_throw[i].first.get_level() << " -> " << me.backpack.list_throw[i].first.get_level() + 1 << '\n';
					std::cout << " " << me.backpack.list_throw[i].first.get_name() << " damage: "
						<< me.backpack.list_throw[i].first.get_damage() << " -> " << me.backpack.list_throw[i].first.get_damage() + 50 << '\n';
					std::cout << " " << me.backpack.list_throw[i].first.get_name() << " effect: "
						<< me.backpack.list_throw[i].first.get_effect() << " -> " << me.backpack.list_throw[i].first.get_effect() - 50 << '\n';
					std::cout << "volume : " << me.backpack.list_throw[i].first.get_vol() << '\n';
					std::cout << "---------------------------\n";
					std::cout << " price: " << me.backpack.list_throw[i].first.get_price() << std::endl;
					ans = getch();
					if(ans == 'u'){
						if(me.get_money() >= me.backpack.list_throw[i].first.get_price()){
							++me.backpack.list_throw[i].second.first;
							me.set_money(me.get_money() - me.backpack.list_throw[i].first.get_price());
							std::cout << "you upgraded " << me.backpack.list_throw[i].first.get_name() << " successfully!" << std::endl;
							me.backpack.list_throw[i].first.upgrade();
						}
						else
							std::cout << "your money isn't enough :(" << std::endl;
					}
					if(ans == 'b'){
						if(me.get_money() >= me.backpack.list_throw[i].first.get_price() && me.backpack.get_vol() + me.backpack.list_throw[i].first.get_vol() <= me.backpack.get_capacity()){
							++me.backpack.list_throw[i].second.second;
							me.set_money(me.get_money() - me.backpack.list_throw[i].first.get_price());
							me.backpack.set_vol(me.backpack.get_vol() + me.backpack.list_throw[i].first.get_vol());
							std::cout << "you bought " << me.backpack.list_throw[i].first.get_name() << " successfully!" << std::endl;
						}
						else
							std::cout << "your money isn't enough or your backpack doesn't have free space :(" << std::endl;
					}
					std::cout << "press any key to continue" << std::endl;
					isdone = true;
					getch();
					break;
				}
			if(isdone)
				continue;
			for(int i = 0; i < 8; ++i)
				if(code == std::to_string(i + 9)){
					if(me.backpack.list_w[i].second){
						std::cout << "Do you want to upgrade (y: yes / any other key: no)?\n";
						std::cout << " " << me.backpack.list_w[i].first.get_name() << " level: "
							<< me.backpack.list_w[i].first.get_level() << " -> " << me.backpack.list_w[i].first.get_level() + 1 << '\n';
						std::cout << " " << me.backpack.list_w[i].first.get_name() << " damage: "
							<<me.backpack.list_w[i].first.get_damage() << " -> " << me.backpack.list_w[i].first.get_damage() + 50 << '\n';
						std::cout << " " << me.backpack.list_w[i].first.get_name() << " effect: "
							<< me.backpack.list_w[i].first.get_effect() << " -> " << me.backpack.list_w[i].first.get_effect() - 50 << std::endl;
					}
					else{
						std::cout << "Do you want to buy(y: yes / any other key: no)?\n";
						std::cout << " " << me.backpack.list_w[i].first.get_name() << " level: "
							<< me.backpack.list_w[i].first.get_level() + 1 << '\n';
						std::cout << " " << me.backpack.list_w[i].first.get_name() << " damage: "
							<< me.backpack.list_w[i].first.get_damage() + 50 << '\n';
						std::cout << " " << me.backpack.list_w[i].first.get_name() << " effect: "
							<< me.backpack.list_w[i].first.get_effect() - 50 << '\n';
						std::cout << "volume : " << me.backpack.list_w[i].first.get_vol() << std::endl;
					}
					std::cout << "---------------------------\n";
					std::cout << " price: " << me.backpack.list_w[i].first.get_price() << std::endl;
					ans = getch();
					if(ans == 'y'){
						if(me.backpack.get_vol() + me.backpack.list_w[i].first.get_vol() > me.backpack.get_capacity())
							std::cout << "you dont have enought free space :(" << std::endl;
						else if(me.get_money() >= me.backpack.list_w[i].first.get_price()){
							bool buy = (me.backpack.list_w[i].second == 0);
							++me.backpack.list_w[i].second;
							if(buy)
								me.backpack.set_vol(me.backpack.get_vol() + me.backpack.list_w[i].first.get_vol());
							me.set_money(me.get_money() - me.backpack.list_w[i].first.get_price());
							std::cout << "you " << (buy ? "bought " : "upgraded ") << me.backpack.list_w[i].first.get_name() << " successfully!" << std::endl;
							me.backpack.list_w[i].first.upgrade();
						}
						else
							std::cout << "your money isn't enough :(" << std::endl;
					}
					std::cout << "press any key to continue" << std::endl;
					isdone = true;
					getch();
					break;
				}
			if(isdone)
				continue;
		}
		return;
	}

    void menu(){
        while(true){
            std::cout << head();
            std::cout << "Main Menu:" << '\n';
            std::cout << "  1. Play" << '\n';
            std::cout << "  2. Shop" << '\n';
            std::cout << "  3. Leaderboard" << '\n';
            std::cout << "  4. History" << '\n';
            std::cout << "  5. Show Back pack" << '\n';
            std::cout << "  6. Sign out" << '\n';
            std::cout << "  7. Clear history and sign out" << '\n';
            std::cout << "  8. Exit" << '\n';
            std::cout << "\n----------------------------\n";
            std::cout << "to choose an option write it's section number" << '\n';
            std::cout << "----------------------------" << std::endl;
            char c = getch();
            if(c == '1')
                Environment::Field::g.open();
            if(c == '2')
                shop();
            if(c == '3')
                leaderboard("");
            if(c == '4')
                history();
            if(c == '5')
                me.show_backpack();
            if(c == '6'){
                std::cout << head();
                std::cout << "are you sure you want to sign out? (n: no / any other key: yes)" << std::endl;
                if(getch() == 'n')
                    continue;
                std::ofstream slo("./accounts/saved_login.txt");
                slo << "" << '\n' << -1 << '\n';
                slo.close();
                user = "";
                return;
            }
            if(c == '7'){
				std::cout << head();
                std::cout << "Are you sure you want to clear history and sign out?\n";
                std::cout << "If you sure, please write \"I'm sure about what I'm doing\"\n";
                std::cout << ">> ";
                std::cout.flush();
                std::string s;
                getline(std::cin, s);
                if(s != "I'm sure about what I'm doing")
                    continue;
                #if defined(__unix__) || defined(__APPLE__)
				system("rm -fr ./accounts/game/");
				#else
				system("rmdir /s /q .\\accounts\\game\\ > nul");
				#endif
				system("mkdir ./accounts/game/");
                std::ofstream f("./accounts/game/.gitkeep");
                f.close();
                std::vector<std::string> mode = {"", "solo", "timer", "squad"};
                for(auto &e: mode){
                    std::ofstream rnk("./accounts/ranking" + e + ".txt");
                    rnk.close();
                }
                std::ofstream pass("./accounts/pass.txt");
                pass.close();
                std::ofstream usr("./accounts/users.txt");
                usr.close();
                std::ofstream slo("./accounts/saved_login.txt");
                slo << "" << '\n' << -1 << '\n';
                slo.close();
                user = "";
                return;
            }
            if(c == '8')
                exit(0);
        }
        return;
    }
}
