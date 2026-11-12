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
#include "enter.hpp"

#if defined(HEADLESS_RUN) && !defined(SHAPED_REWARD)
#pragma message("HEADLESS_RUN: the selected agent is not bots/bot-ppo (check selected_agent.hpp and selected_custom.hpp)")
#endif

#if defined(HEADLESS_RUN)
// ── Windowless batch runner ────────────────────────────────────────────────
// Build with:  -DHIGHLY_OPTIMIZED -DAUTOMATIC -DHEADLESS_RUN   (see tools/pipeline.py build --headless)
// Runs N games back-to-back with no window, no menu, no keyboard and no frame limiter,
// then prints ONE result table.  Training or evaluation depends only on FREEZE_AGENT_BLOCK.
int headless_main(int argc, char** argv){
	sf_headless::Options o = sf_headless::parse(argc, argv);
	if(o.help){
		std::cout << sf_headless::usage();
		return 0;
	}
	#if defined(HEADLESS_SET_THREADS)
	at::set_num_threads(o.threads);
	at::set_num_interop_threads(1);
	#endif
	#if defined(DATA_GATHERING_MODE)
	o.note = "DATA_GATHERING_MODE: Solo has NO kill cap (a game ends by death or --max-steps); map seeds = --seed + game index";
	#endif
	user = o.user;
	Environment::Character::recovery();
	auto t0 = std::chrono::steady_clock::now();
	Environment::Field::g.headless_run(o.episodes, o.mode, o.level, 2 * o.max_steps, o.seed, o.quiet);
	double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	#if defined(SHAPED_REWARD)
	Agent::flush();        // write model / optimizer / critic statistics once at the end of the batch
	#if defined(EXPERT_POLICY)
	std::string agent_name = "SCRIPTED TEACHER (no neural network, no checkpoint, no learning)";
	#else
	std::string agent_name = "bot-ppo (PPO + shaped reward) | " + Agent::status();
	#endif
	#else
	std::string agent_name = "NOT bot-ppo - selected_agent.hpp / selected_custom.hpp do not point to bots/bot-ppo";
	#endif
	sf_headless::print_summary(o, Environment::Field::g.headless_results, secs, agent_name);
	return 0;
}
#endif

signed main(int argc, char** argv){
	std::ios::sync_with_stdio(false), std::cin.tie(nullptr);
	Environment::Item::download_items();
	Environment::Random::make_p();
#if defined(HEADLESS_RUN)
	return headless_main(argc, argv);
#else
	enter();
	return 0;
#endif
}
