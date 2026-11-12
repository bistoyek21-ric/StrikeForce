/*
MIT License (c) 2025 bistoyek21 R.I.C.

HeadlessSummary.hpp
Pure C++17 helpers of the windowless runner (build flag -DHEADLESS_RUN, see docs §4):
    parse()          command-line options
    NullBuf          swallows all game text output (std::cout) during the run
    print_summary()  prints ONE result table from the per-game records of the engine (and saves .txt/.csv)
No SFML / libtorch dependency -> unit-tested in tests/test_headless_summary.cpp
*/
#pragma once
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace sf_headless {

struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
    std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
};

struct Options {
    int  episodes   = 100;
    std::string mode = "Solo";          // wall-clock modes (Timer) are not meaningful at full speed
    int  level      = 1;
    int  max_steps  = 3000;             // episode cap in AGENT STEPS (truncation: counted as "survived")
    unsigned seed   = 1;                // episode e uses seed+e  -> every model sees the SAME 100 maps
    int  threads    = 1;                // torch intra-op threads
    std::string user = "1";             // account whose backpack/loadout is used
    std::string out = "bots/bot-ppo/backup/results";   // directory for the saved result text
    std::string note;                    // extra line in the banner (e.g. DATA_GATHERING_MODE remarks)
    bool quiet = false;
    bool help  = false;
};

inline Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if      (a == "--episodes")   o.episodes   = std::stoi(next());
        else if (a == "--mode")       o.mode       = next();
        else if (a == "--level")      o.level      = std::stoi(next());
        else if (a == "--max-steps") o.max_steps = std::stoi(next());
        else if (a == "--seed")       o.seed       = (unsigned)std::stoul(next());
        else if (a == "--threads")    o.threads    = std::stoi(next());
        else if (a == "--user")       o.user       = next();
        else if (a == "--out")        o.out        = next();
        else if (a == "--quiet")      o.quiet      = true;
        else if (a == "--help" || a == "-h") o.help = true;
        else { std::cerr << "unknown option " << a << "\n"; o.help = true; }
    }
    return o;
}

inline const char* usage() {
    return "StrikeForce headless runner (no window, no menus, prints only the result)\n"
           "  --episodes N     number of games                (default 100)\n"
           "  --level L        game level 1..9                (default 1)\n"
           "  --mode Solo|Timer game mode: Solo ends at 5*level kills, Timer = level*3750 steps, uncapped kills (default Solo)\n"
           "  --max-steps N    cap per episode (agent steps)  (default 3000)\n"
           "  --seed S         first map seed; episode e uses S+e (default 1)\n"
           "  --threads K      torch CPU threads              (default 1)\n"
           "  --user NAME      account used for the loadout   (default 1)\n"
           "  --out DIR        where to save the result text  (default bots/bot-ppo/backup/results)\n"
           "  --quiet          no progress line on stderr\n";
}

inline std::vector<std::vector<std::string>> read_rows(const std::string& path) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream f(path);
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }
        std::vector<std::string> cells; std::stringstream ss(line); std::string c;
        while (std::getline(ss, c, ',')) cells.push_back(c);
        if (!cells.empty()) rows.push_back(cells);
    }
    return rows;
}
inline size_t count_rows(const std::string& path) { return read_rows(path).size(); }

struct Stat { double mean = 0, sd = 0; };
inline Stat stat(const std::vector<double>& v) {
    Stat s; if (v.empty()) return s;
    for (double x : v) s.mean += x;
    s.mean /= v.size();
    if (v.size() > 1) { for (double x : v) s.sd += (x - s.mean) * (x - s.mean); s.sd = std::sqrt(s.sd / (v.size() - 1)); }
    return s;
}

// One finished game, recorded by the ENGINE (gameplay.hpp), independent of which agent is plugged in.
// Definitions follow StrikeForce-Lab-Protocol.md: sums over the agent's decision steps while HP > 0.
struct Episode {
    long   T_alive = 0;       // decision steps with HP > 0
    int    N_kills = 0;
    double A_HP = 0, A_stamina = 0, A_damage = 0;
    bool   died = false;      // HP <= 0 at the end (false = survived / truncated by --max-steps)
    long   act[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // executed actions while alive, index into "+xzqeawsd"
};

inline std::string format_summary(const std::vector<Episode>& eps, const Options& o, double seconds,
                                  const std::string& agent) {
    std::vector<double> T, K, AH, AS, AD; int survived = 0, wins = 0, wins1 = 0; double steps = 0;
    for (const auto& e : eps) {
        T.push_back((double)e.T_alive); K.push_back(e.N_kills); AH.push_back(e.A_HP);
        AS.push_back(e.A_stamina); AD.push_back(e.A_damage); survived += !e.died; steps += e.T_alive;
        wins += (!e.died && e.N_kills >= 5 * o.level);
        wins1 += (!e.died && e.N_kills >= 1);
    }
    const int n = (int)eps.size();
    std::ostringstream s; s << std::fixed << std::setprecision(2);
    s << "==================== StrikeForce headless result ====================\n"
      << "agent : " << agent << "\n"
      << (o.note.empty() ? std::string() : "note  : " + o.note + "\n")
      << "mode=" << o.mode << " level=" << o.level << " episodes=" << n << "/" << o.episodes << " seeds=" << o.seed << ".." << o.seed + o.episodes - 1
      << " max_steps=" << o.max_steps << "\n"
      << "survival rate : " << (n ? (double)survived / n : 0.0) << "   (episodes ended alive)\n"
      << "win rate (alive & >=1 kill)        : " << (n ? (double)wins1 / n : 0.0) << "   <- the definition behind the old 0.60\n"
      << "win rate (alive & >=5*level = " << 5 * o.level << " kills): " << (n ? (double)wins / n : 0.0)
#if defined(DATA_GATHERING_MODE)
      << "   (threshold only: Solo has no kill cap in this mode)\n";
#else
      << "   (the game's own rule)\n";
#endif
    auto line = [&](const char* name, const std::vector<double>& v) { auto st = stat(v);
        s << std::left << std::setw(14) << name << ": " << std::right << std::setw(14) << st.mean << " +- " << std::setw(12) << st.sd << "\n"; };
    double total_kills = 0; for (double k : K) total_kills += k;
    s << "kills / 1000 steps : " << (steps > 0 ? 1000.0 * total_kills / steps : 0.0)
      << "   (comparable across modes and caps; bot-bc-fap reference: 34)\n";
    line("T_alive", T); line("N_kills", K); line("A_HP", AH); line("A_stamina", AS); line("A_damage", AD);
    { long tot[9] = {0}, all = 0; for (const auto& e : eps) for (int i = 0; i < 9; ++i) { tot[i] += e.act[i]; all += e.act[i]; }
      s << "executed actions (+xzqeawsd, %): ";
      for (int i = 0; i < 9; ++i) s << std::setw(6) << (all ? 100.0 * tot[i] / all : 0.0);
      s << "\n"; }
    s << "wall time     : " << seconds << " s   (" << (seconds > 0 ? steps / seconds : 0.0) << " agent-steps/s)\n"
      << "======================================================================\n";
    if (n == 0) s << "WARNING: no game finished - check the build flags and the working directory (must be StrikeForce-client/).\n";
    return s.str();
}

// Same columns as bots/bot-ppo/backup/metrics.csv, so tools/summarize_metrics.py can read it.
inline void write_episodes_csv(const std::string& path, const std::vector<Episode>& eps) {
    std::ofstream f(path);
    if (!f) return;
    f << "unix_time,T_alive,N_kills,A_HP,A_stamina,A_damage,died,return_total,return_kill,return_death,return_area,return_potential\n";
    for (const auto& e : eps)
        f << std::time(nullptr) << ',' << e.T_alive << ',' << e.N_kills << ',' << e.A_HP << ',' << e.A_stamina << ','
          << e.A_damage << ',' << (e.died ? 1 : 0) << ",,,,,\n";
}

inline void print_summary(const Options& o, const std::vector<Episode>& eps, double seconds,
                          const std::string& agent, std::ostream& os = std::cout) {
    std::string text = format_summary(eps, o, seconds, agent);
    os << text << std::flush;
    if (!o.out.empty()) {
        std::error_code ec; std::filesystem::create_directories(o.out, ec);
        const std::string stem = o.out + "/headless_" + std::to_string(std::time(nullptr));
        std::ofstream f(stem + ".txt"); if (f) f << text;
        write_episodes_csv(stem + ".csv", eps);
    }
}

} // namespace sf_headless
