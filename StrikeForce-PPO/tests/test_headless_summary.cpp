// Unit tests for StrikeForce-client/HeadlessSummary.hpp (option parsing + result table)
// Build: g++ -std=c++17 -O2 test_headless_summary.cpp -o t && ./t
#include <cstdio>
#include "../StrikeForce-client/HeadlessSummary.hpp"
using namespace sf_headless;
static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { ++g_fail; std::printf("  FAIL: %s\n", msg); } else std::printf("  ok:   %s\n", msg); } while (0)

int main() {
    std::puts("[1] option parsing");
    {
        const char* argv[] = {"x", "--episodes", "50", "--level", "3", "--max-steps", "800", "--seed", "7", "--threads", "4", "--quiet"};
        Options o = parse(12, (char**)argv);
        CHECK(o.episodes == 50 && o.level == 3 && o.max_steps == 800 && o.seed == 7 && o.threads == 4 && o.quiet && !o.help, "all options parsed");
        const char* bad[] = {"x", "--nope"};
        CHECK(parse(2, (char**)bad).help, "unknown option -> help");
    }
    std::puts("[2] result table from engine-recorded episodes");
    {
        std::vector<Episode> eps(2);
        eps[0] = {300, 1, 300000, 600000, 60000, false};
        eps[1] = {100, 0, 100000, 200000, 20000, true};
        Options o; o.episodes = 2; o.out = "";
        std::string t = format_summary(eps, o, 2.0, "unit-test-agent");
        CHECK(t.find("agent : unit-test-agent") != std::string::npos, "agent name printed");
        CHECK(t.find("episodes=2/2") != std::string::npos, "2 of 2 episodes reported");
        CHECK(t.find("survival rate : 0.50") != std::string::npos, "survival rate 0.50");
        CHECK(t.find("win rate (alive & >=1 kill)        : 0.50") != std::string::npos, "old definition (alive & >=1 kill) is printed: 0.50");
        CHECK(t.find("win rate (alive & >=5*level = 5 kills): 0.00") != std::string::npos, "game rule (5*level kills) is printed: 0.00");
        CHECK(t.find("T_alive       :         200.00") != std::string::npos, "mean T_alive 200");
        CHECK(t.find("(200.00 agent-steps/s)") != std::string::npos, "throughput 400 steps / 2 s");
        CHECK(t.find("WARNING") == std::string::npos, "no warning when episodes exist");
        std::string e0 = format_summary({}, o, 1.0, "x");
        CHECK(e0.find("WARNING: no game finished") != std::string::npos, "explicit warning when nothing finished");
        write_episodes_csv("/tmp/sf_hs_eps.csv", eps);
        auto rows = read_rows("/tmp/sf_hs_eps.csv");
        CHECK(rows.size() == 2 && rows[1][6] == "1", "episodes csv written (died flag in column 7)");
    }
    std::puts("[3] NullBuf swallows output");
    {
        NullBuf nb; std::ostream os(&nb); os << "hello" << 123 << std::endl;
        CHECK(os.good(), "stream stays good, nothing printed");
    }
    std::printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
