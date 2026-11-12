// Unit tests for bots/bot-ppo/PPOMath.hpp (GAE, explained variance, PopArt)
// Build: g++ -std=c++17 -O2 test_ppo_math.cpp -o test_ppo_math && ./test_ppo_math
#include <cstdio>
#include <random>
#include "../StrikeForce-client/bots/bot-ppo/PPOMath.hpp"
using namespace sf_ppo;
static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { ++g_fail; std::printf("  FAIL: %s\n", msg); } else std::printf("  ok:   %s\n", msg); } while (0)

int main() {
    std::puts("[1] GAE");
    {   // lambda = 1  ->  A_t = discounted Monte-Carlo return (+ bootstrap) - V_t
        std::vector<float> r = {1, 0, 2, 0.5f}, v = {0.3f, 0.2f, 0.9f, 0.1f}, adv, ret;
        float g = 0.9f, last = 0.7f;
        gae(r, v, last, g, 1.0f, adv, ret);
        double G = last; bool ok = true;
        for (int t = 3; t >= 0; --t) { G = r[t] + g * G; ok &= std::fabs(ret[t] - G) < 1e-5; }
        CHECK(ok, "lambda=1: critic target equals the discounted Monte-Carlo return");
        gae(r, v, last, g, 0.0f, adv, ret);        // lambda = 0 -> one-step TD error
        bool td = true;
        for (int t = 0; t < 4; ++t) {
            float nv = t == 3 ? last : v[t + 1];
            td &= std::fabs(adv[t] - (r[t] + g * nv - v[t])) < 1e-6;
        }
        CHECK(td, "lambda=0: advantage equals the one-step TD error");
    }
    std::puts("[2] explained variance");
    {
        std::vector<float> ret = {1, 2, 3, 4, 5}, perfect = ret, constant(5, 3.f);
        CHECK(std::fabs(explained_variance(perfect, ret) - 1.f) < 1e-6, "perfect critic -> 1");
        CHECK(std::fabs(explained_variance(constant, ret)) < 1e-6, "constant (mean) critic -> 0");
    }
    std::puts("[3] PopArt keeps every prediction unchanged");
    {
        std::mt19937 g(7); std::normal_distribution<double> N(0, 1);
        double worst = 0;
        for (int trial = 0; trial < 500; ++trial) {
            int d = 8; std::vector<double> w(d), x(d); double b = N(g);
            for (auto& a : w) a = N(g);
            for (auto& a : x) a = N(g);
            double mu = N(g), sigma = 0.1 + std::fabs(N(g)), mu2 = N(g), sigma2 = 0.1 + std::fabs(N(g));
            auto head = [&](const std::vector<double>& ww, double bb) { double y = bb; for (int i = 0; i < d; ++i) y += ww[i] * x[i]; return y; };
            double before = mu + sigma * head(w, b);
            popart_rescale(w, b, mu, sigma, mu2, sigma2);
            double after = mu2 + sigma2 * head(w, b);
            worst = std::max(worst, std::fabs(before - after));
        }
        std::printf("      max |V_before - V_after| over 500 random layers = %.2e\n", worst);
        CHECK(worst < 1e-9, "outputs preserved to round-off");
    }
    std::puts("[4] ValueNorm statistics");
    {
        ValueNorm vn; std::vector<float> ret(1000); std::mt19937 g(3); std::normal_distribution<float> N(0.4f, 0.2f);
        for (auto& r : ret) r = N(g);
        double om, os; vn.update(ret, 0.1, 0.01, /*first=*/true, om, os);
        CHECK(std::fabs(vn.mu - 0.4) < 0.02 && std::fabs(vn.sigma - 0.2) < 0.02, "first update adopts batch mean/std (~0.4/0.2)");
        CHECK(om == 0.0 && os == 1.0, "previous (mu,sigma) returned for the rescale");
        std::vector<float> same(100, 5.f);   // zero variance batch must not collapse sigma
        ValueNorm v2; v2.update(same, 1.0, 0.01, true, om, os);
        CHECK(v2.sigma >= 0.01 - 1e-12, "sigma floor respected");
        CHECK(std::fabs(vn.normalize(vn.denorm(0.37f)) - 0.37f) < 1e-5, "normalize(denorm(x)) == x");
    }
    std::puts("[5] teacher schedule");
    {
        CHECK(teacher_schedule(0, 150) == 1.f, "starts at 1");
        CHECK(std::fabs(teacher_schedule(75, 150) - 0.5f) < 1e-6, "halfway at half the decay");
        CHECK(teacher_schedule(150, 150) == 0.f && teacher_schedule(10000, 150) == 0.f, "0 after the decay, never negative");
        CHECK(teacher_schedule(5, 0) == 0.f, "decay 0 disables the teacher");
    }
    std::printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
