/*
MIT License (c) 2025 bistoyek21 R.I.C.

Custom.hpp — bot-1
Observation extraction is unchanged from bot-1/bot-1-redesigned.
The 9-action space is sufficient and well-proven.
Raw channel values are passed to the agent which applies
running-stat normalisation internally.
*/
#pragma once
#include "../../gameplay.hpp"

namespace Environment::Field {

    // Returns a 32-element feature vector for one grid cell.
    // Channels match the 32-channel layout that the CNN expects.
    static std::vector<float> describe(
        const node& cell,
        const Environment::Character::Human& player)
    {
        std::vector<float> res;
        res.reserve(32);

        // ── [0-6] Object types (7) ──────────────
        res.push_back((float)(cell.s[0] || cell.s[1]));   // char or zombie
        res.push_back((float)cell.s[2]);                   // bullet
        res.push_back((float)cell.s[3]);                   // wall
        res.push_back((float)cell.s[4]);                   // chest
        res.push_back((float)(cell.s[5] || cell.s[6]));   // portal
        res.push_back((float)cell.s[7]);                   // tmp object
        res.push_back((float)cell.s[10]);                  // destructible

        // ── [7-10] Team membership (4) ──────────
        float ally = 0, enemy = 0, npc = 0, zombie = 0;
        if (cell.s[0]) {
            int t = cell.human->get_team();
            if (!t)                       npc   = 1;
            else if (t == player.get_team()) ally  = 1;
            else                          enemy = 1;
        }
        if (cell.s[1]) zombie = 1;
        res.push_back(ally);
        res.push_back(enemy);
        res.push_back(npc);
        res.push_back(zombie);

        // ── [11-14] Character stats (4) ─────────
        if (cell.s[0]) {
            res.push_back(cell.human->get_kills()              / 100.f);
            res.push_back(cell.human->backpack.get_blocks()    /  10.f);
            res.push_back(cell.human->backpack.get_portals()   /  10.f);
            res.push_back(cell.human->backpack.get_portal_ind() != -1 ? 1.f : 0.f);
        } else {
            res.push_back(0); res.push_back(0);
            res.push_back(0); res.push_back(0);
        }

        // ── [15-18] Passability + HP (4) ────────
        float nb = 0, nb_b = 0, dest = 0, hp = 0;
        if (cell.s[3] || cell.s[5] || cell.s[6] || cell.s[0] || cell.s[1]) {
            nb = nb_b = 1;
            dest = (float)(cell.s[10] || cell.s[0] || cell.s[1]);
            if      (cell.s[0])  hp = cell.human->get_Hp()   / 1000.f;
            else if (cell.s[1])  hp = cell.zombie->get_Hp()  / 1000.f;
            else if (cell.s[10]) {
                float cap = cell.s[3] ? (float)lim_block : (float)lim_portal;
                hp = (cap - cell.dmg) / 1000.f;
            }
        } else if (cell.s[7]) {
            nb = 1;
        }
        res.push_back(nb); res.push_back(nb_b);
        res.push_back(dest); res.push_back(hp);

        // ── [19-26] Attack properties (8) ───────
        float dir[4] = {0,0,0,0};
        float dmg = 0, eff = 0, is_bull = 0, stam = 0;

        if (cell.s[0]) {
            dir[cell.human->get_way() - 1] = 1.f;
            auto de = cell.human->get_damage_effect();
            dmg  =  de[0] / 1000.f;
            eff  = -de[1] / 1000.f;
            stam = cell.human->get_stamina() / 1000.f;
        } else if (cell.s[1]) {
            dir[0] = dir[1] = dir[2] = dir[3] = 0.01f;
            dmg = cell.zombie->get_mindamage() / 1000.f;
        } else if (cell.s[2]) {
            is_bull = 1.f;
            auto dc = cell.bullet->get_dcor();
            auto cc = cell.bullet->get_cor();
            int dist = std::abs(cc[1]-dc[1]) + std::abs(cc[2]-dc[2]);
            dir[cell.bullet->get_way() - 1] =
                (cell.bullet->get_range() - dist) / 100.f;
            dmg =  cell.bullet->get_damage() / 1000.f;
            eff = -cell.bullet->get_effect() / 1000.f;
        } else if (cell.s[7]) {
            dmg = 20.f / 1000.f; eff = 10.f / 1000.f;
        }

        res.push_back(is_bull);
        res.push_back(dir[0]); res.push_back(dir[1]);
        res.push_back(dir[2]); res.push_back(dir[3]);
        res.push_back(dmg); res.push_back(eff); res.push_back(stam);

        // ── [27-29] Consumable items (3) ────────
        float cstam = 0, ceff = 0, chp = 0;
        if (cell.s[4]) {
            cstam = cell.cons->get_stamina() / 1000.f;
            ceff  = cell.cons->get_effect()  / 1000.f;
            chp   = cell.cons->get_Hp()      / 1000.f;
        }
        res.push_back(cstam); res.push_back(ceff); res.push_back(chp);

        // ── [30-31] Status effects (2) ──────────
        if (cell.s[0]) {
            res.push_back( cell.human->get_damage() / 1000.f);
            res.push_back(-cell.human->get_effect() / 1000.f);
        } else {
            res.push_back(0); res.push_back(0);
        }

        // assert(res.size() == 32);
        return res;
    }

    // ─────────────────────────────────────────
    char gameplay::bot(Environment::Character::Human& player) const {
        if (!player.get_active_agent()) return '+';

        auto pos    = player.get_cor();
        const int r = 15;   // 31/2

        std::array<std::vector<float>, 32> channels;

        for (int i = pos[1] - r; i <= pos[1] + r; ++i)
            for (int j = pos[2] - r; j <= pos[2] + r; ++j) {
                const auto& cell = (i < 0 || j < 0 || N <= i || M <= j)
                    ? nd
                    : themap[pos[0]][i][j];
                auto feat = describe(cell, player);
                for (int c = 0; c < 32; ++c)
                    channels[c].push_back(feat[c]);
            }

        // Flatten to [32 × 31 × 31] — raw values, agent normalises internally
        std::vector<float> obs;
        obs.reserve(32 * 31 * 31);
        for (int c = 0; c < 32; ++c)
            for (float v : channels[c])
				if (5.0f <= v)
                	obs.push_back(10.0f / (1 + std::exp(5.0f - v)));
				else
					obs.push_back(v);

        return action[player.agent->predict(obs)];
    }

    void gameplay::prepare(Environment::Character::Human& player) {
        action = "+xzqeawsd";          // 9 actions — unchanged
        player.agent = new Agent(/*training=*/true);
        player.set_agent_active();
    }

    void gameplay::view() const {}

}  // namespace Environment::Field