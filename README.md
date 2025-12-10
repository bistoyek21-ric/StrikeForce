# StrikeForce: AI Agent Battle Arena 🎮🤖

> A research-oriented multiplayer battle game where you train AI agents to compete in a zombie-infested arena!

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## 🌟 Overview

**StrikeForce** is not just a game—it's a comprehensive AI training environment disguised as an intense survival shooter! Battle zombies, compete against other players, and most importantly: **train your own AI agent** to master the battlefield.

Whether you're an AI researcher looking for a challenging RL environment, a student learning about neural networks, or just someone who loves competitive games, StrikeForce offers something unique:

- 🧠 **Train custom AI agents** using PyTorch and reinforcement learning
- 🎯 **Multiple game modes**: Solo, Timer, Squad, and Battle Royale
- 🌐 **Online multiplayer** support with dedicated server
- 💾 **Crowdsourced training** - share and improve AI models collaboratively
- 🎨 **Visual interface** powered by SFML with real-time rendering
- 🔄 **Version control** for AI checkpoints via centralized API server

---

## 📋 Table of Contents

1. [Introduction](#-introduction)
2. [Quick Start](#-quick-start)
3. [Game Features](#-game-features)
4. [Creating Custom AI Agents](#-creating-custom-ai-agents)
5. [Observation Space Design](#-observation-space-design)
6. [Training Your Agent](#-training-your-agent)
7. [Server & Online Play](#-server--online-play)
8. [API Server for Version Control](#-api-server-for-version-control)
9. [Custom Map Design](#-custom-map-design)
10. [Project Structure](#-project-structure)
11. [Research Context](#-research-context)
12. [Technical Details](#-technical-details)

---

## 🎮 Introduction

**StrikeForce** is a tactical 2D environment designed for research in imitation learning, human-AI interaction, and crowdsourced training. This guide focuses on **how to use and customize the environment** for your own research needs.

### Key Features

- 🎯 **Fully Customizable**: Modify observations, actions, maps, and agent architectures
- 🧠 **Research-Ready**: Built for imitation learning and human-in-the-loop experiments
- 💾 **Resource-Efficient**: Runs on consumer CPUs with <100MB RAM (idle), <3.2GB (training)
- 🌐 **Crowdsourcing-Ready**: Built-in API server for distributed training
- 📊 **Reproducible**: Deterministic simulation with seed control

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install g++ libsfml-dev libtorch-dev

# macOS
brew install sfml libtorch

# Windows: Download SFML and LibTorch manually
```

### Test Account (Skip Tutorial)

Don't want to start from scratch? Use this pre-configured account:

- **Username**: `1`
- **Password**: `1`

This account has already progressed through early levels and has some items unlocked!

### Build & Run

```bash
cd StrikeForce-client
g++ -std=c++17 main.cpp -o StrikeForce \
    -lsfml-graphics -lsfml-window -lsfml-system \
    -ltorch -ltorch_cpu -lc10

./StrikeForce
```

### First Steps

1. **Create an account** (or use the test account above)
2. **Try Solo Mode** - Learn the controls manually
3. **Visit the Shop** - Buy weapons and items
4. **Play with AI** - Let your agent learn!

---

## 🎮 Game Features

### Game Modes

| Mode | Description | Victory Condition |
|------|-------------|-------------------|
| **Solo** | Face increasing waves of zombies | Kill 5×level enemies |
| **Timer** | Survival against the clock | Stay alive + meet kill quota |
| **Squad** | 5v5 team battles with NPCs | Eliminate rival team |
| **Battle Royale** | Online multiplayer chaos | Last team standing |
| **AI Battle Royale** | Watch AI agents fight! | Spectate mode |

### Action Space Reference (30 actions)

#### Movement & Orientation (9 actions)
```cpp
'+' : No-op (do nothing)
'w' : Move up
's' : Move down
'a' : Move left
'd' : Move right
'`' : Turn left (rotate counterclockwise)
'1' : Turn right (rotate clockwise)
```

#### Combat Actions (2 actions)
```cpp
'p' : Punch (melee attack in facing direction)
'x' : Attack with selected item (weapon/throwable)
```

#### Item Selection - Consumables (4 actions)
```cpp
'f' : Select energy_drink    (stamina boost)
'g' : Select first_aid_box   (HP restoration)
'h' : Select food_package    (HP + stamina)
'j' : Select zombie_vaccine  (major HP + effect removal)
```

#### Item Selection - Throwables (4 actions)
```cpp
'k' : Select gas             (area damage)
'l' : Select flash_bang      (stun effect)
';' : Select acid_bomb       (damage over time)
'\'' : Select stinger        (high single-target damage)
```

#### Item Selection - Weapons (8 actions)
```cpp
'c' : Select push_dagger
'v' : Select wing_tactic
'b' : Select F_898
'n' : Select lochabreask
'm' : Select AK_47
',' : Select M416
'.' : Select MOSSBERG
'/' : Select AWM
```

#### Item Usage & Tactical (3 actions)
```cpp
'u' : Use/consume selected item
'[' : Place block (defensive wall)
']' : Place portal (teleportation point)
```

### Controls Summary

```
Movement:  W/A/S/D  or  8/4/2/6 (NumPad)
Turn:      ` (left), 1 (right)
Attack:    X (use selected item), P (punch)
Block:     [ (place block)
Portal:    ] (place portal)
Items:     F/G/H/J (consumables)
           K/L/;/' (throwables)
           C/V/B/N/M/,/.// (weapons)
Use:       U (consume selected item)
UI:        0 (help), - (backpack), F/O (fullscreen)
Agent:     3 (toggle manual/auto mode)
           Space (pause rendering in auto mode)
```

---

## 🤖 Creating Custom AI Agents

### File Structure

Your custom agent consists of two files:

```
StrikeForce-client/bots/
└── bot-X/              # Your bot folder
    ├── Agent.hpp       # AI implementation
    └── Custom.hpp      # Game integration
```

### Step 1: Create Your Bot Folder

```bash
cd StrikeForce-client/bots
cp -r bot-0 bot-my-agent  # Start from template
```

### Step 2: Implement `Agent.hpp`

**Minimum Required Interface:**

```cpp
class Agent {
public:
    // Constructor - initialize your AI
    Agent(bool training = true) { }
    
    // Destructor - save models, cleanup
    ~Agent() { }
    
    // Main decision function
    // obs: game state observation (vector<float>)
    // Returns: action index (0-8 for default 9-action space)
    int predict(const std::vector<float>& obs) {
        // Your AI logic here
        return action_index;
    }
    
    // Update after action
    // action: what was done
    // imitate: was it manual (human) action?
    void update(int action, bool imitate) {
        // Learn from this step
    }
    
    // Check if currently training
    bool in_training() {
        return is_training;
    }
};
```

**Default Action Mapping (9 actions):**
```cpp
// actions: "+xp`1awsd"
0: '+' Do nothing
1: 'x' Attack with selected weapon
2: 'p' Punch
3: '`' Turn left
4: '1' Turn right
5: 'a' Move left
6: 'w' Move up
7: 's' Move down
8: 'd' Move right
```

### Step 3: Implement `Custom.hpp`

**Required Functions:**

```cpp
namespace Environment::Field {
    // Initialize agent before game
    void gameplay::prepare(Environment::Character::Human& player) {
        player.agent = new Agent(true);  // true = training mode
        player.set_agent_active();
    }
    
    // Get action from agent
    char gameplay::bot(Environment::Character::Human& player) const {
        if (!player.get_active_agent())
            return '+';  // Do nothing
        
        // 1. Extract game state
        std::vector<float> obs = extract_observations(player);
        
        // 2. Get agent's decision
        int action = player.agent->predict(obs);
        
        // 3. Return corresponding character
        return action_chars[action];  // "+xp`1awsd"
    }
    
    // Custom rendering (optional)
    void gameplay::view() const {
        // Additional visual feedback
    }
}
```

### Step 4: Connect Your Bot

Edit `selected_agent.hpp`:

```cpp
#include "./bots/bot-my-agent/Agent.hpp"
```

Edit `selected_custom.hpp`:

```cpp
#include "bots/bot-my-agent/Custom.hpp"
```

### Step 5: Compile & Test

```bash
g++ -std=c++17 main.cpp -o StrikeForce \
    -lsfml-graphics -lsfml-window -lsfml-system \
    -ltorch -ltorch_cpu -lc10

./StrikeForce
```

---

## 👁️ Observation Space Design

### bot-1's Observation Design

Located in `bots/bot-1/Custom.hpp`, the `describe()` function shows one way to encode observations:

```cpp
std::vector<float> describe(const node &cell, 
                           const Environment::Character::Human &player) {
    std::vector<float> res;
    
    // 7 features: Object type indicators
    res.push_back(cell.s[0] || cell.s[1]);  // Character or bullet
    res.push_back(cell.s[2]);               // Bullet only
    res.push_back(cell.s[3]);               // Wall
    res.push_back(cell.s[4]);               // Chest
    res.push_back(cell.s[5] || cell.s[6]);  // Portal
    res.push_back(cell.s[7]);               // Temporary object
    res.push_back(cell.s[10]);              // Destructible
    
    // 8 features: Character information
    std::vector<float> sit = {0, 0, 0, 0};
    if (cell.s[0]) {  // If character present
        int t = cell.human->get_team();
        if (!t) sit[2] = 1;                      // NPC
        else if (t == player.get_team()) sit[0] = 1;  // Ally
        else sit[1] = 1;                         // Enemy
    }
    if (cell.s[1]) sit[3] = 1;  // Zombie
    
    for (int i = 0; i < 4; ++i)
        res.push_back(sit[i]);
    
    // Character stats (normalized)
    if (cell.s[0]) {
        res.push_back(cell.human->get_kills());
        res.push_back(cell.human->backpack.get_blocks());
        res.push_back(cell.human->backpack.get_portals());
        res.push_back(cell.human->backpack.get_portal_ind() != -1);
    } else {
        for (int i = 0; i < 4; ++i) res.push_back(0);
    }
    
    // ... continues for 32 total channels ...
    
    return res;  // Returns feature vector for ONE cell
}
```

### Complete Observation Pipeline

The `obs` vector in `predict()` contains a **39×39 grid** around the player, compressed to **13×13** with **32 feature channels**:

**Feature Channels (32 total):**

```cpp
// Channels 0-6: Object types
[0]  = Character or bullet present
[1]  = Bullet only
[2]  = Wall
[3]  = Chest (item pickup)
[4]  = Portal (in/out)
[5]  = Temporary object
[6]  = Empty

// Channels 7-10: Character info
[7-10] = Team (ally/enemy/NPC/zombie)

// Channels 11-14: Character stats
[11] = Kill count
[12] = Blocks available
[13] = Portals available
[14] = Portal active

// Channels 15-18: Destructibility & HP
[15] = Can't pass (human)
[16] = Can't pass (bullet)
[17] = Can be destroyed
[18] = Health points

// Channels 19-26: Attack properties
[19] = Is bullet
[20-23] = Attack direction (up/right/down/left)
[24] = Attack damage
[25] = Attack effect
[26] = Stamina

// Channels 27-29: Item properties
[27] = Stamina boost
[28] = Effect boost
[29] = HP boost

// Channels 30-31: Status effects
[30] = Total damage dealt
[31] = Total effect dealt
```

---

## 🎓 Training Your Agent

### Training Macros

Control training behavior in `macros.hpp`:

```cpp
// Enable crowdsourced learning (uses API server)
#define CROWDSOURCED_TRAINING

// Freeze parts of the network
#define FREEZE_REWARDNET_BLOCK
#define FREEZE_AGENT_BLOCK

// Transfer learning from reward network
#define TL_IMPORT_REWARDNET
#define FREEZE_TL_BLOCK
```

### Example: PPO Agent (bot-1)

The included `bot-1` demonstrates a complete PPO implementation with:

- **RewardNet**: Learns to score actions (human vs agent)
- **AgentModel**: Policy + value network with GRU memory
- **Transfer Learning**: Shares CNN/GRU layers between networks
- **Crowdsourced Training**: Syncs via API server

```cpp
// Key components
AgentModel model;           // Policy network
RewardNet reward_net;       // Reward learning
std::vector<float> rewards; // Episode rewards
std::vector<int> actions;   // Action history
```

---

## 🌐 Server & Online Play

### Hosting a Match Server

```bash
cd StrikeForce-server
g++ -std=c++17 server.cpp -o MatchServer
./MatchServer
```

**Configuration:**
```
Is it global or local? G/local
Listening on port: 8080
Choose password: mypassword123
Number of players (n): 4
Number of teams (m): 2
Team assignments: 1 1 2 2
```

### Joining a Match

1. Select **"Join Battle Royale"** from menu
2. Enter server IP (displayed by host)
3. Enter port (8080)
4. Enter password
5. Wait for all players to connect

### Match Logging

When compiled with `-DLOGGING`, server saves:
```
<timestamp>_<serial>/
├── match_log.txt        # Game events
├── actions-0.txt        # Player 0 actions
├── actions-1.txt        # Player 1 actions
└── ...
```

---

## 💾 API Server for Version Control

### What is the API Server?

The API server (`server.py`) provides:

- **Backup Management**: Store AI model checkpoints
- **Crowdsourced Training**: Share improvements across users
- **Version Control**: Track model evolution with parent-child lineage

### Setup

```bash
cd StrikeForce-server
pip install flask pycryptodome
python server.py
```

Server runs on `http://0.0.0.0:8080` by default.

### API Endpoints

#### Admin Endpoints (Require API Key)

```bash
# Add a new bot
curl -X POST "http://SERVER_URL/StrikeForce/admin/add_bot?admin_key=YOUR_KEY&bot=my-bot"

# Upload backup
curl -X POST -F "file=@backup.zip" \
  "http://SERVER_URL/StrikeForce/admin/add_backup?admin_key=YOUR_KEY&bot=my-bot"

# Delete backup
curl -X POST \
  "http://SERVER_URL/StrikeForce/admin/delete_backup?admin_key=YOUR_KEY&bot=my-bot&serial=abc123"

# Delete a bot (and all backups)
curl -X POST \
  "http://SERVER_URL/StrikeForce/admin/delete_bot?admin_key=YOUR_KEY&bot=my-bot"
```

**Admin Key Hash** (in `server.py`):
```python
admin_key_hash = "b18b078c272d0ac43301ec84cea2f61b0c1fb1b961de7d6aa5ced573cb9132aa"
```

#### Client Endpoints

```bash
# Request latest backup
curl -o backup.zip "http://SERVER_URL/StrikeForce/api/request_backup?bot=my-bot"

# Submit trained backup
curl -X POST -F "file=@backup.zip" \
  "http://SERVER_URL/StrikeForce/api/return_backup"
```

### Client Integration

With `CROWDSOURCED_TRAINING` defined:

```cpp
// At agent startup
request_and_extract_backup("bots/bot-1/backup", "bot-1");

// At agent shutdown (after training)
zip_and_return_backup("bots/bot-1/backup");
```

**Backup Structure:**
```
bots/bot-1/backup/
├── agent_backup/
│   ├── model.pt
│   ├── optimizer.pt
│   └── agent_log.log
├── reward_backup/
│   ├── model.pt
│   ├── optimizer.pt
│   └── reward_log.log
└── metadata.enc
```

---

## 🗺️ Custom Map Design

### Map File Format

Maps are stored in `StrikeForce-client/map/` as text files. Example from `floor1.txt`:

```
####################^ 2 ^ 2 #################
#.................#.....O....#....#.....#...
#.................#..........#....#.....#...
#.................#..........##..####..##...
```

### Map Symbols

| Symbol | Meaning |
|--------|---------|
| `#` | Solid wall (impassable) |
| `.` | Empty floor (walkable) |
| `O` | Portal entrance |
| `^ N` | Portal exit (N = portal ID) |
| `v N` | Portal entrance with ID |

### Changing Map Dimensions

If you modify map size, update `gameplay.hpp`:

```cpp
// In gameplay.hpp
namespace Environment::Field {
    int constexpr F = 3;    // Number of floors
    int constexpr N = 30;   // Map height
    int constexpr M = 100;  // Map width
    // ...
}
```

### Creating a New Map

1. **Create file**: `StrikeForce-client/map/floor4.txt`
2. **Design layout**: Use symbols above, ensure dimensions match
3. **Update F constant**: Change `F = 3` to `F = 4` in `gameplay.hpp`
4. **Recompile**: `g++ -std=c++17 main.cpp -o StrikeForce ...`

---

## 📁 Project Structure

```
StrikeForce/
├── StrikeForce-client/
│   ├── main.cpp                    # Entry point
│   ├── basic.hpp                   # Core utilities
│   ├── GraphicPrinter.hpp          # SFML rendering
│   ├── gameplay.hpp                # Game loop
│   ├── Character.hpp               # Player/NPC logic
│   ├── Item.hpp                    # Weapons/items
│   ├── enter.hpp                   # Auth system
│   ├── menu.hpp                    # UI menus
│   ├── random.hpp                  # Deterministic RNG
│   ├── selected_agent.hpp          # ← YOUR AGENT HERE
│   ├── selected_custom.hpp         # ← YOUR INTEGRATION HERE
│   ├── macros.hpp                  # Build flags
│   ├── bots/
│   │   ├── bot-0/                  # Dummy agent template
│   │   │   ├── Agent.hpp
│   │   │   └── Custom.hpp
│   │   └── bot-1/                  # PPO example
│   │       ├── Agent.hpp
│   │       ├── Custom.hpp
│   │       └── RewardNet.hpp
│   ├── Items/                      # Item stats
│   ├── map/                        # Level designs
│   ├── character/                  # Character configs
│   └── accounts/                   # User data
│
├── StrikeForce-server/
│   └── server.cpp                  # Match server
│
└── server.py                       # API/backup server
```

---

## 📊 Research Context

### StrikeForce in Academic Research

This environment was developed as part of research in **resource-efficient imitation learning**. Key findings:

- **84.2% win rate** vs expert humans with only 16 participants
- **47.3/50 human-likeness score** using GAIL-RT protocol
- **<100MB RAM** idle, **<3.2GB RAM** during training (CPU-only)

### Paper: "Crowdsourced Training of Human-Like Agents via Adaptive GAIL"

The accompanying paper introduces:
1. **GAIL-RT Protocol**: Representation transfer for stable adversarial imitation
2. **Resource-Optimal Design**: End-to-end efficiency in data, compute, and human effort
3. **Crowdsourcing Pipeline**: Distributed training without centralized datasets

**Citation:**
```bibtex
@article{fouladi2025strikeforce,
  title={StrikeForce: Crowdsourced Training of Human-Like Agents via Adaptive GAIL},
  author={Kasra Fouladi and Hamta Rahmani},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

---

## 🔧 Technical Details

### Dependencies

- **SFML 2.5+**: Graphics and window management
- **LibTorch 1.x**: PyTorch C++ API for neural networks
- **C++17**: Modern C++ features
- **Python 3.8+**: For API server (optional)

### Performance Notes

- **Frame Rate**: 100 FPS (configurable in `GraphicPrinter.hpp`)
- **Action Delay**: 40ms between decisions (see `lim` in `Custom.hpp`)
- **Map Size**: 30×100×3 (floors × width × height)

### Memory Management

⚠️ **Critical**: Agents are created with `new` and must be deleted:

```cpp
// In gameplay::play()
hum[ind].deleteAgent();  // Calls delete on agent pointer
hum[ind].reset();        // Resets player state
```

### Deterministic Random

For reproducible games:

```cpp
Environment::Random::_srand(timestamp, serial_number);
int random_value = Environment::Random::_rand();
```

---

## 🎉 Why StrikeForce?

### For AI Researchers
- **Rich observation space** (32 channels × 13×13 grid)
- **Complex action space** with strategic depth
- **Partial observability** and fog of war
- **Multi-agent competition** out of the box
- **Reward shaping** via human demonstrations

### For Students
- **Learn by doing**: See RL concepts in action
- **Gradual difficulty**: Start with Solo, progress to multiplayer
- **Visual feedback**: Watch your agent learn in real-time
- **Compare approaches**: bot-0 vs bot-1 vs your own

### For Gamers
- **Actually fun to play!** Solid shooter mechanics
- **Progressive unlocks**: Earn weapons and upgrades
- **Leaderboards**: Compete in Solo/Timer/Squad modes
- **Online battles**: Test your skills (or your AI's)

---

## 📝 License

MIT License - see LICENSE file for details.

**Created by**: Kasra Fouladi, Hamta Rahmani, bistoyek21 R.I.C.

---

## 🤝 Contributing

We welcome contributions!

1. **New agents**: Share your bot implementations
2. **Map designs**: Create new levels in `map/`
3. **Game modes**: Extend `gameplay.hpp`
4. **Optimizations**: Improve rendering/networking
5. **Research extensions**: Add new observation schemes or training protocols

### Contribution Process

1. Fork the repository
2. Create a feature branch: `git checkout -b my-new-feature`
3. Implement and test your changes
4. Submit a pull request with clear description

---

## 📞 Support

**Found a bug?** Open an issue on the repository.

**Need help with your agent?** Check `bots/bot-1/` for a complete example.

**Research collaboration?** Email: k4sr405@gmail.com or hamtar693@gmail.com

**Want to play online?** Join our Discord community (link coming soon).

---

**Good luck, Commander! May your agents dominate the battlefield! 🎯🤖**