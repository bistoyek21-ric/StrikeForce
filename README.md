# StrikeForce: Comprehensive Environment Documentation & Customization Guide

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Paper](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/)

> **A resource-efficient, crowdsourced framework for training human-like AI agents in tactical environments**

---

## 📋 Table of Contents

1. [Introduction](#-introduction)
2. [Environment Overview](#-environment-overview)
3. [Action Space Reference](#-action-space-reference)
4. [Creating Custom Agents](#-creating-custom-agents)
5. [Observation Space Design](#-observation-space-design)
6. [Custom Map Design](#-custom-map-design)
7. [API Server Usage](#-api-server-usage)
8. [Complete Workflow Example](#-complete-workflow-example)
9. [Research Context](#-research-context)

---

## 🌟 Introduction

**StrikeForce** is a tactical 2D environment designed for research in imitation learning, human-AI interaction, and crowdsourced training. This guide focuses on **how to use and customize the environment** for your own research needs.

### Key Features

- 🎯 **Fully Customizable**: Modify observations, actions, maps, and agent architectures
- 🧠 **Research-Ready**: Built for imitation learning and human-in-the-loop experiments
- 💾 **Resource-Efficient**: Runs on consumer CPUs with <100MB RAM (idle), <3.2GB (training)
- 🌐 **Crowdsourcing-Ready**: Built-in API server for distributed training
- 📊 **Reproducible**: Deterministic simulation with seed control

### What This Guide Covers

This documentation explains:
- How to **design custom observation spaces** for your agent
- How to **implement custom agent architectures** and integrate them
- How to **create custom maps** and modify environment parameters
- How to **use the API server** for version control and crowdsourced training
- The **complete action space** available in the environment

---

## 🎮 Environment Overview

### Design Philosophy

StrikeForce uses a **2D top-down tile-based** design to:
- **Isolate strategic reasoning** from perceptual complexity
- Enable **efficient execution** on consumer hardware
- Facilitate **reproducible experiments** with deterministic simulation
- Support **crowdsourced data collection** at scale

### Core Components

```
StrikeForce-client/
├── gameplay.hpp          # Main game loop and environment state
├── Character.hpp         # Player/NPC logic and inventory
├── Item.hpp             # Weapons, consumables, and items
├── bots/
│   ├── bot-0/           # Minimal template
│   └── bot-1/           # PPO + GAIL-RT example
├── map/                 # Level designs (floor1.txt, floor2.txt, etc.)
└── selected_custom.hpp  # ← Connect your custom integration here
└── selected_agent.hpp   # ← Connect your agent architecture here
```

---

## 🎯 Action Space Reference

### Default Action Space (30 actions)

The environment supports up to **30 discrete actions**. Here's the complete mapping:

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

#### Item Selection - Cold Weapons (4 actions)
```cpp
'c' : Select push_dagger
'v' : Select wing_tactic
'b' : Select F_898
'n' : Select lochabreask
```

#### Item Selection - Firearms (4 actions)
```cpp
'm' : Select AK_47
',' : Select M416
'.' : Select MOSSBERG
'/' : Select AWM
```

#### Item Usage (1 action)
```cpp
'u' : Use/consume selected item (applies consumable effects)
```

#### Tactical Actions (2 actions)
```cpp
'[' : Place block (defensive wall in facing direction)
']' : Place portal (teleportation point)
```

### Custom Action Spaces

You can define a **subset** of these actions for your agent. For example, `bot-1` uses only 9 actions:

```cpp
// In bots/bot-1/Custom.hpp
std::string action = "+xp`1awsd";  // 9-action subset
// 0: '+' (no-op)
// 1: 'x' (attack)
// 2: 'p' (punch)
// 3: '`' (turn left)
// 4: '1' (turn right)
// 5: 'a' (move left)
// 6: 'w' (move up)
// 7: 's' (move down)
// 8: 'd' (move right)
```

**How to customize:**
1. Choose which actions your agent needs
2. Map them to indices 0, 1, 2, ...
3. Return the corresponding character in `bot()`

---

## 🤖 Creating Custom Agents

### File Structure

```
StrikeForce-client/bots/
└── my-agent/
    ├── Agent.hpp       # Your AI implementation
    └── Custom.hpp      # Environment integration
```

### Step 1: Implement Agent.hpp

This file contains your **agent architecture**. Minimum required interface:

```cpp
#include "basic.hpp"

class Agent {
public:
    Agent(bool training = true) {
        // Initialize your model
        // - Load neural networks
        // - Set up optimizers
        // - Initialize memory buffers
    }
    
    ~Agent() {
        // Cleanup and save checkpoints
    }
    
    // Main decision function
    // obs: game state observation (std::vector<float>)
    // Returns: action index (0 to num_actions-1)
    int predict(const std::vector<float>& obs) {
        // Your inference logic
        return action_index;
    }
    
    // Called after action execution
    // action: what was executed
    // imitate: true if human action, false if agent action
    void update(int action, bool imitate) {
        // Store experience
        // Update models (if training)
    }
    
    // Check if training is in progress
    bool in_training() {
        return is_training;
    }
};
```

### Step 2: Implement Custom.hpp

This file defines **how the environment interacts** with your agent. It's called **"Custom"** because it's completely customizable based on your needs.

```cpp
#include "../../gameplay.hpp"

namespace Environment::Field {
    
    // Initialize agent before game starts
    void gameplay::prepare(Environment::Character::Human& player) {
        // Create your agent instance
        player.agent = new Agent(true);  // true = training mode
        player.set_agent_active();
        
        // Optional: Initialize action mapping
        action = "+xp`1awsd";  // Your chosen action subset
    }
    
    // Get action from agent each frame
    char gameplay::bot(Environment::Character::Human& player) const {
        if (!player.get_active_agent())
            return '+';  // Fallback: do nothing
        
        // 1. Extract observation (YOUR CUSTOM LOGIC)
        std::vector<float> obs = extract_observations(player);
        
        // 2. Get agent decision
        int action_idx = player.agent->predict(obs);
        
        // 3. Map to game action
        return action[action_idx];
    }
    
    // Optional: Custom rendering/debugging
    void gameplay::view() const {
        // Additional visualizations
    }
}
```

### Step 3: Connect Your Agent

Edit `selected_agent.hpp`:
```cpp
#include "./bots/my-agent/Agent.hpp"
```

Edit `selected_custom.hpp`:
```cpp
#include "bots/my-agent/Custom.hpp"
```

---

## 👁️ Observation Space Design

### Understanding Custom.hpp's Role

The **`Custom.hpp`** file is where you define **what information your agent receives**. The environment provides full game state; you decide how to encode it.

### Example: bot-1's Observation Design

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

### bot-1's Complete Observation Pipeline

```cpp
char gameplay::bot(Environment::Character::Human& player) const {
    std::vector<int> v = player.get_cor();  // Player position
    
    // Build 39x39 grid with 32 channels per cell
    std::vector<std::vector<float>> ch(32);
    for (int i = v[1] - 19; i <= v[1] + 19; ++i) {
        for (int j = v[2] - 19; j <= v[2] + 19; ++j) {
            std::vector<float> vec;
            
            if (i < 0 || j < 0 || N <= i || M <= j)
                vec = describe(nd, player);  // Out of bounds
            else
                vec = describe(themap[v[0]][i][j], player);
            
            // Append each channel
            for (int k = 0; k < vec.size(); ++k)
                ch[k].push_back(vec[k]);
        }
    }
    
    // Downsample to 13x13 by pooling 3x3 blocks
    std::vector<float> obs;
    for (int k = 0; k < 32; ++k) {
        for (int i = 0; i < 39; i += 3) {
            for (int j = 0; j < 39; j += 3) {
                for (int i1 = 0; i1 < 3; ++i1) {
                    for (int j1 = 0; j1 < 3; ++j1) {
                        if (std::max(i + i1, j + j1) < 39) {
                            // Power transformation for normalization
                            obs.push_back(std::pow(
                                std::abs(ch[k][(i+i1)*39 + (j+j1)]) / 10, 
                                0.2
                            ));
                        }
                    }
                }
            }
        }
    }
    
    return action[player.agent->predict(obs)];
}
```

### Key Design Principles for Custom Observations

1. **Access Full Game State**: The `themap[floor][x][y]` structure gives you complete information
2. **Encode Semantically**: Extract meaningful features (health, team, item type) rather than raw pixels
3. **Normalize Values**: Scale features to [0, 1] or [-1, 1] for stable training
4. **Consider Partial Observability**: bot-1 uses 39×39 local grid (matching human FOV)
5. **Balance Richness vs Efficiency**: More features = more information, but slower inference

### Your Custom Observation

```cpp
// In bots/my-agent/Custom.hpp
char gameplay::bot(Environment::Character::Human& player) const {
    // Example: Simple distance-based observation
    std::vector<float> obs;
    std::vector<int> pos = player.get_cor();
    
    // Find nearest enemy
    float min_dist = 1000.0f;
    for (int i = 0; i < H; ++i) {
        if (mh[i] && hum[i].get_team() != player.get_team()) {
            std::vector<int> enemy_pos = hum[i].get_cor();
            float dist = std::abs(pos[1] - enemy_pos[1]) + 
                        std::abs(pos[2] - enemy_pos[2]);
            min_dist = std::min(min_dist, dist);
        }
    }
    obs.push_back(min_dist / 100.0);  // Normalized
    
    // Add player stats
    obs.push_back(player.get_Hp() / 1000.0);
    obs.push_back(player.get_stamina() / 1000.0);
    
    // ... add whatever features you need ...
    
    return action[player.agent->predict(obs)];
}
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

### Map Design Tips

- **Balance openness**: Too open = no cover, too closed = tedious navigation
- **Portal placement**: Use for vertical movement and strategic shortcuts
- **Spawn zones**: Place `O` symbols in corners for balanced starts
- **Testing**: Play manually first to verify layout quality

---
## 🌐 Server & Online Play
Hosting a Match Server
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
## 💾 StrikeForceAPI for Checkpoint Version Control

The API server (`server.py`) provides:
- **Checkpoint storage**: Save/load model versions
- **Crowdsourced training**: Aggregate updates from multiple users
- **Version control**: Track model evolution with parent-child lineage

### Starting the Server

```bash
clone https://github.com/bistoyek21-ric/StrikeForceAPI.git
cd StrikeForce-server
pip install flask pycryptodome
python server.py
```

Server runs on `http://0.0.0.0:8080` by default.

### API Endpoints

#### Admin Endpoints (Require API Key)

**1. Add a new bot**
```bash
curl -X POST "http://SERVER_URL/StrikeForce/admin/add_bot?admin_key=YOUR_KEY&bot=my-bot"
```

**2. Upload a backup**
```bash
curl -X POST -F "file=@backup.zip" \
  "http://SERVER_URL/StrikeForce/admin/add_backup?admin_key=YOUR_KEY&bot=my-bot"
```

**3. Delete a backup**
```bash
curl -X POST \
  "http://SERVER_URL/StrikeForce/admin/delete_backup?admin_key=YOUR_KEY&bot=my-bot&serial=abc123"
```

**4. Delete a bot (and all backups)**
```bash
curl -X POST \
  "http://SERVER_URL/StrikeForce/admin/delete_bot?admin_key=YOUR_KEY&bot=my-bot"
```

**5. Get encryption keys**
```bash
curl "http://SERVER_URL/StrikeForce/admin/get_crypto?admin_key=YOUR_KEY"
```

#### Client Endpoints (No Auth Required)

**1. Request latest backup**
```bash
curl -o backup.zip \
  "http://SERVER_URL/StrikeForce/api/request_backup?bot=my-bot"
```

**2. Submit trained backup**
```bash
curl -X POST -F "file=@backup.zip" \
  "http://SERVER_URL/StrikeForce/api/return_backup"
```

### Understanding the API Key

The **admin_key** in `server.py` is a **SHA-256 hash**:

```python
admin_key_hash = "b18b078c272d0ac43301ec84cea2f61b0c1fb1b961de7d6aa5ced573cb9132aa"
```

**Purpose**: 
- Protects admin operations (bot creation/deletion)
- Prevents unauthorized model modifications
- Enables secure crowdsourcing setup

**To change it**:
1. Generate your key: `echo -n "my_secret_key" | sha256sum`
2. Replace the hash in `server.py`
3. Restart server

### Crowdsourced Training Flow

```
1. Player downloads checkpoint:
   curl -o backup.zip "http://SERVER/api/request_backup?bot=my-bot"

2. Player extracts and plays:
   7z x backup.zip -obots/my-bot/backup/

3. Agent trains locally during gameplay

4. Player uploads updated checkpoint:
   cd bots/my-bot/backup/
   7z a -tzip backup.zip ./*
   curl -X POST -F "file=@backup.zip" "http://SERVER/api/return_backup"

5. Server aggregates or creates branch

6. Next player gets improved version
```

### Backup Structure

```
bots/my-bot/backup/
├── agent_backup/
│   ├── model.pt          # PyTorch model
│   ├── optimizer.pt      # Optimizer state
│   └── agent_log.log     # Training logs
├── reward_backup/        # (if using discriminator)
│   ├── model.pt
│   ├── optimizer.pt
│   └── reward_log.log
└── metadata.enc          # Encrypted session token
```

### Security Features

- **AES-256 encryption** for session tokens
- **One-time credentials** embedded in metadata
- **Path traversal protection** during extraction
- **File size limits** (100MB default)
- **Parent-child lineage tracking** for version control

---

## 🔄 Complete Workflow Example

### Scenario: Training a Simple RL Agent

**Goal**: Create an agent that learns to survive by avoiding enemies

#### 1. Create Agent Architecture

```cpp
// bots/survival-agent/Agent.hpp
#include "basic.hpp"
#include <random>

class Agent {
private:
    std::mt19937 rng;
    std::vector<float> q_values;  // Simple Q-table approximation
    
public:
    Agent(bool training = true) {
        rng.seed(std::random_device{}());
        q_values.resize(9, 0.0f);  // 9 actions
    }
    
    int predict(const std::vector<float>& obs) {
        // Epsilon-greedy
        if (std::uniform_real_distribution<>(0,1)(rng) < 0.1) {
            return std::uniform_int_distribution<>(0,8)(rng);
        }
        
        // Greedy action
        return std::max_element(q_values.begin(), q_values.end()) 
               - q_values.begin();
    }
    
    void update(int action, bool imitate) {
        // Simple reward: +1 if human, -1 if agent died
        float reward = imitate ? 1.0f : -1.0f;
        
        // Q-learning update
        float alpha = 0.01f;
        q_values[action] += alpha * reward;
    }
    
    bool in_training() { return false; }
};
```

#### 2. Define Observation Extraction

```cpp
// bots/survival-agent/Custom.hpp
#include "../../gameplay.hpp"

namespace Environment::Field {
    void gameplay::prepare(Environment::Character::Human& player) {
        player.agent = new Agent(true);
        player.set_agent_active();
        action = "+xp`1awsd";  // 9 actions
    }
    
    char gameplay::bot(Environment::Character::Human& player) const {
        if (!player.get_active_agent())
            return '+';
        
        std::vector<float> obs;
        std::vector<int> pos = player.get_cor();
        
        // Feature 1: Distance to nearest enemy
        float min_enemy_dist = 1000.0f;
        for (int i = 0; i < H; ++i) {
            if (mh[i] && hum[i].get_team() != player.get_team()) {
                auto ep = hum[i].get_cor();
                float d = std::abs(pos[1]-ep[1]) + std::abs(pos[2]-ep[2]);
                min_enemy_dist = std::min(min_enemy_dist, d);
            }
        }
        obs.push_back(min_enemy_dist / 100.0);
        
        // Feature 2: Player health
        obs.push_back(player.get_Hp() / 1000.0);
        
        // Feature 3: Stamina
        obs.push_back(player.get_stamina() / 1000.0);
        
        int action_idx = player.agent->predict(obs);
        return action[action_idx];
    }
    
    void gameplay::view() const {}
}
```

#### 3. Connect and Compile

```cpp
// selected_agent.hpp
#include "./bots/survival-agent/Agent.hpp"

// selected_custom.hpp
#include "bots/survival-agent/Custom.hpp"
```

```bash
g++ -std=c++17 main.cpp -o StrikeForce \
    -lsfml-graphics -lsfml-window -lsfml-system
./StrikeForce
```

#### 4. Train Locally

```
1. Run StrikeForce
2. Login (username: 1, password: 1)
3. Select "Play" → "Solo" → "Level 1"
4. Choose "Use AI agent? → y"
5. Play alongside the agent
6. Agent learns from your actions
```

#### 5. Share via API (Optional)

```bash
# Upload your trained agent
cd bots/survival-agent/backup/
7z a -tzip backup.zip ./*
curl -X POST -F "file=@backup.zip" \
  "http://YOUR_SERVER/StrikeForce/api/return_backup"
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

**Full paper**: Available in `paper.tex` (see repository)

### Citation

If you use StrikeForce in your research:

```bibtex
@article{fouladi2025strikeforce,
  title={StrikeForce: Crowdsourced Training of Human-Like Agents via Adaptive GAIL},
  author={Kasra Fouladi and Hamta Rahmani},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions!

### Areas for Improvement

- **New agent architectures**: Implement transformers, world models, etc.
- **Environment extensions**: Add new game modes, items, or mechanics
- **Performance optimizations**: Improve rendering, networking, or training speed
- **Documentation**: Add tutorials, examples, or translations

### Contribution Process

1. Fork the repository
2. Create a feature branch: `git checkout -b my-new-feature`
3. Implement and test your changes
4. Submit a pull request with clear description

---

## 📞 Support

**Questions?**
- Open an issue on GitHub
- Email: k4sr405@gmail.com or hamtar693@gmail.com

**Want to Collaborate?**
- We're open to research partnerships
- Interested in deploying at scale? Contact us

---

## 📝 License

MIT License - See `LICENSE` file for details.

**Created by**: Kasra Fouladi & Hamta Rahmani  
**Organization**: bistoyek21 R.I.C.

---

**Ready to train your agent? Let's go! 🚀**