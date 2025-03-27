import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from collections import deque
import random

class HyperNet(nn.Module):
    def __init__(self, n, m, k):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.LayerNorm([32, n-2, n-2]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.AdaptiveMaxPool2d(4),
            nn.Flatten()
        )
        
        self.self_attn = nn.MultiheadAttention(embed_dim=k, num_heads=4)
        self.enemy_attn = nn.MultiheadAttention(embed_dim=k, num_heads=2)
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 2*k, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, m)
        )

    def forward(self, grid, self_vec, enemy_vec):
        grid = grid.unsqueeze(1)
        grid_feat = self.grid_conv(grid)
        
        self_vec = self_vec.unsqueeze(1)
        self_attn, _ = self.self_attn(self_vec, self_vec, self_vec)
        self_feat = self_attn.mean(dim=1)
        
        enemy_vec = enemy_vec.unsqueeze(1)
        enemy_attn, _ = self.enemy_attn(enemy_vec, enemy_vec, enemy_vec)
        enemy_feat = enemy_attn.mean(dim=1)
        
        combined = torch.cat([grid_feat, self_feat, enemy_feat], dim=1)
        fused = self.fusion(combined)
        
        value = self.value_stream(fused)
        advantage = self.advantage_stream(fused)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class StrategicAgent:
    def __init__(self, n, m, k, device='cuda'):
        self.n = n
        self.m = m
        self.k = k
        self.device = device
        
        self.model = HyperNet(n, m, k).to(device)
        self.target_model = HyperNet(n, m, k).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.RAdam(self.model.parameters(), lr=5e-4)
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.tau = 0.005
        
        self.priorities = deque(maxlen=100000)
        self.alpha = 0.6
        self.beta = 0.4
        
        self.sampling_threshold = 0.7

    def _normalize(self, grid, self_vec, enemy_vec):
        grid = torch.FloatTensor(grid/255.0).to(self.device)
        self_vec = torch.FloatTensor(self_vec).to(self.device)
        enemy_vec = torch.FloatTensor(enemy_vec).to(self.device)
        return grid, self_vec, enemy_vec

    def action(self, grid, self_vec, enemy_vec):
        if random.random() < self.epsilon:
            return random.randint(0, self.m-1)
        
        with torch.no_grad():
            grid, self_vec, enemy_vec = self._normalize(grid, self_vec, enemy_vec)
            q_values = self.model(
                grid.unsqueeze(0), 
                self_vec.unsqueeze(0), 
                enemy_vec.unsqueeze(0)
            )
        return q_values.argmax().item()

    def store(self, grid, self_vec, enemy_vec, action, reward, next_grid, next_self_vec, next_enemy_vec, done):
        self.memory.append((
            grid.copy(), 
            self_vec.copy(), 
            enemy_vec.copy(), 
            next_grid.copy(), 
            next_self_vec.copy(), 
            next_enemy_vec.copy(), 
            action, 
            reward, 
            done
        ))
        self.priorities.append(max(self.priorities, default=1.0))


    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[i] for i in indices]
        
        states = [self._normalize(*s[:3]) for s in samples]
        next_states = [self._normalize(*s[3:6]) for s in samples]
        actions = torch.LongTensor([s[6] for s in samples]).to(self.device)
        rewards = torch.FloatTensor([s[7] for s in samples]).to(self.device)
        dones = torch.FloatTensor([s[8] for s in samples]).to(self.device)
        
        with torch.no_grad():
            current_q = self.model(*zip(*states)).gather(1, actions.unsqueeze(1))
            next_actions = self.model(*zip(*next_states)).argmax(dim=1)
            next_q = self.target_model(*zip(*next_states)).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()
            td_errors = (target_q - current_q.squeeze()).abs().cpu().numpy()
            
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha
            
        weights = (len(self.memory) * probs[indices]) ** -self.beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q = self.model(*zip(*states)).gather(1, actions.unsqueeze(1))
        loss = (weights * F.mse_loss(current_q.squeeze(), target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau*param.data + (1.0-self.tau)*target_param.data)
        
        self.epsilon = max(0.01, self.epsilon * 0.992 - 0.005 * (1 - self.sampling_threshold))


if __name__ == '__main__':
    agent = StrategicAgent(n=11, m=20, k=10)
    while True:
        

        action = agent.action(
            grid=current_grid,
            self_vec=player_features,
            enemy_vec=closest_enemy_features
        )

        agent.store(
            grid=current_grid,
            self_vec=player_features,
            enemy_vec=enemy_features,
            action=selected_action,
            reward=received_reward,
            next_grid=new_grid,
            next_self_vec=new_player_features,
            next_enemy_vec=new_enemy_features,
            done=is_done
        )

        agent.update()