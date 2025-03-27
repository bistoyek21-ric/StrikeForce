import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

class PPONet(nn.Module):
    def __init__(self, n, m, k):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.InstanceNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3),
            nn.AdaptiveMaxPool2d(4),
            nn.Flatten()
        )
        
        self.self_attn = nn.MultiheadAttention(embed_dim=k, num_heads=4)
        self.enemy_attn = nn.MultiheadAttention(embed_dim=k, num_heads=2)
        
        self.policy = nn.Sequential(
            nn.Linear(64 + 2*k, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, m)
        )
        
        self.value = nn.Sequential(
            nn.Linear(64 + 2*k, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, grid, self_vec, enemy_vec):
        grid = grid.view(-1, 1, self.n, self.n)
        grid_feat = self.grid_encoder(grid)
        
        self_vec = self_vec.unsqueeze(1)
        self_attn, _ = self.self_attn(self_vec, self_vec, self_vec)
        self_feat = self_attn.mean(dim=1)
        
        enemy_vec = enemy_vec.unsqueeze(1)
        enemy_attn, _ = self.enemy_attn(enemy_vec, enemy_vec, enemy_vec)
        enemy_feat = enemy_attn.mean(dim=1)
        
        combined = torch.cat([grid_feat, self_feat, enemy_feat], dim=1)
        
        logits = self.policy(combined)
        value = self.value(combined)
        
        return logits, value

class PPOBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
    def store(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
    def compute_advantages(self, last_value):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        advantages = np.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * advantage
            advantages[t] = advantage
            
        returns = advantages + values[:-1]
        return advantages, returns
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

class PPOAgent:
    def __init__(self, n, m, k, device='cuda'):
        self.n = n
        self.m = m
        self.k = k
        self.device = device
        
        self.model = PPONet(n, m, k).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        self.buffer = PPOBuffer()
        
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.epochs = 4
        self.batch_size = 64
        
    def _normalize(self, grid, self_vec, enemy_vec):
        grid = torch.FloatTensor(grid / 255.0).to(self.device)
        self_vec = torch.FloatTensor(self_vec).to(self.device)
        enemy_vec = torch.FloatTensor(enemy_vec).to(self.device)
        return grid, self_vec, enemy_vec
        
    def act(self, grid, self_vec, enemy_vec):
        with torch.no_grad():
            grid, self_vec, enemy_vec = self._normalize(grid, self_vec, enemy_vec)
            logits, value = self.model(
                grid.unsqueeze(0),
                self_vec.unsqueeze(0),
                enemy_vec.unsqueeze(0)
            )
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), value.item(), log_prob.item()
    
    def update(self):
        if len(self.buffer.states) < self.batch_size:
            return
        
        states = [self._normalize(*s) for s in self.buffer.states]
        grid = torch.stack([s[0] for s in states])
        self_vec = torch.stack([s[1] for s in states])
        enemy_vec = torch.stack([s[2] for s in states])
        
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        with torch.no_grad():
            _, last_value = self.model(grid[-1], self_vec[-1], enemy_vec[-1])
        
        advantages, returns = self.buffer.compute_advantages(last_value.item())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            indices = np.arange(len(self.buffer.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start+self.batch_size]
                
                b_grid = grid[batch_indices]
                b_self = self_vec[batch_indices]
                b_enemy = enemy_vec[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]
                
                logits, values = self.model(b_grid, b_self, b_enemy)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), b_returns)
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        
        self.buffer.clear()

import time

if __name__ == '__main__':
    
    actions = "`1pxawsd[]+"

    agent = PPOAgent(n=11, m=len(actions), k=7)

    #initialize: self_vec, enemy_vec, grid, reward, isend

    while True:
        with open('./checkout.txt') as f:
            if f.read().split()[0] == '1':
                break
        time.timer(0.001)

    #get state from and reward form ./pov.txt file
    #and find out is it done or not
    #pov.txt format is like below:
    #P/E (to find out is it done or not P:playing, E:end)
    # a line containing k integers (self_vec elements)
    # a line containing k integers (enemy_vec elements)
    # n lines containging n integers (grid elements)

    while True:

        action, value, log_prob = agent.act(grid, self_vec, enemy_vec)

        with open('./check.txt', 'r') as f:
        f.write('0 +')

        while True:
            with open('./checkout.txt', 'r') as f:
                if f.read().split()[0] == '1':
                    break
            time.timer(0.001)


        pov = open('./pov.txt', 'w').read.split()

        #get state from and reward form ./pov.txt file
        #and find out is it done or not
        #pov.txt format is like below:
        #P/E (to find out is it done or not P:playing, E:end)
        # a line containing k integers (self_vec elements)
        # a line containing k integers (enemy_vec elements)
        # n lines containging n integers (grid elements)

        with open('./check.txt') as f:
            f.write('0 ' + actions[action])

        agent.buffer.store(
            state=(grid, self_vec, enemy_vec),
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob
        )
        agent.update()