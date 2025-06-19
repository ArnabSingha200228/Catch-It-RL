from catchit import CatchItEnvRender
import numpy as np
np.bool8 = np.bool_
import gym
import torch
import torch.nn as nn
import random
import pygame

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
def select_action(state, epsilon):
    if random.random() < epsilon:  # Explore: random action
        return env.action_space.sample()
    else:  # Exploit: best action from policy_net
        with torch.no_grad():
            return policy_net(state).argmax().item()


env = CatchItEnvRender(render_mode='human', timer=True)
state_size = env.observation_space.shape[0]  
action_size = env.action_space.n    

policy_net = DQN(state_size, action_size)  # Main Q-network
policy_net = torch.load("policy_net.pth", weights_only=False)

def watch_trained_model(policy_net, env):
    # Create a new environment with human rendering
    state, _ = env.reset()
    state = torch.FloatTensor(state)  # Shape [4] (no batch dimension needed)
    total_reward = 0
    
    while True:
        # Get action from trained policy
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.argmax().item()
            # print(f"Q-values: {q_values}, Action: {action}")  # Debug print
            # action = select_action(state, 0.05)
        # Take the action and render
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()  # Render after taking action
        
        # Small delay to see the movement
        # pygame.time.delay(1)  # 100ms delay
        
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)
        
        state = next_state
        total_reward += reward
        
        if done:
            break

    print(f"Score : {env.score}")
    env.close()

# Run the visualization
watch_trained_model(policy_net, env)
