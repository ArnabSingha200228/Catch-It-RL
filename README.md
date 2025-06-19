# Catch-It-RL: Reinforcement Learning Agent Using DQN to play a custom 2D Catch-It game

## 📌 Project Overview

This project explores how a reinforcement learning agent can learn to play the Catch-It 2D game (custom made 2D game). The agent is trained using **Deep Q-Networks (DQN)**.

---
## 🎯 The Catch-It game Environment
The **Catch-It** game is a custom-designed 2D game environment where a rectangular catcher object (agent) moves around the screen to Catch-Items that spawn randomly in a field within a specified amount of time. The environment provides a discrete action space allowing the catcher to move left, right, up or down, and a continuous observation space that captures both absolute and relative positions of the catcher and the items. The game challenges the agent to learn spatial awareness and timing to catch the items efficiently, making it a simple yet effective testbed for reinforcement learning algorithms like DQN.



## 🎯 Objective

- Build an environment for simulating the Catch-It game.
- Train an agent to move the catcher left/right/up/down to reach the item and Catch-It.
- Maximize the agent’s performance using Q-learning with neural networks as function approximators (DQN).

---

## 🧩 Environment Details

**Name**: `Catch-ItEnv`  
**Observation Space**:  
- `rect_x`, `rect_y`: Catcher position  
- `item_x`, `item_y`: Falling item position  
- `dx`, `dy`: Relative difference (catcher to item)

**Action Space**:  
- Discrete(4): `Left`, `Right`, `Up`, `Down`

**Termination Condition of the game**:  
- Timeout of timer

**Termination Condition for training**:  
- Catching of item
- Reaching of max_steps

---

## 🧠 Agent: Deep Q-Network (DQN)

- **Model**: MLP with two hidden layers (ReLU activations)
- **Exploration Strategy**: ε-greedy with decay
- **Learning Technique**: Experience Replay + Target Network

**Key Components**:
- `policy_net`: Learns Q-values
- `target_net`: Stabilizes learning
- `replay_buffer`: Stores transitions `(state, action, reward, next_state, done)`
- `loss`: MSE between predicted Q-values and target Q-values

---

## ⚙️ Training Process and Setups

- Environment reset includes diverse initial positions (randomized).
- Reward shaping includes both sparse reward and movement incentive.
- Target network synced periodically.
- Experience replay sampled in mini-batches.
- The original game was constrained to catch the item once the catcher rectangle fully overlaps the items, but for the sake of training the agent this constraint is removed and only partial overlap can catch the item.

## Training Strategies
- Training with randomized actions for exploration
- Training with heuristic guided movement actions for exploration
- Training with mixed (weighted sampling of heuristic guided movements and random movements) actions
- Training in multiple iterations using various epsilon values

---

## 🚧 Challenges Faced

- Agent stuck in random places and oscillating behaviors
- Poor generalization to rare spatial configurations
- Sparse feedback from environment
- Difficult to handle spatial symmetries with vanilla MLP

---

## ✅ Final Outcome

The agent learned to play the Catch-It game

The final agent is capable of **consistently catching items across the environment**, indicating convergence and success.

---

## ⚠️ Evaluation Strategy and Stuck State Mitigation
Sometimes the model is getting stuck at places and can't find the next appropriate action. To solve this isssue we can use stochastic policy execution with 5% randomness at testing times. Although, in most of the cases the policy itself is found to be capable of handling this and is not getting stuck. 

---

## 📌 Future Improvements

- Use **Dueling DQN** or **Double DQN** for more stable learning
- Try **Actor-Critic** methods like PPO
- Include **frame stacking or visual input** for scaling
- Make the catcher accelarate or deaccelarate according to need to catch the items faster to break the bondage of time
---

## 👨‍💻 Author

- Arnab Singha, Computer Science department, RKMVERI

---
