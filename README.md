# Catch-It-RL: Reinforcement Learning Agent Using DQN

## 📌 Project Overview

This project explores how a reinforcement learning agent can learn to control a rectangular paddle to catch falling items in a custom 2D environment (`CatchItEnv`). The environment and agent are designed and trained using **Deep Q-Learning (DQN)**.

---

## 🎯 Objective

- Build a simulated environment where an item falls vertically.
- Train an agent to move a paddle left/right to catch the item.
- Maximize the agent’s performance using Q-learning with function approximation (neural networks).

---

## 🧩 Environment Details

**Name**: `CatchItEnv`  
**Observation Space**:  
- `rect_x`, `rect_y`: Paddle position  
- `item_x`, `item_y`: Falling item position  
- `dx`, `dy`: Relative difference (paddle to item)

**Action Space**:  
- Discrete(5): `Left`, `Right`, `Stay`, `Far Left`, `Far Right`

**Termination Conditions**:  
- Catch or miss the falling item  
- Or max steps reached

**Reward Scheme**:
- `+1` for successful catch  
- `-1` for miss  
- Small distance-based shaping reward for guiding training

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

## ⚙️ Training Strategy

- Environment reset includes diverse initial positions (randomized).
- Reward shaping includes both sparse reward and movement incentive.
- Target network synced periodically.
- Experience replay sampled in mini-batches.

---

## 🚧 Challenges Faced

- Agent stuck in corner or oscillating behaviors
- Poor generalization to rare spatial configurations
- Sparse feedback from environment
- Difficult to handle spatial symmetries with vanilla MLP

---

## ✅ Final Outcome

Despite several tuning challenges and unsuccessful attempts using hard case tracking, the agent eventually learned **through improved state representation, better curriculum of item/paddle placement**, and manual inspection/debugging.

The final agent is capable of **consistently catching items across the environment**, indicating convergence and success.

---

## 📁 Files & Structure

- `catchit_env.py`: Custom Gym-compatible environment
- `dqn_agent.py`: DQN implementation
- `train.py`: Training loop
- `watch.py`: Visualization of trained agent

---

## 📌 Future Improvements

- Use **Dueling DQN** or **Double DQN** for more stable learning
- Try **Actor-Critic** methods like PPO
- Include **frame stacking or visual input** for scaling
- Generalize to **non-rectangular paddle** or **variable falling speed**

---

## 👨‍💻 Author

- Arnab Singha, MSc CS student at RKMVERI

---

