# Assignment 1: From Q-learning to Deep Q-learning (DQN)

**Student Name/ID**: TODO: Add your name and student ID

## Section 1: Tabular Q-learning (FrozenLake)

### 1.1 Implementation Details

TODO: Briefly describe your Q-learning implementation (2-3 sentences).

### 1.2 Hyperparameters

- Learning rate (α): TODO: Final value
- Discount factor (γ): TODO: Final value
- Epsilon decay: TODO: Final value
- Number of episodes: 5000
- Max steps per episode: 100

**Rationale**: TODO: Explain why you chose these hyperparameters (3-4 sentences).

### 1.3 Results

**Summary Metrics**:
- Mean reward: TODO: Value
- Success rate: TODO: Value
- Mean steps to goal: TODO: Value

**Q-table Evolution**:
- After 500 episodes: TODO: Brief observation (1-2 sentences)
- After 2000 episodes: TODO: Brief observation (1-2 sentences)
- Final: TODO: Brief observation (1-2 sentences)

**Plots**:
- Episode rewards plot shows: TODO: Describe trends (2-3 sentences)
- Average steps to goal plot shows: TODO: Describe trends (2-3 sentences)

### 1.4 Analysis

TODO: Analyze the learning process. Discuss convergence, exploration vs exploitation trade-off, and any challenges encountered (4-5 sentences).

## Section 2: Basic DQN (CartPole)

### 2.1 Implementation Details

TODO: Briefly describe your DQN implementation, including network architecture, experience replay, and target network updates (3-4 sentences).

### 2.2 Network Architectures

**3-Layer Architecture**:
- Architecture: TODO: Describe (e.g., 4 → 128 → 128 → 128 → 2)
- Episodes to reach avg reward ≥ 475: TODO: Value (or "Not reached")

**5-Layer Architecture**:
- Architecture: TODO: Describe (e.g., 4 → 128 → 128 → 128 → 128 → 128 → 2)
- Episodes to reach avg reward ≥ 475: TODO: Value (or "Not reached")

**Comparison**: TODO: Compare the two architectures. Which performed better? Why? (3-4 sentences)

### 2.3 Hyperparameters

**Final chosen architecture**: TODO: 3-layer or 5-layer

- Learning rate: TODO: Final value
- Batch size: TODO: Final value
- Target network update frequency (C): TODO: Final value
- Epsilon decay: TODO: Final value
- Replay buffer size: 10000
- Discount factor (γ): 0.99

**Rationale**: TODO: Explain your hyperparameter choices (4-5 sentences).

### 2.4 Results

**Training Metrics**:
- Mean reward: TODO: Value
- Best 100-episode average reward: TODO: Value
- Episodes until avg reward ≥ 475: TODO: Value
- Total training steps: TODO: Value

**Plots**:
- Episode rewards: TODO: Describe trends (2-3 sentences)
- Training loss: TODO: Describe trends (2-3 sentences)

**TensorBoard Logs**: TODO: Mention any interesting observations from TensorBoard (2-3 sentences).

### 2.5 Analysis

TODO: Analyze the DQN training process. Discuss the role of experience replay, target networks, and epsilon-greedy exploration. Comment on convergence and stability (5-6 sentences).

## Section 3: Improved DQN

### 3.1 Improvement Choice

**Selected improvement**: Double DQN (DDQN)

TODO: If you chose a different improvement, describe it here. Otherwise, confirm Double DQN.

**Rationale**: TODO: Explain why you chose this improvement (2-3 sentences).

### 3.2 Implementation Details

TODO: Describe how Double DQN differs from basic DQN. Specifically, explain how the target Q-value computation differs (3-4 sentences).

### 3.3 Hyperparameters

- Learning rate: TODO: Final value (may be same as Section 2 or different)
- Batch size: TODO: Final value
- Target network update frequency (C): TODO: Final value
- Other hyperparameters: TODO: List any changes from Section 2

**Rationale**: TODO: Explain any changes from Section 2 (2-3 sentences).

### 3.4 Results

**Training Metrics**:
- Mean reward: TODO: Value
- Best 100-episode average reward: TODO: Value
- Episodes until avg reward ≥ 475: TODO: Value

**Comparison with Basic DQN**:

| Metric | Basic DQN | Double DQN | Improvement |
|--------|-----------|------------|-------------|
| Episodes to 475 | TODO | TODO | TODO |
| Best 100-ep avg | TODO | TODO | TODO |
| Mean reward | TODO | TODO | TODO |

### 3.5 Analysis

TODO: Analyze the improvement. Did Double DQN perform better than basic DQN? Why or why not? Discuss the theoretical benefits and whether they were observed in practice (5-6 sentences).

## Overall Conclusions

TODO: Summarize the key findings from all three sections. Discuss the progression from tabular Q-learning to DQN to improved DQN, and what you learned (6-7 sentences).

## How to Run

See `README.md` for detailed instructions. Quick commands:

1. **Section 1**: `python run/train_qlearning_frozenlake.py`
2. **Section 2**: `python run/train_dqn_cartpole.py --arch 3` (or `--arch 5`)
3. **Section 3**: `python run/train_ddqn_cartpole.py --arch 3`

All outputs are saved to the `artifacts/` directory.

## References

TODO: Add any references you used (papers, documentation, etc.)

