# Assignment 1: From Q-learning to Deep Q-learning (DQN)

This project implements tabular Q-learning for FrozenLake and Deep Q-Networks (DQN) for CartPole, including an improved version using Double DQN.

## Project Structure

```
qlearning-to-dqn/
├── src/
│   ├── common/          # Shared utilities (seed, logging, plotting, replay buffer, networks)
│   └── envs/            # Environment creation utilities
├── run/                 # Training and testing scripts
├── scripts/             # Utility scripts (e.g., zip submission)
├── report/              # Report template
├── artifacts/           # Output directory for plots, metrics, checkpoints
│   ├── section1/       # Q-learning outputs
│   ├── section2/       # Basic DQN outputs
│   │   ├── arch3/      # 3-layer architecture
│   │   └── arch5/      # 5-layer architecture
│   └── section3/       # Improved DQN (Double DQN) outputs
├── README.md
└── requirements.txt
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Section 1: Tabular Q-learning (FrozenLake)

Train a Q-learning agent on FrozenLake:

```bash
python run/train_qlearning_frozenlake.py
```

With custom hyperparameters:
```bash
python run/train_qlearning_frozenlake.py \
    --seed 42 \
    --episodes 5000 \
    --learning-rate 0.1 \
    --discount 0.99 \
    --epsilon-decay 0.995
```

**Outputs** (saved to `artifacts/section1/`):
- `qtable_episode_500.png` - Q-table visualization after 500 episodes
- `qtable_episode_2000.png` - Q-table visualization after 2000 episodes
- `qtable_final.png` - Final Q-table visualization
- `episode_rewards.png` - Reward per episode plot
- `avg_steps_to_goal.png` - Average steps to goal (last 100 episodes, every 100 episodes)
- `metrics.json` - Summary metrics

### Section 2: Basic DQN (CartPole)

Train a basic DQN agent with 3-layer architecture:
```bash
python run/train_dqn_cartpole.py --arch 3
```

Train with 5-layer architecture:
```bash
python run/train_dqn_cartpole.py --arch 5
```

With custom hyperparameters:
```bash
python run/train_dqn_cartpole.py \
    --arch 3 \
    --seed 42 \
    --episodes 1000 \
    --lr 1e-3 \
    --batch-size 32 \
    --target-update 10 \
    --epsilon-decay 0.995
```

**Outputs** (saved to `artifacts/section2/arch3/` or `artifacts/section2/arch5/`):
- `episode_rewards.png` - Reward per episode plot
- `training_loss.png` - Training loss per step
- `metrics.json` - Summary metrics including episodes to 475
- `checkpoints/` - Model checkpoints
- TensorBoard logs in `logs/` directory

**View TensorBoard logs:**
```bash
tensorboard --logdir artifacts/section2/arch3/logs
```

### Section 3: Improved DQN (Double DQN)

Train Double DQN agent:
```bash
python run/train_ddqn_cartpole.py --arch 3
```

Compare with basic DQN:
```bash
python run/train_ddqn_cartpole.py \
    --arch 3 \
    --compare-with artifacts/section2/arch3/metrics.json
```

**Outputs** (saved to `artifacts/section3/`):
- Same as Section 2, plus:
- `comparison.json` - Comparison with basic DQN (if `--compare-with` is provided)

### Testing Trained Agents

Test a trained agent:
```bash
python run/test_agent.py \
    --checkpoint artifacts/section2/arch3/checkpoints/final.pt \
    --env cartpole \
    --arch 3 \
    --episodes 5
```

Test without rendering:
```bash
python run/test_agent.py \
    --checkpoint artifacts/section2/arch3/checkpoints/final.pt \
    --env cartpole \
    --arch 3 \
    --episodes 10 \
    --no-render
```

## Hyperparameters

Default hyperparameters are provided, but **you should tune them** for optimal performance. Look for `# TODO:` comments in the code for hyperparameters that need tuning:

- **Section 1 (Q-learning)**: `learning_rate` (alpha), `discount_factor` (gamma), `epsilon_decay`
- **Section 2 (DQN)**: `learning_rate`, `batch_size`, `target_update_freq` (C)
- **Section 3 (Double DQN)**: Same as Section 2

## Creating Submission

Create a submission zip file:
```bash
python scripts/zip_submission.py
```

Include artifacts (optional):
```bash
python scripts/zip_submission.py --include-artifacts
```

## Report Template

A report template is provided in `report/report_template.md` with placeholders for:
- Short answers (6-7 sentences max)
- Hyperparameter choices
- Results and analysis
- Comparison between basic DQN and improved DQN

## Notes

- All scripts use deterministic seeding by default (seed=42)
- CPU is used by default; use `--device cuda` if GPU is available
- FrozenLake version compatibility: The code tries `FrozenLake-v0` first, then falls back to `FrozenLake-v1` if needed
- Q-table visualizations include numeric values in cells
- Metrics are saved as JSON for easy reporting

## TODO Items

Before final submission, complete the following:

1. **Hyperparameter tuning**: Tune hyperparameters marked with `# TODO:` comments
2. **Final architecture choice**: Choose between 3-layer and 5-layer architectures
3. **Final improvement choice**: Confirm Double DQN or switch to another improvement (e.g., Prioritized Experience Replay)
4. **Report completion**: Fill in all placeholders in `report/report_template.md`
5. **Student information**: Add your name/ID where required

