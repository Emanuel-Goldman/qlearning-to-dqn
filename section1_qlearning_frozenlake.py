"""Section 1: Tabular Q-learning for FrozenLake environment.
Self-contained script with all dependencies embedded."""

import argparse
import json
import numpy as np
import os
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import gymnasium as gym


# ============================================================================
# SEEDING UTILITIES
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not needed for Q-learning
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_frozenlake_env(
    version: str = "v0",
    is_slippery: bool = True,
    render_mode: Optional[str] = None
) -> gym.Env:
    """Create FrozenLake environment with version fallback."""
    env_name = f"FrozenLake-{version}"
    try:
        env = gym.make(env_name, is_slippery=is_slippery, render_mode=render_mode)
        return env
    except Exception:
        if version == "v0":
            print(f"Warning: {env_name} not available, trying FrozenLake-v1")
            try:
                env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode=render_mode)
                return env
            except Exception as e2:
                raise RuntimeError(f"Failed to create FrozenLake environment: {e2}")
        else:
            raise RuntimeError(f"Failed to create {env_name}")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def save_reward_plot(rewards: List[float], save_path: str, title: str = "Episode Rewards") -> None:
    """Save a plot of episode rewards."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_steps_to_goal_plot(
    avg_steps: List[float],
    episode_indices: List[int],
    save_path: str,
    title: str = "Average Steps to Goal (Last 100 Episodes)"
) -> None:
    """Save a plot of average steps to goal."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(episode_indices, avg_steps, marker='o', markersize=4)
    plt.title(title)
    plt.xlabel("Episode (every 100)")
    plt.ylabel("Average Steps to Goal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_qtable_colormap(
    q_table: np.ndarray,
    save_path: str,
    title: str = "Q-Table",
    include_values: bool = True
) -> None:
    """Save a colormap visualization of the Q-table."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_states, n_actions = q_table.shape
    fig, ax = plt.subplots(figsize=(max(8, n_actions * 2), max(6, n_states * 0.5)))
    im = ax.imshow(q_table, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Q-value')
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(title)
    ax.set_xticks(range(n_actions))
    ax.set_yticks(range(n_states))
    if include_values:
        for i in range(n_states):
            for j in range(n_actions):
                text = ax.text(
                    j, i, f'{q_table[i, j]:.3f}',
                    ha="center", va="center",
                    color="white" if q_table[i, j] < q_table.max() * 0.5 else "black",
                    fontsize=8
                )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_metrics_json(metrics: dict, save_path: str) -> None:
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    serializable_metrics = convert_to_serializable(metrics)
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Tabular Q-learning agent."""
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Update Q-table using Q-learning update rule."""
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_qlearning(
    env,
    agent: QLearningAgent,
    num_episodes: int = 5000,
    max_steps_per_episode: int = 100,
    save_checkpoints: List[int] = [500, 2000],
    output_dir: str = "results/section1",
    seed: Optional[int] = None
) -> Tuple[List[float], List[int], dict]:
    """Train Q-learning agent."""
    episode_rewards = []
    steps_to_goal = []
    os.makedirs(output_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        if episode == 0 and seed is not None:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
        
        if not done:
            steps = max_steps_per_episode
        
        episode_rewards.append(total_reward)
        steps_to_goal.append(steps)
        agent.decay_epsilon()
        
        if (episode + 1) in save_checkpoints:
            checkpoint_path = os.path.join(output_dir, f"qtable_episode_{episode + 1}.png")
            save_qtable_colormap(agent.q_table, checkpoint_path, title=f"Q-Table after {episode + 1} Episodes")
    
    final_path = os.path.join(output_dir, "qtable_final.png")
    save_qtable_colormap(agent.q_table, final_path, title="Q-Table (Final)")
    
    metrics = {
        "num_episodes": num_episodes,
        "final_epsilon": agent.epsilon,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_steps_to_goal": float(np.mean(steps_to_goal)),
        "std_steps_to_goal": float(np.std(steps_to_goal)),
        "success_rate": float(np.mean([r > 0 for r in episode_rewards])),
        "q_table_max": float(np.max(agent.q_table)),
        "q_table_min": float(np.min(agent.q_table)),
        "q_table_mean": float(np.mean(agent.q_table))
    }
    
    return episode_rewards, steps_to_goal, metrics


def compute_moving_average_steps(steps_to_goal: List[int], window: int = 100) -> Tuple[List[float], List[int]]:
    """Compute moving average of steps to goal."""
    averages = []
    indices = []
    for i in range(window - 1, len(steps_to_goal), 100):
        window_data = steps_to_goal[i - window + 1:i + 1]
        avg = np.mean(window_data)
        averages.append(avg)
        indices.append(i + 1)
    return averages, indices


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Section 1: Train Q-learning on FrozenLake")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--learning-rate", type=float, default=0.3, help="Learning rate (alpha)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--output-dir", type=str, default="results/section1", help="Output directory")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = make_frozenlake_env(version="v0", is_slippery=True)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Collect hyperparameters
    hparams = {
        "seed": args.seed,
        "num_episodes": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "learning_rate": args.learning_rate,
        "discount_factor": args.discount,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "n_states": n_states,
        "n_actions": n_actions
    }
    
    # Create agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay
    )
    
    # Train
    print("Starting Q-learning training...")
    episode_rewards, steps_to_goal, metrics = train_qlearning(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Save plots
    reward_plot_path = os.path.join(args.output_dir, "episode_rewards.png")
    save_reward_plot(episode_rewards, reward_plot_path)
    
    avg_steps, episode_indices = compute_moving_average_steps(steps_to_goal, window=100)
    steps_plot_path = os.path.join(args.output_dir, "avg_steps_to_goal.png")
    save_steps_to_goal_plot(avg_steps, episode_indices, steps_plot_path)
    
    # Save hyperparameters
    hparams_path = os.path.join(args.output_dir, "hparams.json")
    save_metrics_json(hparams, hparams_path)
    
    # Save metrics (with hyperparameters included)
    metrics_with_hparams = {**hparams, **metrics}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    save_metrics_json(metrics_with_hparams, metrics_path)
    
    # Print summary
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Number of episodes: {metrics['num_episodes']}")
    print(f"Final epsilon: {metrics['final_epsilon']:.4f}")
    print(f"Mean reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"Mean steps to goal: {metrics['mean_steps_to_goal']:.2f} ± {metrics['std_steps_to_goal']:.2f}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Q-table stats - Max: {metrics['q_table_max']:.4f}, Min: {metrics['q_table_min']:.4f}, Mean: {metrics['q_table_mean']:.4f}")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("="*50)
    
    env.close()


if __name__ == "__main__":
    main()
