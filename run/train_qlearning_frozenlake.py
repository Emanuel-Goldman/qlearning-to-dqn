"""Tabular Q-learning for FrozenLake environment."""

import argparse
import numpy as np
import os
from typing import List, Tuple

from src.common.seed import set_seed
from src.common.plotting import (
    save_reward_plot,
    save_steps_to_goal_plot,
    save_qtable_colormap,
    save_metrics_json
)
from src.envs.make_env import make_frozenlake_env


class QLearningAgent:
    """Tabular Q-learning agent."""
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,  # TODO: Tune alpha for FrozenLake
        discount_factor: float = 0.99,  # TODO: Tune gamma for FrozenLake
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995  # TODO: Tune epsilon_decay for FrozenLake
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table to zeros
        self.q_table = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_qlearning(
    env,
    agent: QLearningAgent,
    num_episodes: int = 5000,
    max_steps_per_episode: int = 100,
    save_checkpoints: List[int] = [500, 2000],
    output_dir: str = "artifacts/section1"
) -> Tuple[List[float], List[int], dict]:
    """
    Train Q-learning agent.
    
    Args:
        env: Gymnasium environment
        agent: Q-learning agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_checkpoints: Episode numbers at which to save Q-table visualizations
        output_dir: Directory to save outputs
        
    Returns:
        Tuple of (episode_rewards, steps_to_goal, metrics_dict)
    """
    episode_rewards = []
    steps_to_goal = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for episode in range(num_episodes):
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
        
        # If episode didn't reach goal, set steps to max_steps_per_episode
        if not done:
            steps = max_steps_per_episode
        
        episode_rewards.append(total_reward)
        steps_to_goal.append(steps)
        
        agent.decay_epsilon()
        
        # Save Q-table checkpoints
        if (episode + 1) in save_checkpoints:
            checkpoint_path = os.path.join(
                output_dir,
                f"qtable_episode_{episode + 1}.png"
            )
            save_qtable_colormap(
                agent.q_table,
                checkpoint_path,
                title=f"Q-Table after {episode + 1} Episodes"
            )
    
    # Save final Q-table
    final_path = os.path.join(output_dir, "qtable_final.png")
    save_qtable_colormap(
        agent.q_table,
        final_path,
        title="Q-Table (Final)"
    )
    
    # Compute metrics
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
    """
    Compute moving average of steps to goal.
    
    Args:
        steps_to_goal: List of steps per episode
        window: Window size for moving average
        
    Returns:
        Tuple of (averages, episode_indices) where indices are every 100 episodes
    """
    averages = []
    indices = []
    
    for i in range(window - 1, len(steps_to_goal), 100):
        window_data = steps_to_goal[i - window + 1:i + 1]
        avg = np.mean(window_data)
        averages.append(avg)
        indices.append(i + 1)  # 1-indexed episode number
    
    return averages, indices


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning on FrozenLake")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="artifacts/section1", help="Output directory")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = make_frozenlake_env(version="v0", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Create agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon_decay=args.epsilon_decay
    )
    
    # Train
    print("Starting Q-learning training...")
    episode_rewards, steps_to_goal, metrics = train_qlearning(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir
    )
    
    # Save plots
    reward_plot_path = os.path.join(args.output_dir, "episode_rewards.png")
    save_reward_plot(episode_rewards, reward_plot_path)
    
    # Compute and save moving average steps plot
    avg_steps, episode_indices = compute_moving_average_steps(steps_to_goal, window=100)
    steps_plot_path = os.path.join(args.output_dir, "avg_steps_to_goal.png")
    save_steps_to_goal_plot(avg_steps, episode_indices, steps_plot_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    save_metrics_json(metrics, metrics_path)
    
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

