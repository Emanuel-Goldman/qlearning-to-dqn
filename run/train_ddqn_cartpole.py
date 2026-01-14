"""Improved DQN (Double DQN) training for CartPole environment."""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import List, Tuple, Optional

from src.common.seed import set_seed
from src.common.logging import TensorBoardLogger
from src.common.plotting import save_reward_plot, save_loss_plot, save_metrics_json
from src.common.replay_buffer import ReplayBuffer
from src.common.networks import create_dqn_network
from src.envs.make_env import make_cartpole_env


class DoubleDQNAgent:
    """Double DQN agent (improved DQN)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_hidden_layers: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,  # TODO: Tune learning rate for CartPole (DDQN)
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,  # TODO: Tune target_update_freq (C) for CartPole (DDQN)
        batch_size: int = 32,  # TODO: Tune batch size for CartPole (DDQN)
        replay_buffer_size: int = 10000,
        device: str = "cpu",
        loss_type: str = "mse"  # "mse" or "huber"
    ):
        """
        Initialize Double DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_hidden_layers: Number of hidden layers (3 or 5)
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial epsilon
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay rate
            target_update_freq: Frequency of target network updates (C)
            batch_size: Batch size for training
            replay_buffer_size: Size of replay buffer
            device: Device to use ('cpu' or 'cuda')
            loss_type: Loss function type ('mse' or 'huber')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.loss_type = loss_type
        
        # Create networks
        self.q_network = create_dqn_network(
            state_dim, action_dim, num_hidden_layers, hidden_dim
        ).to(self.device)
        self.target_network = create_dqn_network(
            state_dim, action_dim, num_hidden_layers, hidden_dim
        ).to(self.device)
        
        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Training step counter
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self) -> Optional[float]:
        """
        Update Q-network using Double DQN update rule.
        
        Double DQN: Use online network to select action, target network to evaluate it.
        
        Returns:
            Loss value if update was performed, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use online network to select action, target network to evaluate
        with torch.no_grad():
            # Select best action using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


def train_ddqn(
    env,
    agent: DoubleDQNAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 500,
    log_dir: str = "artifacts/section3/logs",
    output_dir: str = "artifacts/section3",
    save_checkpoint_freq: int = 100
) -> Tuple[List[float], List[float], dict]:
    """
    Train Double DQN agent.
    
    Args:
        env: Gymnasium environment
        agent: Double DQN agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        log_dir: Directory for TensorBoard logs
        output_dir: Directory for outputs
        save_checkpoint_freq: Frequency of checkpoint saves
        
    Returns:
        Tuple of (episode_rewards, training_losses, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = TensorBoardLogger(log_dir, comment="ddqn_cartpole")
    
    episode_rewards = []
    training_losses = []
    moving_avg_rewards = []
    window_size = 100
    
    best_avg_reward = -float('inf')
    episodes_to_475 = None
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update network
            loss = agent.update()
            if loss is not None:
                training_losses.append(loss)
                logger.log_scalar("training/loss", loss)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Compute moving average
        if len(episode_rewards) >= window_size:
            avg_reward = np.mean(episode_rewards[-window_size:])
            moving_avg_rewards.append(avg_reward)
            
            # Check for 475 threshold
            if episodes_to_475 is None and avg_reward >= 475:
                episodes_to_475 = episode + 1
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        else:
            avg_reward = np.mean(episode_rewards)
            moving_avg_rewards.append(avg_reward)
        
        # Logging
        logger.log_scalar("episode/reward", total_reward, episode)
        logger.log_scalar("episode/epsilon", agent.epsilon, episode)
        if len(moving_avg_rewards) > 0:
            logger.log_scalar("episode/moving_avg_reward", moving_avg_rewards[-1], episode)
        
        # Save checkpoint
        if (episode + 1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"checkpoint_ep{episode + 1}.pt")
            agent.save_checkpoint(checkpoint_path)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {total_reward:.1f}, "
                  f"Avg (last 100): {moving_avg_rewards[-1]:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    logger.close()
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_dir, "checkpoints", "final.pt")
    agent.save_checkpoint(final_checkpoint_path)
    
    # Compute metrics
    metrics = {
        "num_episodes": num_episodes,
        "final_epsilon": agent.epsilon,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "best_100_ep_avg_reward": float(best_avg_reward) if best_avg_reward > -float('inf') else None,
        "episodes_to_475": episodes_to_475,
        "total_training_steps": agent.training_step,
        "mean_loss": float(np.mean(training_losses)) if training_losses else None,
        "final_100_ep_avg_reward": float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else None
    }
    
    return episode_rewards, training_losses, metrics


def compare_with_basic_dqn(
    ddqn_metrics: dict,
    basic_dqn_metrics_path: str,
    output_path: str
) -> None:
    """
    Compare Double DQN metrics with basic DQN and save comparison.
    
    Args:
        ddqn_metrics: Double DQN metrics dictionary
        basic_dqn_metrics_path: Path to basic DQN metrics JSON file
        output_path: Path to save comparison JSON
    """
    import json
    
    # Load basic DQN metrics
    with open(basic_dqn_metrics_path, 'r') as f:
        basic_dqn_metrics = json.load(f)
    
    # Create comparison
    comparison = {
        "basic_dqn": basic_dqn_metrics,
        "double_dqn": ddqn_metrics,
        "improvement": {
            "episodes_to_475_diff": (
                ddqn_metrics["episodes_to_475"] - basic_dqn_metrics["episodes_to_475"]
                if ddqn_metrics["episodes_to_475"] is not None and basic_dqn_metrics["episodes_to_475"] is not None
                else None
            ),
            "best_avg_reward_diff": (
                ddqn_metrics["best_100_ep_avg_reward"] - basic_dqn_metrics["best_100_ep_avg_reward"]
                if ddqn_metrics["best_100_ep_avg_reward"] is not None and basic_dqn_metrics["best_100_ep_avg_reward"] is not None
                else None
            ),
            "mean_reward_diff": (
                ddqn_metrics["mean_reward"] - basic_dqn_metrics["mean_reward"]
            )
        }
    }
    
    # Save comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison
    print("\n" + "="*50)
    print("Comparison: Basic DQN vs Double DQN")
    print("="*50)
    print(f"Episodes to 475:")
    print(f"  Basic DQN: {basic_dqn_metrics['episodes_to_475']}")
    print(f"  Double DQN: {ddqn_metrics['episodes_to_475']}")
    if comparison['improvement']['episodes_to_475_diff'] is not None:
        print(f"  Difference: {comparison['improvement']['episodes_to_475_diff']} episodes")
    print(f"\nBest 100-episode avg reward:")
    print(f"  Basic DQN: {basic_dqn_metrics['best_100_ep_avg_reward']:.2f}")
    print(f"  Double DQN: {ddqn_metrics['best_100_ep_avg_reward']:.2f}")
    if comparison['improvement']['best_avg_reward_diff'] is not None:
        print(f"  Difference: {comparison['improvement']['best_avg_reward_diff']:.2f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Train Double DQN on CartPole")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--arch", type=int, choices=[3, 5], default=3, help="Number of hidden layers")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--target-update", type=int, default=10, help="Target network update frequency")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="artifacts/section3", help="Output directory")
    parser.add_argument("--compare-with", type=str, default=None, help="Path to basic DQN metrics JSON for comparison")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = make_cartpole_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_hidden_layers=args.arch,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        epsilon_decay=args.epsilon_decay,
        device=args.device
    )
    
    # Train
    print(f"Starting Double DQN training with {args.arch}-layer architecture...")
    log_dir = os.path.join(args.output_dir, "logs")
    episode_rewards, training_losses, metrics = train_ddqn(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        log_dir=log_dir,
        output_dir=args.output_dir
    )
    
    # Save plots
    reward_plot_path = os.path.join(args.output_dir, "episode_rewards.png")
    save_reward_plot(episode_rewards, reward_plot_path)
    
    if training_losses:
        loss_plot_path = os.path.join(args.output_dir, "training_loss.png")
        save_loss_plot(training_losses, loss_plot_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    save_metrics_json(metrics, metrics_path)
    
    # Compare with basic DQN if path provided
    if args.compare_with:
        comparison_path = os.path.join(args.output_dir, "comparison.json")
        compare_with_basic_dqn(metrics, args.compare_with, comparison_path)
    
    # Print summary
    print("\n" + "="*50)
    print("Training Summary (Double DQN)")
    print("="*50)
    print(f"Architecture: {args.arch} hidden layers")
    print(f"Number of episodes: {metrics['num_episodes']}")
    print(f"Final epsilon: {metrics['final_epsilon']:.4f}")
    print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    if metrics['episodes_to_475'] is not None:
        print(f"Episodes until avg reward >= 475: {metrics['episodes_to_475']}")
    else:
        print("Episodes until avg reward >= 475: Not reached")
    print(f"Best 100-episode avg reward: {metrics['best_100_ep_avg_reward']:.2f}")
    print(f"Total training steps: {metrics['total_training_steps']}")
    if metrics['mean_loss'] is not None:
        print(f"Mean training loss: {metrics['mean_loss']:.4f}")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("="*50)
    
    env.close()


if __name__ == "__main__":
    main()

