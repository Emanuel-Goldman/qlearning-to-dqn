"""Section 3: Double DQN training for CartPole environment.
Self-contained script with all dependencies embedded."""

import argparse
import json
import numpy as np
import os
import random
from typing import List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# SEEDING UTILITIES
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_cartpole_env(render_mode: Optional[str] = None) -> gym.Env:
    """Create CartPole-v1 environment."""
    return gym.make("CartPole-v1", render_mode=render_mode)


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


def save_loss_plot(losses: List[float], save_path: str, title: str = "Training Loss") -> None:
    """Save a plot of training losses."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, alpha=0.6, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
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
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    serializable_metrics = convert_to_serializable(metrics)
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Fixed-size experience replay buffer using deque."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network with configurable hidden layers."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128, 128)):
        super(DQN, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def create_dqn_network(state_dim: int, action_dim: int, num_hidden_layers: int = 3, hidden_dim: int = 128) -> DQN:
    """Create a DQN network with specified number of hidden layers."""
    if num_hidden_layers == 3:
        hidden_dims = (hidden_dim, hidden_dim, hidden_dim)
    elif num_hidden_layers == 5:
        hidden_dims = (hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim)
    else:
        raise ValueError(f"Unsupported number of hidden layers: {num_hidden_layers}. Use 3 or 5.")
    return DQN(state_dim, action_dim, hidden_dims)


# ============================================================================
# TENSORBOARD LOGGER
# ============================================================================

class TensorBoardLogger:
    """Helper class for TensorBoard logging."""
    
    def __init__(self, log_dir: str, comment: Optional[str] = None):
        if comment:
            log_dir = os.path.join(log_dir, comment)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def close(self) -> None:
        self.writer.close()


# ============================================================================
# DOUBLE DQN AGENT
# ============================================================================

class DoubleDQNAgent:
    """Double DQN agent (improved DQN)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_hidden_layers: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        batch_size: int = 32,
        replay_buffer_size: int = 10000,
        device: str = "cpu",
        loss_type: str = "mse"
    ):
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
        
        self.q_network = create_dqn_network(state_dim, action_dim, num_hidden_layers, hidden_dim).to(self.device)
        self.target_network = create_dqn_network(state_dim, action_dim, num_hidden_layers, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self) -> Optional[float]:
        """Update Q-network using Double DQN update rule.
        
        Double DQN: Use online network to select action, target network to evaluate it.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_ddqn(
    env,
    agent: DoubleDQNAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 500,
    log_dir: str = "results/section3/logs",
    output_dir: str = "results/section3",
    save_checkpoint_freq: int = 100,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float], dict]:
    """Train Double DQN agent."""
    os.makedirs(output_dir, exist_ok=True)
    logger = TensorBoardLogger(log_dir, comment="ddqn_cartpole")
    
    episode_rewards = []
    training_losses = []
    moving_avg_rewards = []
    window_size = 100
    
    best_avg_reward = -float('inf')
    episodes_to_475 = None
    
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
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            loss = agent.update()
            if loss is not None:
                training_losses.append(loss)
                logger.log_scalar("training/loss", loss)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        if len(episode_rewards) >= window_size:
            avg_reward = np.mean(episode_rewards[-window_size:])
            moving_avg_rewards.append(avg_reward)
            
            if episodes_to_475 is None and avg_reward >= 475:
                episodes_to_475 = episode + 1
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        else:
            avg_reward = np.mean(episode_rewards)
            moving_avg_rewards.append(avg_reward)
        
        logger.log_scalar("episode/reward", total_reward, episode)
        logger.log_scalar("episode/epsilon", agent.epsilon, episode)
        if len(moving_avg_rewards) > 0:
            logger.log_scalar("episode/moving_avg_reward", moving_avg_rewards[-1], episode)
        
        if (episode + 1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"checkpoint_ep{episode + 1}.pt")
            agent.save_checkpoint(checkpoint_path)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {total_reward:.1f}, "
                  f"Avg (last 100): {moving_avg_rewards[-1]:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    logger.close()
    
    final_checkpoint_path = os.path.join(output_dir, "checkpoints", "final.pt")
    agent.save_checkpoint(final_checkpoint_path)
    
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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Section 3: Train Double DQN on CartPole")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--arch", type=int, choices=[3, 5], default=3, help="Number of hidden layers")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--target-update", type=int, default=10, help="Target network update frequency")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--replay-buffer-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results/section3", help="Output directory")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = make_cartpole_env()
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Collect hyperparameters
    hparams = {
        "seed": args.seed,
        "num_episodes": args.episodes,
        "num_hidden_layers": args.arch,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.lr,
        "discount_factor": args.discount,
        "batch_size": args.batch_size,
        "target_update_freq": args.target_update,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "replay_buffer_size": args.replay_buffer_size,
        "device": args.device,
        "state_dim": state_dim,
        "action_dim": action_dim
    }
    
    # Create agent
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_hidden_layers=args.arch,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        discount_factor=args.discount,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        replay_buffer_size=args.replay_buffer_size,
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
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Save plots
    reward_plot_path = os.path.join(args.output_dir, "episode_rewards.png")
    save_reward_plot(episode_rewards, reward_plot_path)
    
    if training_losses:
        loss_plot_path = os.path.join(args.output_dir, "training_loss.png")
        save_loss_plot(training_losses, loss_plot_path)
    
    # Save hyperparameters
    hparams_path = os.path.join(args.output_dir, "hparams.json")
    save_metrics_json(hparams, hparams_path)
    
    # Save metrics (with hyperparameters included)
    metrics_with_hparams = {**hparams, **metrics}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    save_metrics_json(metrics_with_hparams, metrics_path)
    
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
