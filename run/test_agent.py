"""Test a trained agent by running episodes with optional rendering."""

import argparse
import torch
import numpy as np

from src.common.seed import set_seed
from src.common.networks import create_dqn_network
from src.envs.make_env import make_cartpole_env, make_frozenlake_env


def test_dqn_agent(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    num_hidden_layers: int = 3,
    hidden_dim: int = 128,
    num_episodes: int = 5,
    render: bool = True,
    env_name: str = "cartpole"
):
    """
    Test a trained DQN agent.
    
    Args:
        checkpoint_path: Path to model checkpoint
        state_dim: State dimension
        action_dim: Action dimension
        num_hidden_layers: Number of hidden layers
        hidden_dim: Hidden layer dimension
        num_episodes: Number of test episodes
        render: Whether to render episodes
        env_name: Environment name ('cartpole' or 'frozenlake')
    """
    device = torch.device("cpu")
    
    # Create network
    network = create_dqn_network(state_dim, action_dim, num_hidden_layers, hidden_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['q_network_state_dict'])
    network.eval()
    
    # Create environment
    if env_name == "cartpole":
        env = make_cartpole_env(render_mode="human" if render else None)
    elif env_name == "frozenlake":
        env = make_frozenlake_env(version="v0", render_mode="human" if render else None)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = network(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()
    
    print(f"\nAverage reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Test a trained agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env", type=str, choices=["cartpole", "frozenlake"], default="cartpole", help="Environment")
    parser.add_argument("--arch", type=int, choices=[3, 5], default=3, help="Number of hidden layers")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.env == "cartpole":
        state_dim = 4
        action_dim = 2
    elif args.env == "frozenlake":
        state_dim = 16
        action_dim = 4
    else:
        raise ValueError(f"Unsupported environment: {args.env}")
    
    test_dqn_agent(
        checkpoint_path=args.checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        num_hidden_layers=args.arch,
        hidden_dim=args.hidden_dim,
        num_episodes=args.episodes,
        render=not args.no_render,
        env_name=args.env
    )


if __name__ == "__main__":
    main()

