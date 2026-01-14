"""Environment creation utilities with version compatibility."""

import gymnasium as gym
from typing import Tuple, Optional


def make_frozenlake_env(
    version: str = "v0",
    is_slippery: bool = True,
    render_mode: Optional[str] = None
) -> gym.Env:
    """
    Create FrozenLake environment with version fallback.
    
    Tries the specified version first, then falls back to v1 if v0 fails.
    
    Args:
        version: Environment version ('v0' or 'v1')
        is_slippery: Whether the environment is slippery
        render_mode: Render mode ('human', 'rgb_array', or None)
        
    Returns:
        Gymnasium environment instance
    """
    env_name = f"FrozenLake-{version}"
    
    try:
        env = gym.make(env_name, is_slippery=is_slippery, render_mode=render_mode)
        return env
    except Exception as e:
        # Fallback to v1 if v0 fails
        if version == "v0":
            print(f"Warning: {env_name} not available, trying FrozenLake-v1")
            try:
                env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode=render_mode)
                return env
            except Exception as e2:
                raise RuntimeError(f"Failed to create FrozenLake environment: {e2}")
        else:
            raise RuntimeError(f"Failed to create {env_name}: {e}")


def make_cartpole_env(render_mode: Optional[str] = None) -> gym.Env:
    """
    Create CartPole-v1 environment.
    
    Args:
        render_mode: Render mode ('human', 'rgb_array', or None)
        
    Returns:
        Gymnasium environment instance
    """
    return gym.make("CartPole-v1", render_mode=render_mode)

