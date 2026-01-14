"""Plotting utilities for visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple
import json


def save_reward_plot(
    rewards: List[float],
    save_path: str,
    title: str = "Episode Rewards",
    xlabel: str = "Episode",
    ylabel: str = "Reward"
) -> None:
    """
    Save a plot of episode rewards.
    
    Args:
        rewards: List of reward values per episode
        save_path: Path to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_steps_to_goal_plot(
    avg_steps: List[float],
    episode_indices: List[int],
    save_path: str,
    title: str = "Average Steps to Goal (Last 100 Episodes)",
    xlabel: str = "Episode (every 100)",
    ylabel: str = "Average Steps to Goal"
) -> None:
    """
    Save a plot of average steps to goal.
    
    Args:
        avg_steps: List of average step values
        episode_indices: List of episode indices corresponding to avg_steps
        save_path: Path to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_indices, avg_steps, marker='o', markersize=4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    """
    Save a colormap visualization of the Q-table.
    
    Args:
        q_table: Q-table array (n_states x n_actions)
        save_path: Path to save the plot
        title: Plot title
        include_values: Whether to include numeric values in cells
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_states, n_actions = q_table.shape
    
    fig, ax = plt.subplots(figsize=(max(8, n_actions * 2), max(6, n_states * 0.5)))
    
    # Create colormap
    im = ax.imshow(q_table, cmap='viridis', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Q-value')
    
    # Set labels
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(title)
    ax.set_xticks(range(n_actions))
    ax.set_yticks(range(n_states))
    
    # Add numeric values if requested
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


def save_loss_plot(
    losses: List[float],
    save_path: str,
    title: str = "Training Loss",
    xlabel: str = "Training Step",
    ylabel: str = "Loss"
) -> None:
    """
    Save a plot of training losses.
    
    Args:
        losses: List of loss values per training step
        save_path: Path to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, alpha=0.6, linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_metrics_json(
    metrics: dict,
    save_path: str
) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        save_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
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

