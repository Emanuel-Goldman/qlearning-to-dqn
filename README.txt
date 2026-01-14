================================================================================
RL Assignment Submission - Q-Learning to DQN
================================================================================

This submission contains three self-contained Python scripts, one for each
section of the assignment.

STRUCTURE:
----------
qlearning-to-dqn/
  ├── section1_qlearning_frozenlake.py  (Section 1: Tabular Q-Learning)
  ├── section2_dqn_cartpole.py         (Section 2: Basic DQN)
  ├── section3_ddqn_cartpole.py        (Section 3: Double DQN)
  ├── requirements.txt                 (Python dependencies)
  ├── README.txt                        (This file)
  └── results/                         (Output directory)
      ├── section1/                    (Section 1 outputs)
      ├── section2/                    (Section 2 outputs)
      └── section3/                    (Section 3 outputs)

Each script is fully self-contained and includes all necessary dependencies
embedded within the file. No external imports from src/ or run/ folders are
required.

REQUIREMENTS:
-------------
- Python 3.7+
- numpy
- matplotlib
- gymnasium
- torch (for sections 2 and 3)
- tensorboard (for sections 2 and 3)

SETUP INSTRUCTIONS:
-------------------

1. Create a virtual environment (recommended):
   
   On Linux/Mac:
     python3 -m venv venv
     source venv/bin/activate
   
   On Windows:
     python -m venv venv
     venv\Scripts\activate

2. Install dependencies:
     pip install -r requirements.txt

3. Verify installation:
     python -c "import numpy, matplotlib, gymnasium, torch; print('All packages installed successfully!')"

Note: If you don't want to use a virtual environment, you can install directly:
     pip install -r requirements.txt

RUNNING THE SCRIPTS:
--------------------

IMPORTANT: All scripts should be run from the project root directory.

Step-by-step instructions:
--------------------------

1. Navigate to the project root directory (where the scripts are located):
     cd /path/to/qlearning-to-dqn

2. Make sure your virtual environment is activated (if using one):
     source venv/bin/activate    # Linux/Mac
     # OR
     venv\Scripts\activate      # Windows

3. Run the desired section script (see examples below)

4. Wait for training to complete (this may take several minutes to hours depending
   on your hardware and number of episodes)

5. Check the results/ folder for generated outputs

What to expect during execution:
---------------------------------
- Section 1: Progress messages will appear, showing training completion
- Section 2 & 3: Progress updates every 100 episodes showing:
    * Episode number
    * Current episode reward
    * Moving average reward (last 100 episodes)
    * Current epsilon value
- All sections: A summary will be printed at the end with final metrics

Section 1: Q-Learning on FrozenLake
------------------------------------

Command:
  python section1_qlearning_frozenlake.py \
      --episodes 5000 \
      --max-steps 100 \
      --learning-rate 0.3 \
      --discount 0.99 \
      --epsilon-decay 0.995 \
      --seed 42

Explanation:
  - --episodes 5000: Train for 5000 episodes
  - --max-steps 100: Maximum steps per episode before timeout
  - --learning-rate 0.3: Learning rate (alpha) for Q-table updates
  - --discount 0.99: Discount factor (gamma) for future rewards
  - --epsilon-decay 0.995: Epsilon decay rate per episode (exploration decay)
  - --seed 42: Random seed for reproducibility

Quick test (fewer episodes):
  python section1_qlearning_frozenlake.py --episodes 100 --seed 42

Outputs saved to: results/section1/
  - episode_rewards.png          # Plot of rewards per episode
  - avg_steps_to_goal.png        # Moving average of steps to reach goal
  - qtable_episode_500.png        # Q-table visualization at episode 500
  - qtable_episode_2000.png       # Q-table visualization at episode 2000
  - qtable_final.png             # Final Q-table visualization
  - metrics.json                 # All metrics with hyperparameters
  - hparams.json                 # Hyperparameters only

Section 2: DQN on CartPole
---------------------------

Command:
  python section2_dqn_cartpole.py \
      --episodes 1000 \
      --arch 3 \
      --hidden-dim 128 \
      --lr 1e-3 \
      --batch-size 32 \
      --target-update 10 \
      --epsilon-decay 0.995 \
      --seed 42

Explanation:
  - --episodes 1000: Train for 1000 episodes
  - --arch 3: Use 3 hidden layers (options: 3 or 5)
  - --hidden-dim 128: Number of neurons per hidden layer
  - --lr 1e-3: Learning rate (0.001)
  - --batch-size 32: Batch size for training from replay buffer
  - --target-update 10: Update target network every 10 steps
  - --epsilon-decay 0.995: Epsilon decay rate per episode
  - --seed 42: Random seed for reproducibility

Quick test (fewer episodes):
  python section2_dqn_cartpole.py --episodes 100 --seed 42

Outputs saved to: results/section2/
  - episode_rewards.png          # Plot of rewards per episode
  - training_loss.png            # Plot of training loss over time
  - metrics.json                 # All metrics with hyperparameters
  - hparams.json                 # Hyperparameters only
  - checkpoints/final.pt         # Final model checkpoint
  - logs/                        # TensorBoard logs (view with: tensorboard --logdir results/section2/logs)

Section 3: Double DQN on CartPole
----------------------------------

Command:
  python section3_ddqn_cartpole.py \
      --episodes 1000 \
      --arch 3 \
      --hidden-dim 128 \
      --lr 1e-3 \
      --batch-size 32 \
      --target-update 10 \
      --epsilon-decay 0.995 \
      --seed 42

Explanation:
  - Same parameters as Section 2, but uses Double DQN algorithm
  - Double DQN improves stability by using online network to select actions
    and target network to evaluate them

Quick test (fewer episodes):
  python section3_ddqn_cartpole.py --episodes 100 --seed 42

Outputs saved to: results/section3/
  - episode_rewards.png          # Plot of rewards per episode
  - training_loss.png            # Plot of training loss over time
  - metrics.json                 # All metrics with hyperparameters
  - hparams.json                 # Hyperparameters only
  - checkpoints/final.pt         # Final model checkpoint
  - logs/                        # TensorBoard logs (view with: tensorboard --logdir results/section3/logs)

Viewing TensorBoard logs (Sections 2 & 3):
-------------------------------------------
  tensorboard --logdir results/section2/logs
  # OR
  tensorboard --logdir results/section3/logs

Then open your browser to: http://localhost:6006

Getting help:
-------------
To see all available command-line options for any script:
  python section1_qlearning_frozenlake.py --help
  python section2_dqn_cartpole.py --help
  python section3_ddqn_cartpole.py --help

HYPERPARAMETERS:
----------------
All hyperparameters can be set via command-line arguments. Use --help on any
script to see all available options.

Each script saves hyperparameters in two places:
  1. hparams.json - standalone hyperparameter file
  2. metrics.json - metrics with hyperparameters included

REPRODUCIBILITY:
----------------
All scripts support the --seed argument for reproducibility. The seed controls:
  - Python random module
  - NumPy random number generator
  - PyTorch random number generator (sections 2 and 3)
  - Gymnasium environment action/observation spaces
  - Initial environment state

Running the same command twice with the same seed will produce identical
results (or extremely close, within numerical precision).

TROUBLESHOOTING:
----------------

Common issues and solutions:

1. "ModuleNotFoundError" or "No module named 'X'":
   - Make sure you've installed all dependencies: pip install -r requirements.txt
   - Verify your virtual environment is activated

2. "Permission denied" or "Access denied":
   - On Linux/Mac: Make sure scripts are executable or use: python script_name.py
   - On Windows: Use: python script_name.py (not python3)

3. Script runs but produces no output:
   - Check that you're running from the project root directory
   - Outputs are saved to results/section{N}/ relative to where you run the script

4. Out of memory errors (Sections 2 & 3):
   - Reduce batch size: --batch-size 16
   - Reduce hidden dimensions: --hidden-dim 64
   - Use CPU instead of GPU: --device cpu

5. Training is too slow:
   - Reduce number of episodes for testing: --episodes 100
   - Use GPU if available: --device cuda (requires CUDA-enabled PyTorch)

6. Results differ between runs:
   - Make sure you're using the same --seed value
   - Check that you're using the same hyperparameters

7. Can't find TensorBoard:
   - Install it: pip install tensorboard
   - Make sure it's in your PATH or use: python -m tensorboard.main --logdir ...

NOTES:
------
- All scripts are self-contained and do not require the src/ or run/ folders
- Output directories are created automatically in results/section{N}/
- Checkpoints are saved periodically during training (sections 2 and 3)
- TensorBoard logs can be viewed with: tensorboard --logdir results/section{N}/logs
- Training time varies: Section 1 (~5-15 min), Sections 2&3 (~30 min - 2 hours) depending on hardware
- For faster testing, use fewer episodes (e.g., --episodes 100)
