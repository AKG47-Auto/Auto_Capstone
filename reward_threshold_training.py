from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from training_march import CargoBalancingEnv
import time
import numpy as np

# Reward Threshold Callback
class RewardThresholdCallback(BaseCallback):
    def __init__(self, reward_threshold, patience=5, verbose=1):
        super(RewardThresholdCallback, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.consecutive_success = 0

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            reward = self.locals['episode']['r']
            self.episode_rewards.append(reward)
            
            # Compute mean reward over last 100 episodes
            if len(self.episode_rewards) > 100:
                mean_reward = np.mean(self.episode_rewards[-100:])
                
                if mean_reward > self.reward_threshold:
                    self.consecutive_success += 1
                    if self.consecutive_success >= self.patience:
                        print(f"✅ Reward threshold {self.reward_threshold} achieved for {self.patience} consecutive episodes. Stopping training.")
                        return False  # Stop training
                else:
                    self.consecutive_success = 0
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

                if self.verbose > 0:
                    print(f"Mean reward over last 100 episodes: {mean_reward:.2f}")

        return True

# Set random seed for reproducibility
SEED = 123
print('This is the start of training script')

# Create folders for logs and models
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Connect to the environment
print('Connecting to environment...')
org_env = CargoBalancingEnv("/home/aayush/Documents/Steering_Training/rover_scaled.xml", render_mode="human")
env = DummyVecEnv([lambda: org_env])

env.reset()
print('Env has been reset as part of launch')

# Define the model with logging
model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01, tensorboard_log=logdir)

# Reward threshold and patience
reward_threshold = 200  # Target average reward
patience = 5  # Number of consecutive successful evaluations before stopping

# Initialize the callback
reward_callback = RewardThresholdCallback(reward_threshold, patience=patience)

# Training loop
TIMESTEPS = 100000
iters = 0
while iters < 4:
    iters += 1
    print(f"Iteration {iters} is to commence...")
    
    # Train the model with the callback
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=reward_callback)
    
    print(f"Iteration {iters} has been trained")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")

print("✅ Training completed.")
