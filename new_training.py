from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
from training_march import CargoBalancingEnv
import time

SEED = 123
print('Starting training script...')

# Setup logging and model save directories
TIMESTAMP = int(time.time())
models_dir = f"models/{TIMESTAMP}/"
logdir = f"logs/{TIMESTAMP}/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Initialize environment
print('Connecting to environment...')
org_env = CargoBalancingEnv("./rover_scaled.xml", render_mode="human")
org_env = Monitor(org_env)
env = DummyVecEnv([lambda: org_env])

env.reset()
print('Environment has been reset.')

# Initialize PPO model with logging and adjusted entropy coefficient
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    ent_coef=0.005,  # Lower entropy for more stable learning
    tensorboard_log=logdir,
    seed=SEED
)

# Callbacks
eval_callback = EvalCallback(
    env,
    best_model_save_path=models_dir,
    log_path=logdir,
    eval_freq=5000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save model every 10k steps
    save_path=models_dir,
    name_prefix="ppo_model"
)

# Training loop
TIMESTEPS = 100_000  # Train in smaller increments for better monitoring
iters = 0
TOTAL_ITERS = 10

print(f"Starting training for {TOTAL_ITERS} iterations...")

while iters < TOTAL_ITERS:
    iters += 1
    print(f"\nIteration {iters} starting...")
    
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        tb_log_name="PPO",
        # callback=[eval_callback, checkpoint_callback]
    )
    
    print(f"Iteration {iters} complete.")
    model.save(f"{models_dir}/ppo_{TIMESTEPS * iters}")
    print(f"Model saved at iteration {iters}.")

print("Training complete!")

# Save final model
model.save(f"{models_dir}/ppo_final")
print("Final model saved.")
