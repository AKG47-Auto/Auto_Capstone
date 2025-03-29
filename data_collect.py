import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from SB3_RL_Training import CargoBalancingEnv

# Load and test trained model
model_path = "./models/1743114407/2500000"  # Modify this to the correct path
model = PPO.load(model_path)

org_env = CargoBalancingEnv("./rover_scaled.xml", render_mode="human")
vec_env = DummyVecEnv([lambda: org_env])

num_episodes = 10  # Run multiple episodes before saving
success_count = 0

for episode in range(num_episodes):
    obs = vec_env.reset()
    org_env.render()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        
        mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
        org_env.viewer.sync()
        time.sleep(0.006)

        if done:
            print("Episode done. Checking Success...")
            
            if org_env.success_state or org_env.terminal_state:  # Only save if episode was successful
                successful_episodes += 1
                file_name = f"state_history_success_{successful_episodes}.csv"
                org_env.save_state_history(file_name)
                print(f"✅ Successfully completed path! Data saved to {file_name}")
            else:
                print("❌ Episode failed. No data saved.")
            
            break  # Move to next episode

print(f"\n🎯 Simulation complete. {successful_episodes}/{num_episodes} episodes were successful!")
