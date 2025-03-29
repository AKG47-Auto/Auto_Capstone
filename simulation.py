import time
import numpy as np
import mujoco
import csv
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from SB3_RL_Training import CargoBalancingEnv

#Load and test trained model
model_path = "./models/1743282777/2500000"  # Modify this to the correct path
# Load the trained PPO model
model = PPO.load(model_path)

org_env =  CargoBalancingEnv(xml_file="./rover_scaled.xml", waypoint_file="./curve_line_waypoints.csv", render_mode="human")
vec_env = DummyVecEnv([lambda: org_env])
obs = vec_env.reset()
org_env.render()
episode_count = 0
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    print("Observation shape:", obs.shape)
    print("Observation:", obs)
    print(f"Episode Count: {episode_count}")
    print(f"Vehicle_x: {info[0].get('vehicle_x',0)}")

    if done:
        episode_count += 1  # Increment the episode counter
        print(f"Episode {episode_count} done!")
        #and info[0].get('vehicle_x', 0) > 195:
        if episode_count == 1:
            org_env.save_state_history("state_history.csv")

        if episode_count >= 2:
            print("Resetting environment...")
            obs = vec_env.reset()

    mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
    org_env.viewer.sync()
    time.sleep(0.006)


## Original Code..keep it safe ##
# while True:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
#     print("Observation shape:", obs.shape)
#     print("Observation:", obs)
    
#     if done:
#         episode_count += 1
#         print(f"Episode {episode_count} done!")
#         if episode_count == 2:
#             print("Storing data after second episode...")
#             data_to_store.append(obs)
            
#     mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
#     org_env.viewer.sync()
#     time.sleep(0.006)

    # if done:
    #     print("Episode done. Resetting environment...")
    #     org_env.save_state_history("state_history_episode_straight.csv")
    #     obs = vec_env.reset()
        
    #     # obs = np.squeeze(obs)
    #     break