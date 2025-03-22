import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from training_git import CargoBalancingEnv

#Load and test trained model
model_path = "./models/1742621613/250000"  # Modify this to the correct path
# Load the trained PPO model
model = PPO.load(model_path)

org_env =  CargoBalancingEnv("./rover_scaled.xml", render_mode="human")
vec_env = DummyVecEnv([lambda: org_env])
# print("Waiting for 15 seconds before starting...")
# time.sleep(15)
obs = vec_env.reset()
org_env.render()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    print("Observation shape:", obs.shape)
    print("Observation:", obs)
    #lookahead = org_env.get_lookahead_waypoint()
    # waypoint = org_env.get_waypoint()

    # #print("lookahead:", lookahead)
    # print("current waypoint:", waypoint)
    # x = obs[0][0]
    # y = obs[0][1]
    # vx = obs[0][2]
    # vy = obs[0][3]
    # velocity = vx
    # print(f"Location: {location[0]}, {location[1]}")
    # print(f"Velocity: {velocity}")
    # obs = np.squeeze(obs)
    mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
    org_env.viewer.sync()
    time.sleep(0.006)

    # if done:
    #     print("Episode done. Resetting environment...")
    #     org_env.save_state_history("state_history_episode_straight.csv")
    #     obs = vec_env.reset()
        
    #     # obs = np.squeeze(obs)
    #     break