import gymnasium as gym
import numpy as np
import pandas as pd
import math
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import matplotlib.pyplot as plt

class CargoBalancingEnv(gym.Env):
    def __init__(self, xml_file="/home/aayush/My_Project/rover_scaled.xml", waypoint_file="/home/aayush/My_Project/straight_line_waypoints.csv", render_mode=None):
        super(CargoBalancingEnv, self).__init__()
        
        # Load MuJoCo model from XML file
        waypoints_df = pd.read_csv(waypoint_file)
        self.waypoints = waypoints_df[['x', 'y']].to_numpy()
        self.current_waypoint_index = 0
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)   
        # self.load_sensor_1_1_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "load_sensor_1_1")
        # self.load_sensor_2_2_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "load_sensor_2_2")
        # self.load_sensor_3_3_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "load_sensor_3_3")
        # self.load_sensor_4_4_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "load_sensor_4_4")
        # self.plate_gyro_sensor_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "plate_gyro_sensor")
        # self.plate_accel_sensor_index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "plate_accel_sensor")
        # self.trajectory = []
        
        # Define observation space: vehicle states + cargo dynamics
        obs_low = np.array([0, 0, -np.pi, -15, -10, -10])  
        obs_high = np.array([np.inf, np.inf, np.pi, 15, 10, 10])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None
        # Action space: [acceleration, braking, steering]
        action_low = np.array([0, -1, -1 ])
        action_high = np.array([1, 0, 1 ])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.state = np.zeros(6)
        self.state[0] = -1.04641496e-05 
        self.state[1] = -8.56172410e-09
        self.state[2] = 0
        self.state[3] = 2.82629646e-15
        self.state[4] = 1.51938895e-16  
        self.state[5] = 1.39498814e-19   
        self.time_step = 0
    def get_next_waypoint(self):
        if self.current_waypoint_index < self.waypoints.shape[0]:
            return self.waypoints[self.current_waypoint_index]
        return None
        
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        accel = action[0]
        brake = action[1]
        steering = action[2] 
        # Apply actions
        self.data.ctrl[:] = np.array([accel, brake, steering])  # Send control signals
        waypoint = self.get_next_waypoint()
        
        mujoco.mj_step(self.model, self.data)  # Advance simulation
        quaternion = self.data.qpos[3:7]
        r = R.from_quat(quaternion)
        yaw = r.as_euler('xyz', degrees=False)[2]
        # Update state based on MuJoCo simulation
        self.state[0] = self.data.qpos[0]  # Vehicle position in X
        self.state[1] = self.data.qpos[1]  # Vehicle position in Y
        self.state[2] = yaw  # Vehicle yaw angle
        self.state[3] = self.data.qvel[0]  # Vehicle velocity in X
        self.state[4] = self.data.qvel[1]  # Vehicle velocity in Y
        self.state[5] = self.data.qvel[5] #self.data.sensordata[self.plate_gyro_sensor_index + 2]  # Yaw rate (qvel[5])
        # self.state[6] = self.data.qpos[16]  # Cargo position in X axis
        # self.state[7] = self.data.qpos[17]  # Cargo position in Y axis
        # #self.state[5] = self.data.qpos[17]  # Cargo position in Z axis
        # self.state[8] = self.data.qvel[15]  # Cargo velocity in X axis
        # self.state[9] = self.data.qvel[16]  # Cargo velocity in Y axis

        if waypoint is not None:
            target_x = waypoint[0]
            target_y = waypoint[1]
            end_x = self.waypoints[-1, 0]
            end_y = self.waypoints[-1, 1]
            max_distance = 200 # np.linalg.norm([self.waypoints[1,0] - end_x, self.waypoints[0,1] - end_y])
            direction = np.arctan2(target_y - self.state[1], target_x - self.state[0])
            steering = np.clip(direction, -1.0, 1.0)
            action = np.array([accel, brake, steering])
            distance = np.linalg.norm([self.state[0] - target_x, self.state[1] - target_y])
            distance_norm = distance/max_distance
            velocity = np.sign([self.state[3]])*np.linalg.norm([self.state[3], self.state[4]])
            distance_end = np.linalg.norm([self.state[0] - end_x, self.state[1] - end_y])
            angle = abs(math.degrees(math.atan2(self.state[1], self.state[0])))
            distance_end_norm = distance_end/max_distance
            stability = velocity * abs(self.state[5])
            lateral_distance = abs(self.state[1]- target_y)
            max_steps = 10000



        if waypoint is not None and distance < 0.1:
            self.current_waypoint_index += 1  
        end_distance_diff = 200 - distance_end
        # cargo_distance = np.linalg.norm([self.state[6] - self.state[0], self.state[7] - self.state[1]])
        # cargo_distance_limit = 2.0
        # cargo_penalty = max(0, cargo_distance - cargo_distance_limit) 
        
        # Reward function (minimize cargo displacement and yaw rate)
        # Max velocity, Min dist from end, Mini dist to waypoints, Penalize cargo if far rover, Mini yaw rate, Min cargo deviation
        # reward = 0 
        # if velocity > 0:
        #     reward += 10 if velocity > 0 else -10
        # elif action[1] > 0:
        #     reward -= 10 
        # elif distance > max_distance_waypoint:
        #     reward -= 10
        # elif distance_end > 201:
        #     reward -= 10
        # elif distance_end < 200:
        #     reward += 8
        # elif velocity > 0:
        #     reward += 8
        # else :
        #     reward = 8 * stability
        reward = 0
        reward += end_distance_diff #50 if distance_end > 200 else +10
        reward +=  10*velocity if velocity >0 else -200
        # reward += 8*stability
        reward -= distance #50 if distance > 0.1 else +10
        reward -= 10*angle
        reward -= 100*lateral_distance
        reward -= 100*self.time_step/max_steps


        # Check termination conditions
        terminated = (distance_end < 1)
        truncated = False  # Change if early stopping conditions exist
        if self.time_step > max_steps:  # Prevent overly long episodes
            truncated = True
        # if self.state[0] > 200:
        #     truncated = True
        # if distance_end > 205:
        #     truncated = True  
        # if angle > 30:
        #     truncated = True 
        
        self.time_step += 1
        # self.trajectory.append((self.state[0], self.state[1]))
        print(f"location: {self.state[0]}, {self.state[1]} ")
        print(f"velocity: {velocity}")
        #print(f"cargo_distance: {cargo_distance}")
        print(f"end_dist: {distance_end}")
        print(f"steps: {self.time_step}")
        print(f"distance: {distance}")

        
        return self.state, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state[0] = -1.04641496e-05 
        self.state[1] = -8.56172410e-09
        self.state[2] = 0
        self.state[3] = 2.82629646e-15
        self.state[4] = 1.51938895e-16  
        self.state[5] = 1.39498814e-19
        self.time_step = 0
        self.current_waypoint_index = 0
  #      self.trajectory = []  # Reset trajectory
        mujoco.mj_resetData(self.model, self.data)  # Reset simulation
        return self.state, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Wrap environment for vectorized training
org_env =  CargoBalancingEnv("/home/aayush/My_Project/rover_scaled.xml", render_mode="human")
env = DummyVecEnv([lambda: org_env])

# Training Block
# ##Define PPO model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_line_follow_logs/")

# #Train model
# checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./ppo_line_follow_checkpoints/")
# model.learn(total_timesteps=200000, callback=checkpoint_callback)

# #Save trained model
# model.save("ppo_line_follow")

#Simulation block
# ##Load and test trained model
model = PPO.load("ppo_line_follow/")
vec_env = DummyVecEnv([lambda: org_env])
obs = vec_env.reset()
org_env.render()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
    org_env.viewer.sync()
    time.sleep(0.006)

    if done:
        obs = vec_env.reset()
        break

#Trajectory plotting map
# print("Trajectory list:", org_env.trajectory)
# trajectory = np.array(org_env.trajectory)
# print(trajectory.shape)

# plt.figure(figsize=(8, 6))
# plt.plot(trajectory[:, 0], trajectory[:, 1], label="Rover Path", color="blue")
# plt.scatter(org_env.waypoints[:, 0], org_env.waypoints[:, 1], color="red", marker="x", label="Waypoints")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Rover Trajectory in MuJoCo")
# plt.legend()
# plt.grid()
# plt.show()