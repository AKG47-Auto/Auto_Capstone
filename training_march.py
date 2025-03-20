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
from scipy.interpolate import make_interp_spline

SEED = 123
SECONDS_PER_EPISODE = 30

class CargoBalancingEnv(gym.Env):
    def __init__(self, xml_file="/home/aayush/Documents/RL_Training/rover_scaled.xml", waypoint_file="/home/aayush/Documents/RL_Training/straight_line_waypoints.csv", render_mode=None):
        super(CargoBalancingEnv, self).__init__()
        
        # Load MuJoCo model from XML file
        waypoints_df = pd.read_csv(waypoint_file)
        print(f"waypoint:{waypoints_df.head()}")
        self.waypoints = waypoints_df[['x', 'y', 'theta']].to_numpy()
        print(f"self waypoints{self.waypoints}")
        self.current_waypoint_index = 0
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)   
        self.goal_position = self.waypoints[-1]
        print(f"goal: {self.goal_position}")
        shape = self.waypoints.shape
        print(f"shape: {shape}")
        self.x_new, self.y_smooth, self.spline = self.create_curve_from_waypoints()

    
        # Define observation space: vehicle states + cargo dynamics
        large_value = 1e10
        self.observation_space = gym.spaces.Box(low=-large_value, high=large_value, shape=(8,), dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None
        # Action space: [acceleration, braking]
        action_low = np.array([0, -1, -1])
        action_high = np.array([1, 0, 1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.state = np.zeros(8)
        #vehicle initial pose
        self.initial_x = -1.04390184e-05
        self.initial_y = -8.56218164e-09
        self.initial_vx = -3.71728527e-14
        self.initial_vy = 1.41909649e-16
        #cargo initial pose
        self.initial_cx = 1.05120351e-05
        self.initial_cy = -5.81037754e-10 
        self.initial_cz = 1.34967297e+00
        self.initial_yaw = 0  
        self.time_step = 0
        self.state_history = []
                
    def create_curve_from_waypoints(self):
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        
        if len(x) < 2:
            raise ValueError("At least two waypoints are required to create a curve.")
        
        x_new = np.linspace(min(x), max(x), 300)
        spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
        y_smooth = spline(x_new)
        
        return x_new, y_smooth, spline
    
    def shortest_distance_to_spline(self, point):
        x_point, y_point = point
        distances = []
        
        for x_spline, y_spline in zip(self.x_new, self.y_smooth):
            dist = np.sqrt((x_spline - x_point)**2 + (y_spline - y_point)**2)
            distances.append(dist)
        
        min_index = np.argmin(distances)
        closest_point = (self.x_new[min_index], self.y_smooth[min_index])
        spline_derivative = self.spline.derivative()
        heading = spline_derivative(self.x_new[min_index])
        looahead_index = min_index + 15
        if looahead_index < len(self.x_new):
            lookahead_point = (self.x_new[looahead_index], self.y_smooth[looahead_index])
            lookahead_heading = spline_derivative(self.x_new[min_index + 15])
            lookahead_distance = np.sqrt((lookahead_point[0] - x_point)**2 + (lookahead_point[1] - y_point)**2)
        else:
            lookahead_point = None
            lookahead_heading = None
            lookahead_distance = None
        
        return distances[min_index], closest_point, heading , lookahead_point, lookahead_heading , lookahead_distance
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.step_counter +=1
        self.time_step += 1

        accel = action[0]
        brake = action[1]
        steering = action[2] 
        
        self.data.ctrl[:] = np.array([accel, brake, steering])  # Send control signals
       
        
        mujoco.mj_step(self.model, self.data)  # Advance simulation
        q_w, q_x, q_y, q_z = self.data.qpos[3:7]
        r = R.from_quat([q_x, q_y, q_z, q_w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        # Update state based on MuJoCo simulation
        self.state[0] = self.data.qpos[0]  # Vehicle position in X
        self.state[1] = self.data.qpos[1]  # Vehicle position in Y
        self.state[2] = yaw #Vehicle Yaw
        #self.state[2] = yaw  # Vehicle yaw angle
        self.state[3] = self.data.qvel[0]  # Vehicle velocity in X
        self.state[4] = self.data.qvel[1]  # Vehicle velocity in Y
        #self.state[5] = self.data.qvel[5] #self.data.sensordata[self.plate_gyro_sensor_index + 2]  # Yaw rate (qvel[5])
        self.state[5] = self.data.qpos[16]  # Cargo position in X axis
        self.state[6] = self.data.qpos[17]  # Cargo position in Y axis
        self.state[7] = self.data.qpos[18]  # Cargo position in Z axis

        test_point = (self.state[0], self.state[1])
        lateral_distance, closest_point, heading, lookahead, target_heading, lookahead_distance = self.shortest_distance_to_spline(test_point)
        
        self.state_history.append(self.state.copy())

        
        start_x, start_y, start_theta = self.waypoints[0]
        end_x, end_y, end_theta = self.waypoints[-1]
        path_length = np.linalg.norm([(start_x - end_x), (start_y - end_y)])
        vehicle_heading = self.state[2]
        heading_error = np.abs(vehicle_heading - heading)
        goal_proximity = np.linalg.norm([(self.state[0] - end_x),(self.state[1] - end_y) ])
        max_deviation = 3
        max_velocity = 15
        min_velocity = 1
        target_velocity = 10
        done = False
        reward = 0
        velocity = np.linalg.norm([self.state[3], self.state[4]])
        if lookahead is not None or target_heading is not None or lookahead_distance is not None:
            lookahead_heading_error = np.abs(vehicle_heading - target_heading)
            total_heading = heading_error + lookahead_heading_error
            centering_factor = max(1.0 - lateral_distance / max_deviation, 0.0)
            angle_factor = max(1.0 - abs(total_heading / np.deg2rad(30)), 0.0)
            #cargo falling off penalty
            if  self.state[7] < 1.2:
                reward = reward - 300
            if lookahead_distance > 0.1:
                reward = reward - 100
            if goal_proximity < path_length:
                reward = reward + (path_length-goal_proximity)/path_length 
            if lateral_distance > max_deviation:
                reward = reward - 300
            elif lateral_distance < 0.5:
                reward = reward + 10
            if self.episode_start + SECONDS_PER_EPISODE < time.time():
                reward = reward - 300
            if self.state[3] <= 0:
                reward = reward - 400
            elif self.state[3] < min_velocity:
                reward = reward - 10*(self.state[3] / min_velocity) * centering_factor * angle_factor
            if velocity > target_velocity:
                reward = reward - 10*(1.0 - (velocity-target_velocity) / (max_velocity-target_velocity)) * centering_factor * angle_factor
            elif velocity > max_velocity:
                reward = reward - 100
            # elif velocity < min_velocity:
            #     reward = reward - 300
        else:
            reward = reward + 300
            print("Goal Reached")
            done = True

        # print(f"State shape in step(): {self.state.shape}")
        return self.state, reward, done, done, {}
    
    def reset(self, seed=SEED, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
        
        mujoco.mj_resetData(self.model, self.data)  # Reset simulation
        self.state[:] = [self.initial_x, self.initial_y,self.initial_yaw, self.initial_vx, self.initial_vy, 
                 self.initial_cx, self.initial_cy, self.initial_cz]
        
            # Set MuJoCo state directly to match the initial state
        self.data.qpos[:2] = [self.initial_x, self.initial_y]
        q_x, q_y, q_z, q_w = R.from_euler('xyz', [0, 0, self.initial_yaw]).as_quat()
        self.data.qpos[3:7] = [q_w, q_x, q_y, q_z]
        self.data.qvel[:2] = [self.initial_vx, self.initial_vy]
        self.data.qpos[16:19] = [self.initial_cx, self.initial_cy, self.initial_cz]

        self.data.ctrl[:] = np.zeros_like(self.data.ctrl)

        self.time_step = 0
        self.step_counter = 0
        self.current_waypoint_index = 0
        self.episode_start = time.time()
        # self.state_history = []
        self.state = np.array(self.state, dtype=np.float32)
        
  #      self.trajectory = []  # Reset trajectory
        # print(f"State shape in reset(): {self.state.shape}")
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
    
    def save_state_history(self, file_path="state_history.csv"):
    # ✅ Save state history to CSV
        df = pd.DataFrame(self.state_history, columns=[
            'Vehicle X', 'Vehicle Y', 'Vehicle_Yaw' 'Velocity X', 'Velocity Y',
            'Cargo X', 'Cargo Y', 'Cargo Z'
        ])
        df.to_csv(file_path, index=False)
        print(f"State history saved to {file_path}")

# # Wrap environment for vectorized training
#org_env =  CargoBalancingEnv("/home/aayush/My_Project/rover_scaled.xml", render_mode="human")
# env = DummyVecEnv([lambda: org_env])

# # Training Block
# ##Define PPO model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_line_follow_logs/")

# #Train model
# checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./ppo_line_follow_checkpoints/")
# model.learn(total_timesteps=200000, callback=checkpoint_callback)

# #Save trained model
# model.save("ppo_line_follow")

#Simulation block
##Load and test trained model
# model = PPO.load("/home/aayush/My_Project/ppo_line_follow/")
# vec_env = DummyVecEnv([lambda: org_env])
# obs = vec_env.reset()
# org_env.render()
# while True:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
#     mujoco.mj_step(vec_env.envs[0].model, vec_env.envs[0].data)
#     org_env.viewer.sync()
#     time.sleep(0.006)

#     if done:
#         obs = vec_env.reset()
#         break

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
