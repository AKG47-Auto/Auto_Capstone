import gymnasium as gym
import numpy as np
import pandas as pd
import math
import queue
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

SEED = 123
SECONDS_PER_EPISODE = 100

class CargoBalancingEnv(gym.Env):
    def __init__(self, xml_file="./rover_scaled.xml", waypoint_file="./curve_line_waypoints.csv", render_mode=None):
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
        self.x_new, self.y_smooth, self.spline, self.total_path_length = self.create_curve_from_waypoints()

    
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
        self.initial_cz = 2.34967297e+00
        self.initial_yaw = 0  
        self.time_step = 0
        self.state_history = []
        
        self.min_speed = 5
        self.target_speed = 15
        self.max_speed = 20
        self.max_std = 0.4
        self.max_cstd = 0.04
        self.distance_from_center_history = queue.Queue(maxsize=100)
        self.cargo_deviation_history = queue.Queue(maxsize=100)

        self.lateral_error_sum = 0.0
        self.num_steps = 0
        self.total_reward = 0.0
        self.speed_accum = 0.0
        # self.logdir = logdir
        # self.writer = SummaryWriter(log_dir=self.logdir)  # TensorBoard writer


                
    def create_curve_from_waypoints(self):
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        
        if len(x) < 2:
            raise ValueError("At least two waypoints are required to create a curve.")
        
        x_new = np.linspace(min(x), max(x), 300)
        spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
        y_smooth = spline(x_new)

            # Calculate spline length using numerical integration
        dx = np.diff(x_new)
        dy = np.diff(y_smooth)
        spline_length = np.sum(np.sqrt(dx**2 + dy**2))  # Approximate arc length
        
        return x_new, y_smooth, spline, spline_length
    
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
        
        dx_remaining = np.diff(self.x_new[min_index:])
        dy_remaining = np.diff(self.y_smooth[min_index:])
        remaining_spline_length = np.sum(np.sqrt(dx_remaining ** 2 + dy_remaining ** 2))  # Approximate arc length
        
        return min_index, distances[min_index], closest_point, heading, remaining_spline_length
    
    def lookahead(self, point):
        x_point, y_point = point
        min_index, _, _, _, _ = self.shortest_distance_to_spline(point)
        spline_derivative = self.spline.derivative()
        lookahead_index = min_index + 5
        if lookahead_index < len(self.x_new):
            lookahead_point = (self.x_new[lookahead_index], self.y_smooth[lookahead_index])
            lookahead_heading = spline_derivative(self.x_new[lookahead_index])
            lookahead_distance = np.sqrt((lookahead_point[0] - x_point)**2 + (lookahead_point[1] - y_point)**2)
        else:
            lookahead_point = None
            lookahead_heading = None
            lookahead_distance = None
        
        return lookahead_distance, lookahead_heading, lookahead_point


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.num_steps +=1
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

        self.state_history.append(self.state)

        test_point = (self.state[0], self.state[1])
        _, lateral_distance, closest_point, heading, remaining_path_length = self.shortest_distance_to_spline(test_point)
        # lookahead, target_heading, lookahead_distance,
        vehicle_heading = self.state[2]
        heading_error = np.abs(vehicle_heading - heading)
        max_deviation = 3
        cargo_deviation = np.sqrt((self.state[5] - self.state[0])**2 + (self.state[6] - self.state[1])**2)

        lat_error = lateral_distance
        self.lateral_error_sum += lat_error
        avg_lateral_error = self.lateral_error_sum / self.num_steps if self.num_steps > 0 else 0.0
        #self.writer.add_scalar('Average_Lateral_Error', avg_lateral_error, self.time_step)

   
        max_heading_deviation = np.deg2rad(30)  
        if self.distance_from_center_history.full():
            self.distance_from_center_history.get()  # Remove the oldest item if the queue is full
        self.distance_from_center_history.put(lat_error)

        if self.cargo_deviation_history.full():
            self.cargo_deviation_history.get()  
        self.cargo_deviation_history.put(cargo_deviation)

        centering_factor = max(1.0 - lateral_distance / max_deviation, 0.0)
        angle_factor = max(1.0 - abs(heading_error / max_heading_deviation), 0.0)
        std = np.std(list(self.distance_from_center_history.queue))
        sdf = max(1.0 - abs(std/self.max_std), 0.0)
        c_std = np.std(list(self.cargo_deviation_history.queue))
        c_stdf = max(1.0 - abs(c_std/self.max_cstd), 0.0)
        total_distance = remaining_path_length + lat_error
        total_distance_factor = max(1.0 - total_distance / (self.total_path_length+10), 0.0)
        speed = np.sqrt(self.state[3]**2 + self.state[4]**2)
        self.speed_accum += speed
        penalty = -10
        reward = 0
        
        #Check Early Termination
        if not self.early_terminal_state(lat_error, remaining_path_length):
            reward += self.reward_fn(centering_factor,angle_factor,sdf,c_stdf,total_distance_factor)
        else:
            self.terminal_state = True
            reward += penalty
            print("Vehicle reached early termination")

        #Start Checking for completion if vehicle crossed first waypoint
        start_x = self.waypoints[:, 0]
        #print(np.shape(start_x))
        if self.state[0] > start_x[0]:
            lookahead_distance, target_heading, lookahead = self.lookahead(test_point)
            if all(v is None for v in [lookahead, target_heading, lookahead_distance]):
                self.success_state = True
                print("Vehicle reached goal")

        #Check truncation for taking too long
        if self.episode_start + SECONDS_PER_EPISODE < time.monotonic():
                self.truncated = True
                reward += penalty
                print("Episode took over 30 seconds")
        
        self.total_reward += reward

        info = {
            'average_deviation': avg_lateral_error,
            'total_reward': self.total_reward,
            'avg_speed': (self.speed_accum / self.num_steps),
            'mean_reward': (self.total_reward / self.num_steps),
            'vehicle_x' : self.state[0]
        }
                
        return self.state, reward, self.terminal_state or self.success_state, self.truncated, info
    
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

        self.initial_cargo_distance = np.sqrt((self.initial_cx - self.initial_x)**2 + (self.initial_cy - self.initial_y)**2)
        self.time_step = 0
        self.step_counter = 0
        self.current_waypoint_index = 0
        self.episode_start = time.monotonic()
        self.terminal_state = False
        self.success_state = False
        self.truncated = False
        # self.state_history = []
        self.state = np.array(self.state, dtype=np.float32)
        
        time.sleep(2)

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
            'Vehicle X', 'Vehicle Y', 'Vehicle_Yaw', 'Velocity X', 'Velocity Y',
            'Cargo X', 'Cargo Y', 'Cargo Z'
        ])
        df.to_csv(file_path, index=False)
        print(f"State history saved to {file_path}")

    def reward_fn(self, centering_factor, angle_factor, sdf, c_stdf, total_distance_factor):
        # Speed reward
        speed = np.linalg.norm([self.state[3], self.state[4]])
        min_speed = self.min_speed
        target_speed = self.target_speed
        max_speed = self.max_speed
        if speed < min_speed:
            speed_reward = speed/min_speed
        elif speed > self.target_speed:
            speed_reward = 1.0 - (speed-target_speed)/(max_speed - target_speed)
        else:
            speed_reward = 1.0

        reward = speed_reward*centering_factor*angle_factor*sdf*c_stdf*total_distance_factor

        return reward
    
    def early_terminal_state(self, lat_error, remaining_path_length):

        total_distance = remaining_path_length + lat_error

        if self.state[7] < 1:
            print(f"Early termination: Vehicle height dropped below initial level. Cargo_z = {self.state[7]}, initial_cz = {self.initial_cz}")
            return True
        if self.state[3] < -1:
            print(f"Early termination: Vehicle moving in reverse. V_x = {self.state[3]}")
            return True
        # if total_distance > (self.total_path_length + 1):
        #     print(f"Early termination: Total distance exceeded threshold. total_distance = {total_distance}, threshold = {self.total_path_length}")
        #     return True
        if lat_error > 3:
            print(f"Early termination: Lateral error exceeded threshold. lat_error = {lat_error}, threshold = 3")
            return True
        return False

        