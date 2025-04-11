import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
beta_max = CONFIG["reward_params"]["beta_max"]
lookahead_distance = CONFIG["reward_params"]["lookahead_distance"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps
            speed = env.vehicle.get_speed()
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# Reward_fn5
# def reward_fn5(env):
#     """
#         reward = Positive speed reward for being close to target speed,
#                  however, quick decline in reward beyond target speed
#                * centering factor (1 when centered, 0 when not)
#                * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
#                * distance_std_factor (1 when std from center lane is low, 0 when not)
#     """

#     angle = env.vehicle.get_angle(env.current_waypoint)
#     speed_kmh = env.vehicle.get_speed()
#     slip_angle = env.vehicle.get_slip_angle()  
#     # slip_angle_rad = np.deg2rad(slip_angle)

#     #Compute Speed Reward
#     if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
#         speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
#     elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
#         # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
#         speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
#     else:  # Otherwise
#         speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

#     #Compute Slip Angle Reward
#     if abs(slip_angle) <= beta_max:
#         slip_reward = 1.0
#     else:
#         slip_reward = 1.0 - (abs(slip_angle) - beta_max) / (beta_max - 1)
 
#     w_slip = min(abs(slip_angle) / (beta_max), 1.0)
#     w_speed = 1.0 - w_slip

#     total_reward = w_slip * slip_reward + w_speed * speed_reward

#     # Interpolated from 1 when centered to 0 when 3 m from center
#     centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

#     # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
#     angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

#     std = np.std(env.distance_from_center_history)
#     distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)
#     #Compute slip angle factor with a preview model
#     lookahead_slip = env.vehicle.get_lookahead_slip(lookahead_distance)

#     slip_angle_factor = max(1.0 - abs(lookahead_slip/beta_max), 0.0)
#     # Final reward
#     reward = total_reward * centering_factor * angle_factor * distance_std_factor #* slip_angle_factor

#     return reward

def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """
    #Add a preview stability factor to the reward function

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    speed_mps = speed_kmh * 5/18
    slip_angle = env.vehicle.get_slip_angle()
    prev_slip_angle = env.prev_slip_angle
    delta_slip = slip_angle - prev_slip_angle  
    angular_velocity = env.vehicle.get_angular_velocity()
    yaw_rate = angular_velocity.z
    eps = 1e-5
    w_yaw_rate = 1.0
    w_curvature = 1.0
    if not env.preview_points:
        return 0
    curvature = env.vehicle.estimate_curvature(env.preview_points)

    yaw_contrib = abs(yaw_rate / max(speed_mps, eps)) #(yaw_rate / speed_mps) ** 2
    if curvature < eps :
        curvature_contrib = 0
    else:
        curvature_contrib = abs(speed_mps / max(curvature,eps)) #(speed_mps / curvature) ** 2

    for var_name, var_value in {
        "speed_kmh": speed_kmh, "yaw_rate": yaw_rate, "curvature": curvature,
        "yaw_contrib": yaw_contrib, "curvature_contrib": curvature_contrib
    }.items():
        if np.isnan(var_value) or np.isinf(var_value):
            print(f"[WARN] NaN/Inf detected in {var_name}: {var_value}")
            return 0.0

    #Compute Stability Reward
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        stability_reward = 1.0 - (min_speed - speed_kmh)**2 / (min_speed**2)
        #stability_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        stability_reward = 1.0 - (speed_kmh - target_speed)**2 / ((max_speed - target_speed)**2)
    else:  # Otherwise
        stability_reward = 1.0/(1 + w_yaw_rate * yaw_contrib + w_curvature * curvature_contrib)  # Return 1 for speeds in range [min_speed, target_speed]

    #Compute Slip Angle Reward
    if abs(slip_angle) < beta_max and (slip_angle*delta_slip) < 0:
        slip_reward = 1.0
    else:
        slip_reward = 1.0 - (abs(slip_angle + delta_slip) - beta_max) / (beta_max)
 
    w_slip = min(abs(slip_angle) / (beta_max), 1.0)
    w_speed = 1.0 - w_slip

    total_reward = w_slip * slip_reward + w_speed * stability_reward

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = total_reward * centering_factor * angle_factor * distance_std_factor 

    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_waypoints(env):
    """
        reward
            - Each time the vehicle overpasses a waypoint, it will receive a reward of 1.0
            - When the vehicle does not pass a waypoint, it receives a reward of 0.0
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)
