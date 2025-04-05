from stable_baselines3.common import env_checker
from new_car_training import CargoBalancingEnv
env = CargoBalancingEnv()
env_checker.check_env(env, warn=True, skip_render_check=True)