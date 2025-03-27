from stable_baselines3.common import env_checker
from SB3_RL_Training import CargoBalancingEnv
env = CargoBalancingEnv()
env_checker.check_env(env, warn=True, skip_render_check=True)