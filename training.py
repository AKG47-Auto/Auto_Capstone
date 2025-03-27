from stable_baselines3 import PPO #PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
from SB3_RL_Training import CargoBalancingEnv
import time

# class StopTrainingOnGoalCallback(BaseCallback):
#     def __init__(self, goal_position, verbose=0):
#         super(StopTrainingOnGoalCallback, self).__init__(verbose)
#         self.goal_position = goal_position
    
#     def _on_step(self) -> bool:
#         obs = self.training_env.get_attr('state')[0]  # Get current state
#         x = obs[0][0] 
#         y = obs[0][1]  # Assuming first two states are x and y coordinates
        
#         # Example condition: Stop if the agent reaches within 0.1 distance from the goal
#         if abs(x - self.goal_position[0]) < 0.1 and abs(y - self.goal_position[1]) < 0.1:
#             print(f"Goal reached at position ({x}, {y}) — Stopping training!")
#             return False
        
#         return True

SEED = 123
print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')
org_env =  CargoBalancingEnv("./rover_scaled.xml", render_mode="human")
# env = CargoBalancingEnv()
env = DummyVecEnv([lambda: org_env])

env.reset()
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.005, tensorboard_log=logdir, seed=SEED)

TIMESTEPS = 50000 # how long is each training iteration - individual steps
iters = 0
# goal_reached = False
# goal_position = env.goal_position
# print(f"Goal position: {goal_position}")
while iters<5 :  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")