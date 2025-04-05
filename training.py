from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from utils import CustomTensorBoardCallback 
import os
from new_car_training import CargoBalancingEnv
import time

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
#org_env = CargoBalancingEnv(xml_file="./rover_scaled.xml", waypoint_file="./straight_line_waypoints.csv", render_mode="human", logdir=logdir)
org_env =  CargoBalancingEnv("./car.xml", render_mode="human")
env = DummyVecEnv([lambda: org_env])

env.reset()
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.005, tensorboard_log=logdir, seed=SEED)
callback = CustomTensorBoardCallback(log_dir=logdir)

TIMESTEPS = 500000 # how long is each training iteration - individual steps
iters = 0
# goal_reached = False
# goal_position = env.goal_position
# print(f"Goal position: {goal_position}")
while iters<5 :  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	#model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )
	model.learn(total_timesteps=TIMESTEPS, callback=callback, reset_num_timesteps=False)
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")