# utils.py
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class CustomTensorBoardCallback(BaseCallback):
    def __init__(self, log_dir: str):
        super(CustomTensorBoardCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.writer.add_scalar("custom/avg_center_dev", self.locals['infos'][0]['average_deviation'], self.num_timesteps)
            self.writer.add_scalar("custom/total_reward", self.locals['infos'][0]['total_reward'], self.num_timesteps)
            self.writer.add_scalar("custom/avg_speed", self.locals['infos'][0]['avg_speed'], self.num_timesteps)
            self.writer.add_scalar("custom/mean_reward", self.locals['infos'][0]['mean_reward'], self.num_timesteps)

        return True
    
    def _on_training_end(self) -> None:
        # Close the TensorBoard writer to save all data
        self.writer.close()
