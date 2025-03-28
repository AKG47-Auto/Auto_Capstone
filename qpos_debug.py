import mujoco
import mujoco.viewer
import time
import numpy as np

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("rover_scaled.xml")  # Replace with your model file
data = mujoco.MjData(model)

# Create a viewer (optional, for visualization)
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    # Simulate for 5 seconds
    while data.time < 600:
        mujoco.mj_step(model, data)
        
        # If using a viewer, update the visualization
        if viewer.is_running():
            viewer.sync()

# Print the qpos values after 5 seconds
print("qpos after 600 seconds:", data.qpos)
print("qvel after 600 seconds:", data.qvel)
