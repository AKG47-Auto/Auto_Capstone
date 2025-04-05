import mujoco
import mujoco.viewer
import time
import numpy as np

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("car.xml")  # Replace with your model file
data = mujoco.MjData(model)

# Print mass of each body
print("=== Body Masses ===")
for i in range(model.nbody):
    name = model.body(i).name
    mass = model.body_mass[i]
    print(f"Body '{name}': mass = {mass:.4f} kg")

# Print total mass
total_mass = np.sum(model.body_mass)
print(f"Total model mass: {total_mass:.4f} kg\n")

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
