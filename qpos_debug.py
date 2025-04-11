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

# Set full forward throttle (ctrl=1 for max input)
forward_actuator_id = model.actuator("forward").id
data.ctrl[forward_actuator_id] = 1.0  # full throttle

# Simulation parameters
dt = model.opt.timestep
sim_time = 10.0  # seconds
steps = int(sim_time / dt)
acc_samples = []

# Viewer for visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    for step in range(steps):
        # Store previous linear velocity (world frame)
        v_prev = np.copy(data.qvel[0:3])
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Compute acceleration: (v_new - v_prev) / dt
        v_new = data.qvel[0:3]
        acc = (v_new - v_prev) / dt
        acc_samples.append(acc)

        # Optionally print every 1 sec
        if step % int(1 / dt) == 0:
            print(f"Time {data.time:.2f}s | Linear Acceleration (x,y,z): {acc}")

        if viewer.is_running():
            viewer.sync()

# After simulation
avg_acc = np.mean(acc_samples, axis=0)
print("\n=== Final State ===")
print("qpos:", data.qpos)
print("qvel:", data.qvel)
print(f"\nAverage linear acceleration over {sim_time:.1f} sec: {avg_acc} m/s²")
