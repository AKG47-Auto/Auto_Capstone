import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'state_history_episode.csv'  # Update this with the correct path if needed
data = pd.read_csv(file_path)

# Plot y vs x
plt.figure(figsize=(8, 6))
plt.plot(data['Vehicle X'], data['Vehicle Y'], label='Trajectory', marker='o', linestyle='-')
plt.xlabel('X Position(m)')
plt.ylabel('Y Position(m)')
plt.title('Y vs X Trajectory')
plt.ylim(-0.0002, 0.0002)
plt.legend()
plt.grid(True)
plt.show()
