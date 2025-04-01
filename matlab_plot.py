import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define key transition points for smooth fillet-like curve
x_control = np.array([0, 30, 60, 140, 180, 200])  # X-coordinates
y_control = np.array([0, 0, 0, 50, 8, 5])  # Y-coordinates

# Create a cubic spline interpolation for smooth transition
spline = CubicSpline(x_control, y_control, bc_type='natural')

# Generate more points for a smoother curve
x_smooth = np.linspace(0, 200, 1000)  # More points for smoothness
y_smooth = spline(x_smooth)

# Compute heading as the slope of the curve at each x coordinate
dy_dx = spline.derivative()(x_smooth)
heading_smooth = np.arctan2(dy_dx, np.ones_like(dy_dx))

# Plot the improved smooth path
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, 'bo-', label="Smooth Path Waypoints")
plt.quiver(x_smooth, y_smooth, np.cos(heading_smooth), np.sin(heading_smooth), scale=10, color='r', label="Heading")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("Improved Smooth Path with Fillet-like Transition")
plt.legend()
plt.grid()
plt.show()

# Plot heading over x-smooth to verify correctness
plt.figure(figsize=(10, 4))
plt.plot(x_smooth, np.degrees(heading_smooth), label="Heading (degrees)", color='g')
plt.xlabel("X Coordinate (m)")
plt.ylabel("Heading (degrees)")
plt.title("Heading Along the Path")
plt.legend()
plt.grid()
plt.show()

# Store waypoints as (x, y, heading)
waypoints_smooth = np.column_stack((x_smooth, y_smooth, heading_smooth))

# Save waypoints to a file
np.savetxt("new_curve_waypoints.csv", waypoints_smooth, delimiter=",", header="x,y,theta", comments='')

# Display first few waypoints
print(waypoints_smooth[:5])
