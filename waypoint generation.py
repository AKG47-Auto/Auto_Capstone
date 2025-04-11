import numpy as np
import matplotlib.pyplot as plt

# Define parameters
length = 10  # total length of the S-curve
num_waypoints = 1000  # number of waypoints
x = np.linspace(0, length, num_waypoints)  # x coordinates from 0 to 200

# Define y coordinates using a sine wave for the "S" shape
# You can adjust the amplitude and frequency to get your desired shape
amplitude = 0.5  # maximum height of the S curve
frequency = 1  # frequency of the sine wave
# Sigmoid function to smoothly transition between straight and sine curve
transition_length = length / 4  # length over which to transition
transition = 1 / (1 + np.exp(-10 * (x - transition_length) / length))
y_linear = np.zeros_like(x)
y_sine = amplitude * np.sin(frequency * x)
y = (1 - transition) * y_linear + transition * y_sine # y coordinates
#y = np.zeros_like(x)
dy = np.gradient(y, x)  # derivative of y with respect to x
headings = np.arctan(dy)

# Combine x and y into waypoints
waypoints = np.column_stack((x, y, headings))
plt.plot(waypoints[:, 0], waypoints[:, 1], label='straight-line Path')
plt.scatter(waypoints[:, 0], waypoints[:, 1], color='red')  # waypoints
plt.title('Straight-line Waypoints')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
np.savetxt("curve_line_waypoints.csv", waypoints, delimiter=",", header="x,y,theta", comments="")