import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd


class VehicleController:
    def __init__(self):
        # Generate random waypoints
        np.random.seed(42)  # For reproducibility
        #self.waypoints = np.cumsum(np.random.rand(20, 2) * 5, axis=0)
        waypoint_file="./s_shaped_waypoints.csv"
        waypoints_df = pd.read_csv(waypoint_file)
        self.waypoints = waypoints_df[['x', 'y']].to_numpy()

        self.state = [170.0, 50, 2.0, 0.5]  # x, y, vx, vy
        self.current_waypoint_index = 0

    def create_curve_from_waypoints(self):
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        
        if len(x) < 2:
            raise ValueError("At least two waypoints are required to create a curve.")
        
        x_new = np.linspace(min(x), max(x), 300)
        spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
        y_smooth = spline(x_new)
        
        self.spline = spline
        self.x_new = x_new
        self.y_smooth = y_smooth
        return x_new, y_smooth, spline
    
    def shortest_distance_to_spline(self, point):
        x_point, y_point = point
        distances = []
        
        for x_spline, y_spline in zip(self.x_new, self.y_smooth):
            dist = np.sqrt((x_spline - x_point)**2 + (y_spline - y_point)**2)
            distances.append(dist)
        
        min_index = np.argmin(distances)
        closest_point = (self.x_new[min_index], self.y_smooth[min_index])
        spline_derivative = self.spline.derivative()
        heading = spline_derivative(self.x_new[min_index])
        looahead_index = min_index + 5
        if looahead_index < len(self.x_new):
            lookahead_point = (self.x_new[looahead_index], self.y_smooth[looahead_index])
            lookahead_heading = spline_derivative(self.x_new[min_index + 15])
        else:
            lookahead_point = None
            lookahead_heading = None
        lookahead_distance = np.sqrt((lookahead_point[0] - x_point)**2 + (lookahead_point[1] - y_point)**2)
        return distances[min_index], closest_point, heading , lookahead_point, lookahead_heading , lookahead_distance
    
    def calculate_spline_length(self):
        """
        Calculate the length of the spline curve by integrating the distance between points along the curve.
        
        Returns:
            length: Total length of the spline curve
        """
        # Generate points along the spline
        x_points = self.x_new
        y_points = self.y_smooth
        
        # Calculate the differences between consecutive points
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        
        # Calculate the Euclidean distance between each consecutive pair of points
        distances = np.sqrt(dx**2 + dy**2)
        
        # Sum up the distances to get the total length
        total_length = np.sum(distances)
        
        return total_length , distances

# Example usage:
# Assuming 'controller' is an instance of the class containing the method


controller = VehicleController()

# Create spline from waypoints
controller.create_curve_from_waypoints()
spline_length, distances = controller.calculate_spline_length()
print(f"Spline length: {spline_length}")

# Get lookahead waypoint
# lookahead = controller.get_lookahead_waypoint()


# Find shortest distance to spline from a random point
test_point = (controller.state[0], controller.state[1])
distance, closest_point, heading, lookahead, target_heading, lookahead_distance = controller.shortest_distance_to_spline(test_point)
e_cross_track = np.sqrt((closest_point[0] - controller.state[0])**2 + (closest_point[1] - controller.state[1])**2)
print("lookahead distance", lookahead_distance)

print("error", e_cross_track)
print("Lookahead Waypoint:", lookahead)
print("Lookahead Heading:", target_heading)
print("Shortest Distance to Spline:", distance)
print("Closest Point on Spline:", closest_point)
print("Heading at Closest Point:", heading)
head = np.arctan(5 * e_cross_track / (5))
print("arctan:", head)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(controller.waypoints[:, 0], controller.waypoints[:, 1], 'o--', label='Original Waypoints')
plt.plot(controller.x_new, controller.y_smooth, '-', label='Spline Path')
if lookahead is not None:
    plt.plot(*lookahead, 'ro', label='Lookahead Waypoint')
else:
    print("No lookahead available:")
plt.plot(*test_point, 'go', label='Test Point')
plt.plot(*closest_point, 'mo', label='Closest Point on Spline')
plt.legend()
plt.title('Path Following and Lookahead Waypoint')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
#plt.plot(controller.x_new, distances, label="Distance to Spline")
plt.axvline(x=closest_point[0], color='r', linestyle='--', label="Closest Point")
plt.title("Distance to Spline vs X-coordinate")
plt.xlabel("X-coordinate of Spline")
plt.ylabel("Distance to Spline")
plt.legend()
plt.grid()
plt.show()