import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import make_interp_spline

def create_curve_from_csv(csv_file):
    # Read the CSV file and extract x, y, and heading
    x = []
    y = []
    heading = []
    
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if there is one
        for row in reader:
            x.append(float(row[0]))  # Assuming x is in the first column
            y.append(float(row[1]))  # Assuming y is in the second column
            heading.append(float(row[2]))  # Assuming heading is in the third column
    
    if len(x) < 2:
        raise ValueError("At least two waypoints are required to create a curve.")
    
    # Create a dense range of x-values for smoothness
    x_new = np.linspace(min(x), max(x), 300)
    
    # Create a cubic spline curve
    spline = make_interp_spline(x, y, k=min(3, len(x) - 1))  # k=3 for cubic, adjust if fewer points
    y_smooth = spline(x_new)
    
    # Plot the original waypoints and the smooth curve
    plt.plot(x, y, 'o', label='Waypoints')
    plt.plot(x_new, y_smooth, label='Smooth Curve')
    plt.title('Generated Curve from Waypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
csv_file = 'curve_line_waypoints.csv'  # Replace with your CSV file path
create_curve_from_csv(csv_file)
