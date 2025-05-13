import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
min_x, max_x = -30, 30  # x-axis range
min_y, max_y = 0, 20  # y-axis range
n_points_total = 861
n_rows = 30  # Number of rows to distribute points
spacing_y = (max_y - min_y) / (n_rows - 1)  # Vertical spacing between rows
points = []

# Generate trapezoid points (horizontal lines with decreasing width)
for i, y in enumerate(np.linspace(min_y, max_y, n_rows)):
    # Calculate the half-width at this y-level (linearly decreasing from 30 to 0)
    half_width = 30 * (1 - y / max_y)
    # Number of points in this row (roughly proportional to width)
    n_points_row = int(45 * (1 - y / max_y)) + 1  # Adjust to approximate total points
    if n_points_row < 1:
        n_points_row = 1
    x_values = np.linspace(-half_width, half_width, n_points_row)
    for x in x_values:
        points.append((x, y))

# Trim or pad to exactly 861 points
points = points[:n_points_total]
points = np.array(points)
if len(points) < n_points_total:
    while len(points) < n_points_total:
        idx = np.random.randint(0, len(points))
        points = np.vstack((points, points[idx]))

# Save to CSV
df = pd.DataFrame(points, columns=['x', 'y'])
df.to_csv('trapeze_points_861.csv', index=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=1, c='blue')
plt.title(f"Trapeze (n={len(points)})", fontsize=16, fontweight='bold')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.xlim(-30, 30)
plt.ylim(0, 20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()