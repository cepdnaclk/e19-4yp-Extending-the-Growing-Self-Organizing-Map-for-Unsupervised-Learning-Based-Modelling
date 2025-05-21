import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
min_x, max_x = -10, 10  # x-axis range
min_y, max_y = 0, 20  # y-axis range
n_points_total = 441
n_rows = 21  # Since sqrt(441) = 21, a 21x21 grid
spacing = (max_x - min_x) / (n_rows - 1)  # Equal spacing for x and y to form a square grid
points = []

# Generate square points (grid pattern)
for y in np.linspace(min_y, max_y, n_rows):
    for x in np.linspace(min_x, max_x, n_rows):
        points.append((x, y))

# Ensure exactly 441 points
points = points[:n_points_total]
points = np.array(points)

# Save to CSV
df = pd.DataFrame(points, columns=['x', 'y'])
df.to_csv('square_points_441.csv', index=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=1, c='blue')
plt.title(f"Square (n={len(points)})", fontsize=16, fontweight='bold')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.xlim(-10, 10)
plt.ylim(0, 20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()