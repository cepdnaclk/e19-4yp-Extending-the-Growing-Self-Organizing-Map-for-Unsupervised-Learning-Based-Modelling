import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
max_width = 20  # x ranges from -20 to 20
max_height = 20  # y ranges from 0 to 20
spacing = 1  # spacing between points
points = []

# Generate rectangle points (horizontal lines forming a rectangle)
for y in np.arange(0, max_height + spacing, spacing):
    x_values = np.arange(-max_width, max_width + spacing, spacing)
    for x in x_values:
        points.append((x, y))

# Keep only the first 861 points

points = np.array(points)

# Save to CSV
df = pd.DataFrame(points, columns=['x', 'y'])
df.to_csv('rectangle_points_861.csv', index=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=1, c='blue')
plt.title(f"Rectangle (n={len(points)})", fontsize=16, fontweight='bold')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.xlim(-20, 20)
plt.ylim(0, 17.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()