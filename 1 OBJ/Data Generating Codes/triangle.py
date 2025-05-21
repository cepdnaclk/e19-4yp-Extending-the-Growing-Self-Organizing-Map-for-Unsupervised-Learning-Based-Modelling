import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
max_height = 22
spacing = 1  # spacing between points
points = []

# Generate triangle points
for y in range(max_height + 1):
    half_width = y
    x_values = np.arange(-half_width, half_width + 1, spacing)
    for x in x_values:
        points.append((x, y))

# Keep only the first 484 points
points = points[:484]
points = np.array(points)

# Save to CSV
df = pd.DataFrame(points, columns=['x', 'y'])
df.to_csv('triangle_points_484.csv', index=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=15)
plt.title(f"Triangle (n={len(points)})", fontsize=16, fontweight='bold')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
