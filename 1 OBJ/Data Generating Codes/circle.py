import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
radius = 100
spacing = 7.15  # spacing between points
y_values = np.arange(-radius, radius + spacing, spacing)
points = []

# Generate horizontal lines of points within a circle
for y in y_values:
    value = radius**2 - y**2
    if value >= 0:  # Ensure the value is non-negative
        x_limit = int(value**0.5)
        x_values = np.arange(-x_limit, x_limit + spacing, spacing)
        for x in x_values:
            points.append((x, y))

points = np.array(points)

# Save to CSV
df = pd.DataFrame(points, columns=['x', 'y'])
df.to_csv('circle_points_652.csv', index=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=20)
plt.title(f"Circle (n={len(points)})", fontsize=16, fontweight='bold')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-radius - 10, radius + 10)
plt.ylim(-radius - 10, radius + 10)
plt.xticks(np.arange(-radius, radius + 1, 25))
plt.yticks(np.arange(-radius, radius + 1, 25))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()