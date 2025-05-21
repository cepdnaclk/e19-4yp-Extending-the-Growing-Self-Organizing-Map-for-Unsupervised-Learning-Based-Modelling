import pandas as pd
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import os

# === USER CONFIGURATION ===
dataset_path = "data/shapes/circle_radius100_652_points.csv"  # dataset
output_folder = "outputs"
output_name_prefix = "som_circle"  # different dataset runs

map_width = 20  # Width of SOM grid
map_height = 20  # Height of SOM grid
iterations = 1000  # Training iterations
learning_rate = 0.1  # Initial learning rate
sigma = 3.0  # Initial neighborhood radius
# =============================

# Load & normalize data
df = pd.read_csv(dataset_path)
data_numeric = df.select_dtypes(include=[np.number])
data_normalized = (data_numeric - data_numeric.min()) / (data_numeric.max() - data_numeric.min() + 1e-10)

# Initialize SOM
som = MiniSom(x=map_width, y=map_height, input_len=data_normalized.shape[1],
              sigma=sigma, learning_rate=learning_rate)
som.random_weights_init(data_normalized.to_numpy())

# Train SOM
print(f"Training SOM ({map_width}x{map_height}) on {dataset_path}...")
som.train_random(data_normalized.to_numpy(), num_iteration=iterations)

# Extract coordinates & weights
coords = np.array([[i, j] for i in range(map_width) for j in range(map_height)])
node_coords = pd.DataFrame(coords, columns=["x", "y"])
weights = som.get_weights().reshape(-1, data_normalized.shape[1])
weights_df = pd.DataFrame(weights, columns=[f"w{i+1}" for i in range(weights.shape[1])])

# Save outputs
os.makedirs(output_folder, exist_ok=True)
coord_path = f"{output_folder}/{output_name_prefix}_node_coords.csv"
weights_path = f"{output_folder}/{output_name_prefix}_node_weights.csv"
node_coords.to_csv(coord_path, index=False)
weights_df.to_csv(weights_path, index=False)

print(f"Coordinates saved: {coord_path}")
print(f"Weights saved:     {weights_path}")

# Visualize node map
plt.figure(figsize=(6, 6))
plt.scatter(node_coords["x"], node_coords["y"], c='purple', s=80, label="SOM Nodes")
plt.title(f"SOM Node Map ({map_width}x{map_height})")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
