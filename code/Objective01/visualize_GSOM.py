import pandas as pd
import matplotlib.pyplot as plt
import os

# ✅ Step 1: Load saved GSOM node coordinates
node_file = "outputs/gsom_circle_node_coordinates.csv"
if not os.path.exists(node_file):
    raise FileNotFoundError(f"File not found: {node_file}")

nodes = pd.read_csv(node_file)

# ✅ Step 2: Plot the nodes
plt.figure(figsize=(6, 6))
plt.scatter(nodes["x"], nodes["y"], c='blue', s=100, label="GSOM Nodes")

# Optional: Add node indices
for i, row in nodes.iterrows():
    plt.text(row["x"], row["y"], str(i), ha='center', va='center', fontsize=8, color='white')

# ✅ Step 3: Style the plot
plt.title("GSOM Node Map: Circle Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.gca().invert_yaxis()  # To match GSOM growth pattern
plt.tight_layout()
plt.legend()
plt.show()
