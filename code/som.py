import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm

# Load Zoo dataset
df = pd.read_csv("zoo.txt")

# Extract features (drop Name and label)
X = df.drop(columns=["Name", "label"])
X.columns = [f"f{i}" for i in range(X.shape[1])]  # Ensure unique column names
X_scaled = MinMaxScaler().fit_transform(X.values)

# Initialize SOM (5x5 grid)
som_x, som_y = 5, 5
som = MiniSom(som_x, som_y, X_scaled.shape[1], sigma=0.5, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train(X_scaled, num_iteration=1000)

# Assign colors to class labels
unique_labels = sorted(df['label'].unique())
color_map = cm.get_cmap('tab10', len(unique_labels))  # Use compatible syntax
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

# Plot SOM Map
plt.figure(figsize=(10, 10))
plt.title("SOM 5x5 – One Dot per Animal (Color by Class)")

for i, x in enumerate(X_scaled):
    x_winner, y_winner = som.winner(x)
    label = df['label'].iloc[i]
    animal = df['Name'].iloc[i]
    color = label_to_color[label]

    # Plot dot and label
    plt.plot(x_winner, y_winner, 'o', color=color, markersize=6)
    plt.text(x_winner + 0.1, y_winner, animal, fontsize=8, color=color)

# Add legend
for label, color in label_to_color.items():
    plt.plot([], [], 'o', color=color, label=f"Class {label}")
plt.legend(title="Animal Class", loc='upper right')

# Grid setup
plt.xticks(np.arange(som_x))
plt.yticks(np.arange(som_y))
plt.grid(True)
plt.gca().invert_yaxis()  # Matches GSOM map layout
plt.xlabel("SOM X")
plt.ylabel("SOM Y")
plt.tight_layout()

# Save the figure
plt.savefig("som_5x5_animals_colored.png", dpi=300)
plt.show()
