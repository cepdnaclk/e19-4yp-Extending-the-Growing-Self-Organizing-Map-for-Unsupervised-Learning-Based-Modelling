import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Zoo dataset
df = pd.read_csv("zoo.txt")

# Map numeric labels to class names
class_names = {
    1: 'Mammal',
    2: 'Bird',
    3: 'Reptile',
    4: 'Fish',
    5: 'Amphibian',
    6: 'Insect',
    7: 'Invertebrate'
}

# Add class name column
df['class_name'] = df['label'].map(class_names)

# Add jitter to avoid overlapping points
np.random.seed(0)
df['w6_jitter'] = df['w6'] + np.random.normal(0, 0.05, size=len(df))
df['w13_jitter'] = df['w13'] + np.random.normal(0, 0.2, size=len(df))

# Plot
plt.figure(figsize=(10, 6))
for label, name in class_names.items():
    subset = df[df['label'] == label]
    plt.scatter(subset['w6_jitter'], subset['w13_jitter'], s=100, alpha=0.7, edgecolors='k', label=name)

# Labels and formatting
plt.xlabel('w6 (e.g., Airborne) + Jitter')
plt.ylabel('w13 (Number of Legs) + Jitter')
plt.title('Zoo Dataset Scatter Plot with Jitter (Colored by Animal Class)')
plt.legend(title='Animal Class')
plt.grid(True)

# Save figure
plt.savefig("zoo_scatter_plot_jittered.png", dpi=300, bbox_inches='tight')
plt.show()
