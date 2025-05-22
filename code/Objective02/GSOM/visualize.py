import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

def plot(output, index_col, gsom_map=None, file_name="gsom", file_type=".png",
         figure_label="GSOM Map", max_text=3, max_length=30,
         cmap_colors="Paired", show_index=True, n_nodes=180, show_plot=True):
    """
    Plot GSOM nodes with their clustered data points and POS tree paths.
    """
    if output.empty:
        print("Warning: Output DataFrame is empty. Nothing to plot.")
        return

    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map(max_count, alpha=0.9, cmap_colors=cmap_colors)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw POS skeleton lines between parent-child nodes
    if gsom_map:
        try:
            paths = gsom_map.get_paths()
            for path in paths:
                for i in range(len(path) - 1):
                    x1, y1 = path[i].x, path[i].y
                    x2, y2 = path[i + 1].x, path[i + 1].y
                    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)
        except Exception as e:
            print("Warning: Failed to draw POS paths:", e)

    # Draw clustered nodes with hit_count color and labels
    for _, row in output.iterrows():
        x, y = row["x"], row["y"]
        hit = row["hit_count"]
        color = listed_color_map.colors[min(hit, len(listed_color_map.colors) - 1)]
        ax.plot(x, y, 'o', color=color, markersize=6)

        if show_index:
            label = ", ".join(map(str, row[index_col][:max_text])) if hit > 0 else ""
            txt = ax.text(x, y, label, ha='left', va='center', wrap=True, fontsize=6)
            txt._get_wrap_line_width = lambda: max_length

    ax.set_title(figure_label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Save plot
    plt.savefig(file_name + file_type, bbox_inches='tight')
    print(f"Plot saved to {file_name + file_type}")

    # Display plot if requested
    if show_plot:
        plt.show()


def _get_color_map(max_count, alpha=0.5, cmap_colors="Reds"):
    """
    Create a transparent custom color map with max_count steps
    """
    np.random.seed(1)
    cmap = cm.get_cmap(cmap_colors, max_count + 1)
    color_list = [(r * alpha, g * alpha, b * alpha) for r, g, b, _ in cmap(np.arange(cmap.N))]
    return colors.ListedColormap(color_list, name='gsom_color_list')
