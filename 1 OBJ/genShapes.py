import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate points inside a shape
def generate_circle_points(center, radius, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, num_points))  # Uniform distribution inside circle
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return x, y

def generate_triangle_points(vertices, num_points):
    # Use barycentric coordinates for uniform sampling inside triangle
    v0, v1, v2 = vertices
    r1 = np.random.uniform(0, 1, num_points)
    r2 = np.random.uniform(0, 1, num_points)
    # Ensure points are inside the triangle
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    x = v0[0] * (1 - r1 - r2) + v1[0] * r1 + v2[0] * r2
    y = v0[1] * (1 - r1 - r2) + v1[1] * r1 + v2[1] * r2
    return x, y

def generate_rectangle_points(bottom, width, height, num_points):
    # Bottom-left corner is 'bottom'
    x = np.random.uniform(bottom[0], bottom[0] + width, num_points)
    y = np.random.uniform(bottom[1], bottom[1] + height, num_points)
    return x, y

def generate_square_points(bottom, side, num_points):
    return generate_rectangle_points(bottom, side, side, num_points)

def generate_trapeze_points(bottom, top_width, bottom_width, height, num_points):
    # Trapeze with bottom-left at 'bottom'
    x_bottom = np.random.uniform(bottom[0], bottom[0] + bottom_width, num_points)
    t = np.random.uniform(0, 1, num_points)  # Interpolation factor for height
    x_width = bottom_width + (top_width - bottom_width) * t  # Linearly interpolate width
    x_offset = (bottom_width - x_width) / 2  # Center the top
    x = bottom[0] + x_offset + np.random.uniform(0, x_width, num_points)
    y = bottom[1] + t * height
    return x, y

def generate_hexagon_points(center, radius, num_points):
    # Approximate hexagon as a circle and filter points to fit hexagon boundaries
    x, y = generate_circle_points(center, radius, num_points * 2)  # Generate extra points
    # Hexagon vertices
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6  # Rotate by 30 degrees
    hex_x = center[0] + radius * np.cos(angles)
    hex_y = center[1] + radius * np.sin(angles)
    # Check if points are inside the hexagon using a point-in-polygon algorithm
    inside = []
    for i in range(len(x)):
        j = 5
        inside_flag = False
        for k in range(6):
            if ((hex_y[k] > y[i]) != (hex_y[j] > y[i])) and \
               (x[i] < (hex_x[j] - hex_x[k]) * (y[i] - hex_y[k]) / (hex_y[j] - hex_y[k]) + hex_x[k]):
                inside_flag = not inside_flag
            j = k
        if inside_flag:
            inside.append((x[i], y[i]))
        if len(inside) >= num_points:
            break
    x_inside, y_inside = zip(*inside[:num_points])
    return np.array(x_inside), np.array(y_inside)

# Updated shapes dictionary with separate file saving
shapes = {
    "circle": {"center": (10, 10), "radius": 5, "num_points": 452},
    "triangle": {"vertices": [(20, 5), (15, 15), (25, 15)], "num_points": 484},
    "rectangle": {"bottom": (30, 5), "width": 8, "height": 4, "num_points": 861},
    "square": {"bottom": (5, 25), "side": 5, "num_points": 441},
    "trapeze": {"bottom": (15, 25), "top_width": 4, "bottom_width": 8, "height": 5, "num_points": 861},
    "hexagon": {"center": (30, 25), "radius": 4, "num_points": 431}
}

# Generate points for each shape and save to separate files
for shape, params in shapes.items():
    if shape == "circle":
        x, y = generate_circle_points(params["center"], params["radius"], params["num_points"])
    elif shape == "triangle":
        x, y = generate_triangle_points(params["vertices"], params["num_points"])
    elif shape == "rectangle":
        x, y = generate_rectangle_points(params["bottom"], params["width"], params["height"], params["num_points"])
    elif shape == "square":
        x, y = generate_square_points(params["bottom"], params["side"], params["num_points"])
    elif shape == "trapeze":
        x, y = generate_trapeze_points(params["bottom"], params["top_width"], params["bottom_width"], params["height"], params["num_points"])
    elif shape == "hexagon":
        x, y = generate_hexagon_points(params["center"], params["radius"], params["num_points"])
    
    # Create a DataFrame for the current shape
    data = [[x[i], y[i], shape] for i in range(len(x))]
    df = pd.DataFrame(data, columns=["x", "y", "shape"])
    
    # Save to a separate CSV file for each shape
    df.to_csv(f"{shape}_dataset.csv", index=False)

print("Datasets generated and saved as separate files: circle_dataset.csv, triangle_dataset.csv, rectangle_dataset.csv, square_dataset.csv, trapeze_dataset.csv, hexagon_dataset.csv")