import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

cam_points = np.array([
    [-41.15661, 17.338966, 762.5427],
    [145.79343, 29.70976, 755.2696],
    [154.19814, -77.062256, 789.6985],
    [-32.7519, -89.433044, 796.9716],
    [-44.46566,   37.00211,  709.6775  ]
])

world_points = np.array([
    [0.0, 0.0, 0.0],
    [187.5, 0.0, 0.0],
    [187.5, 112.5, 0.0],
    [0.0, 112.5, 0.0],
    [0.0, 0.0, 50.0]
])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors for each pair of points
colors = ['r', 'g', 'b', 'c', 'm']

# Plot points
for i in range(len(cam_points)):
    ax.scatter(*world_points[i], color=colors[i], label=f'World {i}' if i == 0 else "")
    ax.scatter(*cam_points[i], color=colors[i], marker='^', label=f'Camera {i}' if i == 0 else "")
    ax.plot([world_points[i, 0], cam_points[i, 0]],
            [world_points[i, 1], cam_points[i, 1]],
            [world_points[i, 2], cam_points[i, 2]],
            color=colors[i], linestyle='dashed')


# Create polygons to form opaque surfaces
faces = [
    [world_points[0], world_points[1], world_points[2], world_points[3]],  # Base
    [cam_points[0], cam_points[1], cam_points[2], cam_points[3]],  # Top
]

for face in faces:
    poly = art3d.Poly3DCollection([face], alpha=0.5, color='gray')
    ax.add_collection3d(poly)

# Labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of World and Camera Points")
ax.legend()

# Show the plot
plt.show()