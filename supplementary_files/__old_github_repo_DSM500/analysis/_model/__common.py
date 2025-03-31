import modulus
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_icosahedron(vertices: np.array, faces: np.array = None):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection='3d')

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.scatter(x, y, z, c='r', marker='o', s=1)
    ax.scatter(0, 0, 0, c='black', marker='o') # center
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Determine axis ranges
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    center_x = (x.max() + x.min()) / 2.0
    center_y = (y.max() + y.min()) / 2.0
    center_z = (z.max() + z.min()) / 2.0

    # Add thin X, Y, Z axis lines
    ax.plot([center_x - max_range, center_x + max_range], [center_y, center_y], [center_z, center_z], color='blue', linewidth=0.5, label='X-axis')
    ax.plot([center_x, center_x], [center_y - max_range, center_y + max_range], [center_z, center_z], color='green', linewidth=0.5, label='Y-axis')
    ax.plot([center_x, center_x], [center_y, center_y], [center_z - max_range, center_z + max_range], color='red', linewidth=0.5, label='Z-axis')

    # Add small arrows at the end of each axis
    arrow_length = max_range * 0.1  # Length of the arrow
    ax.quiver(center_x + max_range, center_y, center_z, arrow_length, 0, 0, color='blue', linewidth=1)
    ax.quiver(center_x, center_y + max_range, center_z, 0, arrow_length, 0, color='green', linewidth=1)
    ax.quiver(center_x, center_y, center_z + max_range, 0, 0, arrow_length, color='red', linewidth=1)

    if faces is not None:
        # Plot edges
        for face in faces:
            face = list(face)
            face += [face[0]]
            coords = vertices[face]
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            ax.plot3D(x, y, z, color='gray', linewidth=0.5)

        # Plot sphere
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, edgecolor='none')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_aspect('equal')
    ax.legend()
    plt.show()

from modulus.distributed import DistributedManager
from modulus.utils_new.caching import Cache

DistributedManager.initialize()
Cache.initialize(dir="/iopsstor/scratch/cscs/stefschu/DSM500/cache", verbose=True)

print("__common.py loaded")