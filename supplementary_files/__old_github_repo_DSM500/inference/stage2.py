import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import json
import hydra
from omegaconf import DictConfig
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from modulus.launch.logging import PythonLogger

from inference import Inference

def plot_on_globe(
    frame_id,
    image_2d,
    vmin,
    vmax,
    edges,
    output_file_path,
    reduction_factor=1.0,
    edge_intensity=1.,
):
    ###### image preprocessing ######
    # Reduce image resolution (to speed up rendering during development)
    new_size = (image_2d.shape[0]//reduction_factor, image_2d.shape[1]//reduction_factor)
    image_2d = resize(image_2d, new_size, anti_aliasing=True)
    edges = resize(edges, new_size, anti_aliasing=True)

    # Make RGB
    image_2d = (image_2d - vmin) / (vmax - vmin)
    cmap = plt.get_cmap("viridis") # alternative: "RdBu_r"
    image_rgb = cmap(image_2d)[..., :3] # Remove the alpha channel

    # Blend in edges
    image_rgb = image_rgb * (1 - edge_intensity * edges[..., None])

    # Flip image upside down
    image_rgb = np.flipud(image_rgb)


    ###### prepare 3D sphere ######
    # Prepare coordinates for the spheric shape
    latitudes, longitudes = image_2d.shape[:2]
    # Spherical coordinates
    phi = np.linspace(-np.pi/2, np.pi/2, latitudes)
    theta = np.linspace(-np.pi, np.pi, longitudes)
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    # Cartesian coordinates
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)


    ###### plot ######
    # Create figure and 3D axis
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=image_rgb, shade=False, antialiased=True)
    
    # Rotate the plot
    ax.view_init(elev=30, azim=-1 * (frame_id*2 % 360))

    # Make background panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False

    ax.grid(False)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    plt.axis('off')
    ax.set_proj_type('persp')
    plt.tight_layout()

    plt.savefig(
        output_file_path, 
        dpi=300, 
        bbox_inches='tight', 
        transparent=True,
        pad_inches=-1
    )

    plt.close(fig)

def load_activities_per_rank(inference):
    with open(inference.plan_file_path, "r") as f:
        plan = json.load(f)
    
    activities = []
    for frame in plan:
        for type, file in frame["files"].items():
            activities.append({
                "type": type,
                "frame_id": frame["frame_id"],
                "filename": inference.images_dir / file,
                "sample": frame["sample"],
                "channel": frame["channel"],
                "step": frame["step"],
                "vmin": frame["vmin"],
                "vmax": frame["vmax"]
            })

    activities_for_this_rank = np.array_split(activities, inference.world_size)[inference.rank]

    return activities_for_this_rank

@hydra.main(version_base="1.3", config_path=".", config_name="inference_config")
def main(cfg: DictConfig):
    inference = Inference(cfg)
    logger = PythonLogger("main")

    edges = inference.load_edges()

    # Load plan and make an activities list
    activities_for_this_rank = load_activities_per_rank(inference)
    logger.info(f"[Rank {inference.rank}] Loaded {len(activities_for_this_rank)} activities for this rank")

    # For each activity, generate the image
    with h5py.File(inference.container_file_path, "r") as f:
        for activity_i, activity in enumerate(activities_for_this_rank):
            # Determine filepath for the image to be rendered
            if activity["filename"].exists():
                logger.info(f"[Rank {inference.rank}] File {activity['filename']} exists. Rendering skipped.")
                continue

            logger.info(f"[Rank {inference.rank}] Processing activity for file {activity['filename']} ...")

            # Extract activity details
            type = activity["type"]
            frame_id = activity["frame_id"]
            filepath = activity["filename"]
            sample = activity["sample"]
            channel = activity["channel"]
            step = activity["step"]
            vmin, vmax = activity["vmin"], activity["vmax"]

            if type == "reanalysis":
                image_2d = f["gc_baseline"]["reanalysis"][sample, step, channel]
            elif type == "gc_baseline":
                image_2d = f["gc_baseline"]["forecast"][sample, step, channel]
            elif type == "gc_improved":
                image_2d = f["gc_improved"]["forecast"][sample, step, channel]
            elif type == "fcn":
                image_2d = f["fcn"]["forecast"][sample, step, channel]
            else:
                raise ValueError(f"Unknown frame type {type}")

            plot_on_globe(
                frame_id=frame_id,
                image_2d=image_2d,
                vmin=vmin,
                vmax=vmax,
                edges=edges,
                reduction_factor=cfg.resolution_reduction_factor,
                output_file_path=filepath
            )

            logger.info(f"[Rank {inference.rank}] File {filepath} created. {len(activities_for_this_rank) - activity_i - 1} activities left.")

        logger.info(f"[Rank {inference.rank}] Done!") 

if __name__ == "__main__":
    main()
