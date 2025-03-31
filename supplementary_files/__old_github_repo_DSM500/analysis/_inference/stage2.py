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

# FIXIT: Move it under modulus
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
    ax.view_init(elev=30, azim=-1 * (frame_id % 360))

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
        if frame["type"] == "pause":
            continue
        elif frame["type"] == "ic":
            activities.append(frame)
        elif frame["type"] == "inference":
            activities.append(frame | {
                "type": "reanalysis",
                "filename": frame["filename"]["reanalysis"]
            })
            activities.append(frame | {
                "type": "forecast",
                "filename": frame["filename"]["forecast"]
            })
        else:
            raise ValueError(f"Unknown frame type {frame['type']}")
    activities_for_this_rank = np.array_split(activities, inference.world_size)[inference.rank]
    return activities_for_this_rank
    

@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()

    inference = Inference(cfg, logger)
    edges = inference.load_edges()
    
    # Load plan and make an activities list
    activities_for_this_rank = load_activities_per_rank(inference)
    logger.info(f"[Rank {inference.rank}] Loaded {len(activities_for_this_rank)} activities for this rank")

    # For each activity, generate the image
    with h5py.File(inference.container_file_path, "r") as f:
        for activity_i, activity in enumerate(activities_for_this_rank):
            # Determine filepath for the image to be rendered
            filepath = inference.images_dir / activity["filename"]
            if filepath.exists():
                logger.info(f"[Rank {inference.rank}] File {filepath} exists. Rendering skipped.")
                continue
            else:
                logger.info(f"[Rank {inference.rank}] Processing activity for file {filepath} ...")

            # Extract activity details
            type = activity["type"]
            sample = activity["sample"]
            channel = activity["channel"]
            frame_id = activity["frame_id"]
            vmin, vmax = activity["vmin"], activity["vmax"]

            if type == "ic":
                image_2d = f["ic"][sample][0][channel]
            else:
                step = activity["step"]
                image_2d = f[type][sample][step][channel]
            
            plot_on_globe(
                frame_id=frame_id,
                image_2d=image_2d,
                vmin=vmin,
                vmax=vmax,
                edges=edges,
                reduction_factor=cfg.inference.resolution_reduction_factor,
                output_file_path=filepath
            )

            logger.info(f"[Rank {inference.rank}] File {filepath} created. {len(activities_for_this_rank) - activity_i - 1} activities left.")

    logger.info(f"[Rank {inference.rank}] Done!") 

if __name__ == "__main__":
    main()

    # Supress [rank0]:[W303 12:53:55.312091090 ProcessGroupNCCL.cpp:1262] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    # Suppress Exception ignored in atexit callback: <function dump_compile_times at 0x400186acc1f0>
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    import os
    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    import atexit
    import torch._dynamo.utils
    atexit.unregister(torch._dynamo.utils.dump_compile_times)