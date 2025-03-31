import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import numpy as np
import json
import h5py
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from datetime import datetime

import numpy as np

from modulus.launch.logging import PythonLogger

# FIXIT: Move it under modulus
from inference import Inference

class VideoFrameTwoGlobes:
    def __init__(
        self,
        dpi=200,
        crop=(105, 60, 60, 115) # l, r, t, b
    ):
        self.crop = crop
        self.dpi = dpi

        self.fig = plt.figure(figsize=(9, 8), dpi=self.dpi)
        self.fig.patch.set_edgecolor('black')
        self.fig.patch.set_linewidth(1)

        gs = GridSpec(
            4, 4,
            height_ratios=[0.2, 1, 0.8, 0.01],
            width_ratios=[0.94, 0.94, 0.05, 0.3],
            hspace=0.4, wspace=0.3
        )

        self.ax_title = self.fig.add_subplot(gs[0, :])
        self.ax_title.axis('off')

        self.ax_img_left = self.fig.add_subplot(gs[1, 0])
        self.ax_img_right = self.fig.add_subplot(gs[1, 1])
        self.ax_cbar = self.fig.add_subplot(gs[1, 2])

        self.ax_chart = self.fig.add_subplot(gs[2, :])

        self.ax_footer = self.fig.add_subplot(gs[3, :])
        self.ax_footer.axis('off')

    def crop_image(self, image):
        l, r, t, b = self.crop
        h, w, _ = image.shape
        return image[t:h-b, l:w-r]

    def title(self, title, subtitle):
        self.ax_title.text(0.5, 0.8, 
            title,
            fontsize=16, 
            fontweight='bold',
            ha='center',
            transform=self.ax_title.transAxes
        )
        self.ax_title.text(0.5, 0.2, 
            subtitle,
            fontsize=12, 
            ha='center',
            transform=self.ax_title.transAxes
        )
        return self

    def footer(self, text):
        self.ax_footer.text(1, 0.0, 
            text,
            fontsize=6,
            ha='right',
            va='bottom',
            transform=self.ax_footer.transAxes,
            position=(1.09, 0)
        )
        return self

    def _img(self, ax, title, file_path):
        img = mpimg.imread(file_path)
        img = self.crop_image(img)

        ax.set_title(title, fontsize=8)
        ax.imshow(img, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        return self

    def img_left(self, title, file_path):
        self._img(self.ax_img_left, title, file_path)
        return self

    def img_right(self, title, file_path):
        self._img(self.ax_img_right, title, file_path)
        return self

    def cbar(self, vmin, vmax, ticks_format):
        def formatter(x, _):
            number, unit = ticks_format.split(" ")
            string = number.format(x).replace(",", "'")
            string = f"${string} \\ \\mathrm{{{unit}}}$"
            if x >= 0:
                string = "  " + string
            return string
        cbar = self.fig.colorbar(cm.ScalarMappable(
            norm=plt.Normalize(vmin, vmax),
            cmap=plt.get_cmap("viridis")
        ), cax=self.ax_cbar)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        return self

    def chart(self, x, y_left, y_right, x_label, y_left_label, y_right_label):
        self.ax_chart.plot(x, y_left, color='blue')
        self.ax_chart.set_xlabel(x_label)
        self.ax_chart.set_ylabel(y_left_label, color='blue')
        self.ax_chart.tick_params(axis='y', labelcolor='blue')
        self.ax_chart.grid(True)

        ax_right = self.ax_chart.twinx()
        ax_right.plot(x, y_right, color='red')
        ax_right.set_ylabel(y_right_label, color='red')
        ax_right.tick_params(axis='y', labelcolor='red')
        return self

    def savefig(self, output_file_path):
        plt.savefig(output_file_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.3)

    def show(self):
        plt.show()
        plt.close(self.fig)




@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()

    inference = Inference(cfg, logger)

    # Load plan, and select activities for this rank
    with open(inference.plan_file_path, "r") as f:
        plan = json.load(f)
    # Filter out pause frames which do not need to be rendered
    activities = [a for a in plan if a["type"] != "pause"]
    activities_for_this_rank = np.array_split(activities, inference.world_size)[inference.rank]
    logger.info(f"[Rank {inference.rank}] Loaded {len(activities_for_this_rank)} activities for this rank")

    # For each activity, generate the image
    with h5py.File(inference.container_file_path, "r") as f:
        for activity_i, activity in enumerate(activities_for_this_rank):
            output_file_path = inference.images_dir / f"frame_{activity['frame_id']}.png"

            if output_file_path.exists():
                logger.info(f"[Rank {inference.rank}] File {output_file_path} already exists. Skipping.")
                continue
            else:
                logger.info(f"[Rank {inference.rank}] Generating {output_file_path}...")

            if activity["type"] == "ic":
                input_file_path_left = input_file_path_right = inference.images_dir / activity["filename"]
                title_left = title_right = "Initial Condition (time 0)"
            else:
                input_file_path_left = inference.images_dir / activity["filename"]["reanalysis"]
                input_file_path_right = inference.images_dir / activity["filename"]["forecast"]
                step = activity["step"]
                title_left = f"Reanalysis (time +{step+1})"
                title_right = f"Forecast (time +{step+1})"

            sample = activity["sample"]
            timestamp = datetime.fromtimestamp(activity["timestamp"])
            time_string = timestamp.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z UTC")

            title = inference.get_channel_title(activity["channel"])
            subtitle = f"Sample {sample} of {cfg.inference.samples} | Time: {time_string}"

            x = np.linspace(0, 10, 100)
            y_sin = np.sin(x)
            y_cos = np.cos(x)

            VideoFrameTwoGlobes()\
                .title(title, subtitle)\
                .img_left(title_left, input_file_path_left)\
                .img_right(title_right, input_file_path_right)\
                .cbar(activity["vmin"], activity["vmax"], inference.get_channel_tick_format(activity["channel"]))\
                .chart(x, y_sin, y_cos, "X-axis", "Y-axis", "Cosine Wave (Right)")\
                .footer("Stefano Schuppli - DSM500")\
                .savefig(output_file_path)

            logger.info(f"[Rank {inference.rank}] File {output_file_path} created. {len(activities_for_this_rank) - activity_i - 1} activities left.")
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