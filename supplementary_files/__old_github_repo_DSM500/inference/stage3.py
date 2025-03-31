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

from inference import Inference

class VideoFrameFourGlobes:
    def __init__(
        self,
        dpi=200,
        crop=(105, 60, 60, 115) # l, r, t, b
    ):
        self.crop = crop
        self.dpi = dpi

        self.fig = plt.figure(figsize=(11, 10), dpi=self.dpi)
        self.fig.patch.set_edgecolor('black')
        self.fig.patch.set_linewidth(1)

        gs = GridSpec(
            6, 8,
            height_ratios=[.4, 1.5, 1., .1, 1., .4],
            width_ratios=[.2, .90, .90, .90, .90, 0.01, .07, .2],
            hspace=0.20, wspace=0.1
        )

        self.ax_title = self.fig.add_subplot(gs[0, :])
        self.ax_title.axis('off')

        self.ax_img_reanalysis = self.fig.add_subplot(gs[1, 1])
        self.ax_img_gc_baseline = self.fig.add_subplot(gs[1, 2])
        self.ax_img_gc_improved = self.fig.add_subplot(gs[1, 3])
        self.ax_img_fcn = self.fig.add_subplot(gs[1, 4])
        self.ax_cbar = self.fig.add_subplot(gs[1, 6])

        self.ax_chart_mse = self.fig.add_subplot(gs[2, 1:])
        self.ax_chart_acc = self.fig.add_subplot(gs[4, 1:])

        self.ax_footer = self.fig.add_subplot(gs[5, :])
        self.ax_footer.axis('off')

    def crop_image(self, image):
        l, r, t, b = self.crop
        h, w, _ = image.shape
        return image[t:h-b, l:w-r]

    def title(self, title, subtitle):
        self.ax_title.text(0.5, 0.7, 
            title,
            fontsize=16, 
            fontweight='bold',
            ha='center',
            transform=self.ax_title.transAxes
        )
        self.ax_title.text(0.5, 0.15, 
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
            position=(1.09, -0.2)
        )
        return self

    def _img(self, ax, title, file_path):
        img = mpimg.imread(file_path)
        img = self.crop_image(img)

        ax.set_title(title, fontsize=10)
        ax.imshow(img, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        return self

    def img_reanalysis(self, title, file_path):
        self._img(self.ax_img_reanalysis, title, file_path)
        return self

    def img_gc_baseline(self, title, file_path):
        self._img(self.ax_img_gc_baseline, title, file_path)
        return self

    def img_gc_improved(self, title, file_path):
        self._img(self.ax_img_gc_improved, title, file_path)
        return self

    def img_fcn(self, title, file_path):
        self._img(self.ax_img_fcn, title, file_path)
        return self
    
    def cbar(self, vmin, vmax, ticks_format):
        def formatter(x, _):
            number, unit = ticks_format.split(" ")
            string = number.format(x).replace(",", "'")
            string = f"{string} $\\mathrm{{{unit}}}$".replace('%', '\%')
            if x >= 0:
                string = "  " + string
            return string
        cbar = self.fig.colorbar(cm.ScalarMappable(
            norm=plt.Normalize(vmin, vmax),
            cmap=plt.get_cmap("viridis")
        ), cax=self.ax_cbar)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        cbar.ax.yaxis.set_tick_params(labelsize=8)
        return self

    def _chart(self, ax, title, rollout_steps, y_gc_baseline, y_gc_improved, y_fcn, y_range):
        steps = rollout_steps + 1
        x = list(range(steps))
        
        # Pad right
        while len(y_gc_baseline) % steps != 0:
            y_gc_baseline.append(None)
            y_gc_improved.append(None)
            y_fcn.append(None) 

        # Split into step series
        y_gc_baseline = np.array(y_gc_baseline, dtype=np.float32).reshape(-1, steps)
        y_gc_improved = np.array(y_gc_improved, dtype=np.float32).reshape(-1, steps)
        y_fcn = np.array(y_fcn, dtype=np.float32).reshape(-1, steps)

        # Plot lines
        for serie_i in range(y_gc_baseline.shape[0]):
            ax.plot(x, y_gc_baseline[serie_i], color='red', alpha=0.4, linewidth=.7)
            ax.plot(x, y_gc_improved[serie_i], color='blue', alpha=0.4, linewidth=.7)
            ax.plot(x, y_fcn[serie_i], color='green', alpha=0.4, linewidth=.7)

        # Plot averages
        y_gc_baseline_mean = np.nanmean(y_gc_baseline, axis=0)
        y_gc_improved_mean = np.nanmean(y_gc_improved, axis=0)
        y_fcn_mean = np.nanmean(y_fcn, axis=0)
        ax.plot(x, y_gc_baseline_mean, color='red', linewidth=1.2, label='GraphCast Baseline (mean value)')
        ax.plot(x, y_gc_improved_mean, color='blue', linewidth=1.2, label='GraphCast Improved (mean value)')
        ax.plot(x, y_fcn_mean, color='green', linewidth=1.2, label='FourCastNet (mean value)')
        
        ax.set_title(title, fontsize=8, fontweight='bold')
    
        ax.xaxis.set_ticks(x)
        ax.set_xlim(-.25, max(x) + .25)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "$\\mathrm{{t_0}}$" if x == 0 else f"$\\mathrm{{t_0}}+{int(x*6)}$h"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="both", nbins=10))

        ax.xaxis.set_tick_params(labelsize=8)        
        ax.yaxis.set_tick_params(labelsize=8)

        y_min, y_max = y_range
        y_range = abs(y_min - y_max)
        ax.set_ylim(y_min - y_range * .1, y_max + y_range * .1)

        ax.grid(True, linestyle="--", alpha=0.5)

        # Add a bit of padding on the left
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.03, pos.y0, pos.width, pos.height])

    def chart_mse(self, rollout_steps, y_gc_baseline, y_gc_improved, y_fcn, max):
        self._chart(
            self.ax_chart_mse,
            "Mean Squared Error (MSE)",
            rollout_steps,
            y_gc_baseline,
            y_gc_improved,
            y_fcn,
            (0, max)
        )
        self.ax_chart_mse.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f" {x:,.2e}"))
        return self

    def chart_acc(self, rollout_steps, y_gc_baseline, y_gc_improved, y_fcn, min):
        self._chart(
            self.ax_chart_acc,
            "Anomaly Correlation Coefficient (ACC)",
            rollout_steps,
            y_gc_baseline,
            y_gc_improved,
            y_fcn,
            (min, 1)
        )
        self.ax_chart_acc.legend(loc="upper center", bbox_to_anchor=(.5, -0.3), ncol=3, frameon=False, fontsize=8)
        self.ax_chart_acc.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f" {x:.0%}"))
        self.ax_chart_acc.set_xlabel("Lead time", fontsize=8)
        return self

    def savefig(self, output_file_path):
        for ax in [self.ax_chart_acc, self.ax_chart_mse]:
            ax.set_box_aspect(0.19)

        plt.savefig(output_file_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.3)
        plt.close(self.fig)

    def show(self):
        for ax in [self.ax_chart_acc, self.ax_chart_mse]:
            ax.set_box_aspect(0.19)

        plt.show()
        plt.close(self.fig)

@hydra.main(version_base="1.3", config_path=".", config_name="inference_config")
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    inference = Inference(cfg)

    # Load plan, and select activities for this rank
    with open(inference.plan_file_path, "r") as f:
        frames = json.load(f)
    activities_for_this_rank = np.array_split(frames, inference.world_size)[inference.rank]
    logger.info(f"[Rank {inference.rank}] Loaded {len(activities_for_this_rank)} activities for this rank")
    
    # For each activity, generate the image
    with h5py.File(inference.container_file_path, "r") as f:
        for activity_i, activity in enumerate(activities_for_this_rank):
            output_file_path = inference.frames_dir / f"frame_{activity['frame_id']:04}.png"

            if output_file_path.exists():
                logger.info(f"[Rank {inference.rank}] File {output_file_path} already exists. Skipping.")
                continue
            logger.info(f"[Rank {inference.rank}] Generating {output_file_path}...")

            sample = activity["sample"]
            channel = activity["channel"]
            title = inference.get_channel_title(channel)
            time_str = datetime.fromtimestamp(activity["timestamp"]).strftime("%A, %d %B %Y at %H:%M:%S UTC")
            subtitle = f"Sample {sample+1} of {cfg.samples} | Time: {time_str} | Lead time: {activity['step']*6} hours"
            global_sample_id = activity["global_sample_id"]
            vmin, vmax = activity["vmin"], activity["vmax"]
            cbar_ticks_format = inference.get_channel_tick_format(channel)

            if "gc_baseline" in activity["files"]:
                img_reanalysis = inference.images_dir / activity["files"]["reanalysis"]
                img_gc_baseline = inference.images_dir / activity["files"]["gc_baseline"]
                img_gc_improved = inference.images_dir / activity["files"]["gc_improved"]
                img_fcn = inference.images_dir / activity["files"]["fcn"]
            else:
                img_reanalysis = img_gc_baseline = img_gc_improved = img_fcn = inference.images_dir / activity["files"]["reanalysis"]

            VideoFrameFourGlobes()\
                .title(title, subtitle)\
                .img_reanalysis("Reanalysis", img_reanalysis)\
                .img_gc_baseline("GraphCast Baseline", img_gc_baseline)\
                .img_gc_improved("GraphCast Improved", img_gc_improved)\
                .img_fcn("FourCastNet", img_fcn)\
                .cbar(vmin, vmax, cbar_ticks_format)\
                .chart_acc(
                    rollout_steps=cfg.rollout_steps,
                    y_gc_baseline=activity["acc"]["gc_baseline"],
                    y_gc_improved=activity["acc"]["gc_improved"],
                    y_fcn=activity["acc"]["fcn"],
                    min=activity["acc"]["min"]
                )\
                .chart_mse(
                    rollout_steps=cfg.rollout_steps,
                    y_gc_baseline=activity["mse"]["gc_baseline"],
                    y_gc_improved=activity["mse"]["gc_improved"],
                    y_fcn=activity["mse"]["fcn"],
                    max=activity["mse"]["max"]
                )\
                .footer(f"Channel id: {channel} | Dataloader global sample id: {global_sample_id} | DSM500 March 2025")\
                .savefig(output_file_path)

            logger.info(f"[Rank {inference.rank}] File {output_file_path} created. {len(activities_for_this_rank) - activity_i - 1} activities left.")
    logger.info(f"[Rank {inference.rank}] Done!")

if __name__ == "__main__":
    main()