import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import cv2
import json
import hydra
from omegaconf import DictConfig

from modulus.launch.logging import PythonLogger

# FIXIT: Move it under modulus
from inference import Inference

class VideoGenerator:
    def __init__(self, output_file_path, fps=10):
        self.out = None
        self.output_file_path = output_file_path
        self.fps = fps
        self.height = None
        self.width = None

    def add_image(self, file_path):
        frame = cv2.imread(file_path)

        if self.out is None:
            self.height, self.width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
            self.out = cv2.VideoWriter(self.output_file_path, fourcc, self.fps, (self.width, self.height))

        assert frame.shape == (self.height, self.width, 3), f"Frame {file_path} has shape {frame.shape}, expected {(self.height, self.width, 3)}"

        self.out.write(frame)

    def close(self):
        self.out.release()

@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()

    inference = Inference(cfg, logger)
    assert inference.world_size == 1, "This stage of Inference is not supposed to be run in distributed mode"

    # Load plan, and select activities for this rank
    with open(inference.plan_file_path, "r") as f:
        plan = json.load(f)

    logger.info(f"Generating video at {inference.video_file_path} ...")
    video_generator = VideoGenerator(inference.video_file_path)

    for frame in plan:
        frame_id = frame["frame_id"]
        file_path = inference.images_dir / f"frame_{frame['frame_id']}.png"
        assert file_path.exists(), f"File {file_path} does not exist"

        video_generator.add_image(file_path)
        print(f"Added {file_path}")

        if frame["type"] == "pause":
            video_generator.add_image(file_path)
            print(f"Added {file_path}")
    
    video_generator.close()
    logger.info(f"Done!")

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