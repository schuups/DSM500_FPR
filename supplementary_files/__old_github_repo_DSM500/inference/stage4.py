import sys
sys.path.append('/iopsstor/scratch/cscs/stefschu/DSM500/github/modulus-a5275d8')

import cv2
import json
import hydra
from omegaconf import DictConfig

from modulus.launch.logging import PythonLogger

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


@hydra.main(version_base="1.3", config_path=".", config_name="inference_config")
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    inference = Inference(cfg)

    assert inference.world_size == 1, "This stage of Inference is not supposed to be run in distributed mode"

    # Load plan, and select activities for this rank
    with open(inference.plan_file_path, "r") as f:
        plan = json.load(f)

    logger.info(f"Generating video at {inference.video_file_path} ...")
    video_generator = VideoGenerator(inference.video_file_path)

    for frame in plan:
        file_path = inference.frames_dir / f"frame_{frame['frame_id']:04}.png"
        assert file_path.exists(), f"File {file_path} does not exist"

        video_generator.add_image(file_path)
        logger.info(f"Added {file_path}")
    
    video_generator.close()
    logger.info(f"Video assembled.")

    # logger.info(f"Cleaning up frames ...")
    # for frame in inference.frames_dir.iterdir():
    #     frame.unlink()
    
    # logger.info(f"Cleaning up images ...")
    # for image in inference.images_dir.iterdir():
    #     image.unlink()
    
    # logger.info(f"Clean up done.")

if __name__ == "__main__":
    main()