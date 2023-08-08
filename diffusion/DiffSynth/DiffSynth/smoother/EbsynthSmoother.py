from tqdm import tqdm
import numpy as np
from PIL import Image
import torch, os

class EbsynthSmoother:
    def __init__(self, bin_path, cache_path, smooth_index=[-2, -1, 1, 2], n_jobs=16):
        self.operating_space = "pixel"
        self.bin_path = bin_path
        self.cache_path = cache_path
        self.smooth_index = smooth_index
        self.n_jobs = n_jobs
        os.makedirs(cache_path, exist_ok=True)

    def prepare(self, frames, d=1):
        for i, frame in enumerate(frames):
            frame.save(os.path.join(self.cache_path, f"source_{i}.jpg"))

    def run_cmd(self, cmd_list):
        cmd_list_file = os.path.join(self.cache_path, "cmd_list.txt")
        with open(cmd_list_file, "w", encoding="utf-8") as f:
            for cmd in cmd_list:
                f.write(cmd)
                f.write("\n")
        cmd = f"cat {cmd_list_file} | xargs -P {self.n_jobs} -I cmd sh -c 'cmd'"
        os.system(cmd)

    def warp(self, frames, index):
        cmd_list = []
        for i in range(len(frames)):
            for d in index:
                j = i + d
                if j>=0 and j<len(frames):
                    style_image = os.path.join(self.cache_path, f"style_{i}.jpg")
                    source_image = os.path.join(self.cache_path, f"source_{i}.jpg")
                    target_image = os.path.join(self.cache_path, f"source_{j}.jpg")
                    output_image = os.path.join(self.cache_path, f"output_{i}_{j}.jpg")
                    cmd = f"{self.bin_path} -style {style_image} -guide {source_image} {target_image} -weight 100.0 -output {output_image} -patchsize 21"
                    cmd_list.append(cmd)
        self.run_cmd(cmd_list)
        warped_frames = [[] for i in range(len(frames))]
        for i in range(len(frames)):
            for d in index:
                j = i + d
                if j>=0 and j<len(frames):
                    output_image = os.path.join(self.cache_path, f"output_{i}_{j}.jpg")
                    output_image = Image.open(output_image)
                    warped_frames[j].append(output_image)
        return warped_frames

    def smooth(self, frames):
        for i, frame in enumerate(frames):
            frame.save(os.path.join(self.cache_path, f"style_{i}.jpg"))
        warped_frames = self.warp(frames, self.smooth_index)
        output_frames = []
        for i in range(len(warped_frames)):
            warped_frames[i].append(frames[i])
            warped_frame = [np.array(frame).astype(np.float32) for frame in warped_frames[i]]
            warped_frame = np.stack(warped_frame)
            warped_frame = warped_frame.mean(axis=0)
            warped_frame = warped_frame.round().astype("uint8")
            warped_frame = Image.fromarray(warped_frame)
            output_frames.append(warped_frame)
        return output_frames