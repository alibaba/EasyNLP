from tqdm import tqdm
import numpy as np
from PIL import Image
import torch, os


class StabilizeNetSmoother:
    def __init__(self, stabilize_net):
        self.operating_space = "final latent"
        self.stabilize_net = stabilize_net

    def prepare(self, flow_frames):
        pass

    def smooth(self, frames, text_embed, num_frames=3, frame_range_l=1, frame_range_r=2, dtype=torch.float16, num_iter=3):
        if num_iter==0:
            return frames
        frames_prefix = torch.concat([frames[:1]]*num_frames, dim=0)
        frames_suffix = torch.concat([frames[-1:]]*num_frames, dim=0)
        frames = torch.concat([frames_prefix, frames, frames_suffix], dim=0)
        frames_smooth = [[] for i in range(len(frames))]
        for i in tqdm(range(len(frames))):
            j = i + num_frames
            if j>len(frames):
                break
            model_input = torch.unsqueeze(frames[i:j], dim=0).float().to(self.stabilize_net.device)
            model_output = self.stabilize_net(model_input, text_embed.float())[0].to("cpu")
            for k in range(frame_range_l, frame_range_r):
                frames_smooth[i+k].append(model_output[k])
        frames_smooth = frames_smooth[num_frames:-num_frames]
        for i in range(len(frames_smooth)):
            frames_smooth[i] = torch.stack(frames_smooth[i]).mean(dim=0)
        frames_smooth = torch.stack(frames_smooth).to(dtype)
        return self.smooth(frames_smooth, text_embed, num_iter=num_iter-1)