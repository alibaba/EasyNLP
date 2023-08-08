from .video_level_processors import VideoControlnetImageProcesserOpenpose, VideoControlnetImageProcesserDepth
from controlnet_aux import ContentShuffleDetector, PidiNetDetector, HEDdetector, OpenposeDetector, MidasDetector
import torch, transformers
import numpy as np
from PIL import Image


class ControlnetImageProcesserDepth:
    def __init__(self, model_path="Intel/dpt-large", device=torch.device("cuda:0"), threshold=None):
        self.depth_estimator = transformers.pipeline(task="depth-estimation", model=model_path, device=device)
        self.threshold = threshold

    def __call__(self, image):
        image = self.depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        if self.threshold is not None:
            image[image<self.threshold] = 0
        image = Image.fromarray(image)
        return image

    
class ControlnetImageProcesserHED:
    def __init__(self, model_path="lllyasviel/Annotators", detect_resolution=None, device="cuda"):
        self.apply_softedge = HEDdetector.from_pretrained(model_path)
        self.apply_softedge.netNetwork = self.apply_softedge.netNetwork.to(device)
        self.detect_resolution = detect_resolution

    def __call__(self, image):
        detect_resolution = self.detect_resolution
        if detect_resolution is None:
            detect_resolution = min(image.size)
        image = self.apply_softedge(
            image,
            detect_resolution=detect_resolution,
            image_resolution=min(image.size)
        )
        return image


class ControlnetImageProcesserShuffle:
    def __init__(self, seed=0):
        self.seed = seed
        self.processor = ContentShuffleDetector()

    def __call__(self, image):
        np.random.seed(self.seed)
        image = self.processor(image)
        return image
    
    
class ControlnetImageProcesserPose:
    def __init__(self, detect_resolution=None, device="cuda"):
        self.processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.detect_resolution = detect_resolution

    def __call__(self, image):
        detect_resolution = self.detect_resolution
        if detect_resolution is None:
            detect_resolution = min(image.size)
        image = self.processor(
            image,
            detect_resolution=detect_resolution,
            image_resolution=min(image.size),
            hand_and_face=True
        )
        return image
    
    
class ControlnetImageProcesserTile:
    def __init__(self, detect_resolution=None, device="cuda"):
        pass

    def __call__(self, image):
        return image