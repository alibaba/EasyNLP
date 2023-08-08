import os, torch, cv2, transformers
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from math import factorial
from PIL import Image
from controlnet_aux.open_pose import (
    Body, Hand, Face,
    HWC3, resize_image, handDetect, faceDetect, draw_pose
)
from controlnet_aux import MidasDetector


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # Copyed from https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


class VideoControlnetImageProcesserOpenpose:
    def __init__(self, body_estimation, hand_estimation=None, face_estimation=None):
        self.body_estimation = body_estimation
        self.hand_estimation = hand_estimation
        self.face_estimation = face_estimation

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="lllyasviel/ControlNet", device=None, filename=None, hand_filename=None, face_filename=None, cache_dir=None):

        if pretrained_model_or_path == "lllyasviel/ControlNet":
            filename = filename or "annotator/ckpts/body_pose_model.pth"
            hand_filename = hand_filename or "annotator/ckpts/hand_pose_model.pth"
            face_filename = face_filename or "facenet.pth"

            face_pretrained_model_or_path = "lllyasviel/Annotators"
        else:
            filename = filename or "body_pose_model.pth"
            hand_filename = hand_filename or "hand_pose_model.pth"
            face_filename = face_filename or "facenet.pth"

            face_pretrained_model_or_path = pretrained_model_or_path

        if os.path.isdir(pretrained_model_or_path):
            body_model_path = os.path.join(pretrained_model_or_path, filename)
            hand_model_path = os.path.join(pretrained_model_or_path, hand_filename)
            face_model_path = os.path.join(face_pretrained_model_or_path, face_filename)
        else:
            body_model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
            hand_model_path = hf_hub_download(pretrained_model_or_path, hand_filename, cache_dir=cache_dir)
            face_model_path = hf_hub_download(face_pretrained_model_or_path, face_filename, cache_dir=cache_dir)

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        body_estimation = Body(body_model_path, device)
        hand_estimation = Hand(hand_model_path, device)
        face_estimation = Face(face_model_path, device)

        return cls(body_estimation, hand_estimation, face_estimation)
    
    
    def smooth_face(self, pose_list):
        faces_list = []
        for i in range(len(pose_list)):
            faces = np.array(pose_list[i]["faces"])
            faces_list.append(faces)
            if i+1>=len(pose_list) or np.any(np.array(pose_list[i+1]["faces"]).shape!=np.array(faces).shape):
                if len(faces_list)<7:
                    faces_list = []
                    continue
                for people in range(len(faces)):
                    for component in range(70):
                        for dimension in range(2):
                            candidates = np.array([faces_list[frame][people][component][dimension] for frame in range(len(faces_list))])
                            candidates = savitzky_golay(candidates, 7, 2)
                            for frame in range(len(faces_list)):
                                faces_list[frame][people][component][dimension] = candidates[frame]
                for j in range(len(faces_list)):
                    pose_list[i-len(faces_list)+j]["faces"] = faces_list[j]
                faces_list = []
        return pose_list
            

    def estimate_pose(self, input_image, detect_resolution=512, include_hand=True, include_face=True):

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        input_image = input_image[:, :, ::-1].copy()
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(input_image)
            hands = []
            faces = []
            if include_hand:
                # Hand
                hands_list = handDetect(candidate, subset, input_image)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(input_image[y:y+w, x:x+w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
            if include_face:
                # Face
                faces_list = faceDetect(candidate, subset, input_image)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(input_image[y:y+w, x:x+w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())

            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)

            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
        return pose
    
    def __call__(self, frames, detect_resolution=512, image_resolution=512):
        if len(frames)==0:
            return []
        temp = np.array(frames[0], dtype=np.uint8)
        temp = HWC3(temp)
        temp = resize_image(temp, detect_resolution)
        H, W, _ = temp.shape
        temp = resize_image(temp, image_resolution)
        H_out, W_out, _ = temp.shape
        
        pose_list = []
        for frame in tqdm(frames, desc="Estimating pose"):
            pose = self.estimate_pose(frame, detect_resolution=detect_resolution)
            pose_list.append(pose)
            
        pose_list = self.smooth_face(pose_list)
        
        detected_maps = []
        for pose in tqdm(pose_list, desc="Drawing pose"):
            canvas = draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True)
            detected_map = HWC3(canvas)
            detected_map = cv2.resize(detected_map, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
            detected_map = Image.fromarray(detected_map)
            detected_maps.append(detected_map)
        return detected_maps
    
    
class VideoControlnetImageProcesserDepth:
    def __init__(self, model_path="Intel/dpt-large", sigmoid=False, threshold=1.0):
        self.depth_estimator = transformers.pipeline(task="depth-estimation", model=model_path, device=torch.device("cuda:0"))
        self.sigmoid = sigmoid
        self.threshold = threshold

    def estimate_depth(self, image):
        image = self.depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = image.astype(np.float32)
        image = np.concatenate([image, image, image], axis=2)
        return image
    
    def __call__(self, frames):
        depth_frames = []
        for frame in tqdm(frames, desc="Estimating Depth"):
            depth_frames.append(self.estimate_depth(frame))
        depth_frames_mean = np.stack(depth_frames).mean()
        normalized_depth_frames = []
        for frame in depth_frames:
            frame_mean = frame.mean()
            frame = frame - frame_mean + depth_frames_mean
            if self.sigmoid:
                frame = (frame / (255/2) - self.threshold) * 5
                frame = 1.0 / (1 + np.exp(-frame))
                frame = frame * 255
            frame = frame.clip(0, 255)
            frame = Image.fromarray(frame.astype(np.uint8))
            normalized_depth_frames.append(frame)
        return normalized_depth_frames
