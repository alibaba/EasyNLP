import imageio, os, cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def read_video_from_images(folder):
    num_frames = len(os.listdir(folder))
    video = [Image.open(f"{folder}/{i}.png") for i in tqdm(range(num_frames), desc="Loading frames")]
    return video

def save_gif(frames, file_name, duration=1/15):
    imageio.mimsave(file_name, frames, 'GIF', duration=duration)

def save_images(frames, folder):
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(folder, f"{i}.png"))

def save_video(frames, file_name, fps=30):
    width, height = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)[:,:,[2,1,0]]
        videoWriter.write(frame)
    videoWriter.release()

def save_compare_video(frames_list, file_name, fps=30):
    width, height = frames_list[0][0].size
    width *= len(frames_list)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
    for i in range(len(frames_list[0])):
        frame = np.concatenate([np.array(frames_list[j][i])[:,:,[2,1,0]] for j in range(len(frames_list))], axis=1)
        videoWriter.write(frame)
    videoWriter.release()
    
def read_video_from_video(video_file, height=None, width=None, num_frames=None):
    capture = cv2.VideoCapture(video_file)
    frame_list = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        if height is None or width is None:
            frame = Image.fromarray(frame)
        elif frame_height/frame_width<height/width:
            croped_width = int(frame_height/height*width)
            left = (frame_width-croped_width)//2
            frame = frame[:,left:left+croped_width]
            frame = Image.fromarray(frame).resize((width, height))
        else:
            croped_height = int(frame_width/width*height)
            left = (frame_height-croped_height)//2
            frame = frame[left:left+croped_height,:]
            frame = Image.fromarray(frame).resize((width, height))
        frame_list.append(frame)
        if num_frames is not None and len(frame_list)>=num_frames:
            break
    capture.release()
    return frame_list
