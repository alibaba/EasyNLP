import os
from xmlrpc.client import boolean
from cv2 import idct
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from PIL import Image

# ! pip install decord
from decord import VideoReader

import argparse

import base64
from io import BytesIO

# convert image to base64
def image_to_base64(img_numpy):
    # 将info转成为uint8格式，否则报错"cannot convert this type"
    img = Image.fromarray(np.uint8(img_numpy)).convert('RGB')
    img_buffer = BytesIO()
    #img.save(img_buffer, format=img.format)   # 报错ValueError: unknown file extension: 
    img.save(img_buffer, format='jpeg')
    byte_data = img_buffer.getvalue()
    base64_str = str(base64.b64encode(byte_data), 'utf-8')
 
    return base64_str

# save image to file
def image_to_file(img_numpy, save_path):
    # 将info转成为uint8格式，否则报错"cannot convert this type"
    img = Image.fromarray(np.uint8(img_numpy)).convert('RGB')
    img.save(save_path)
    return save_path

# extracte several candidate frames from videos (one frame per second)
def extract_video_frames(video_path, frame_num = -1):
    container = VideoReader(video_path, num_threads=16)
    total_frames = len(container)

    if frame_num == -1:
        # 每1秒取一帧
        fps = container.get_avg_fps()
        idx = list(range(0, total_frames, int(fps)))

    else:
        # 平均取frame_num帧 (time=frame_num)
        idx = list(range(0, total_frames, int(total_frames/frame_num)))

        if len(idx) == frame_num + 1:
            idx = idx[:-1]
        assert len(idx) == frame_num

    # shape=time*H*W*C 的numpy数组 通道顺序为RGB
    image_list = container.get_batch(idx).asnumpy() 
    
    return image_list


# save video frames to image byte64 codes
def extraction_process_save_base64(input_file, video_root_dir, frame_num, output_file):
    outfile = open(output_file, 'w', encoding='utf-8')

    with open(input_file, 'r', encoding='utf-8') as input:
        lines = input.readlines()
        for line in tqdm(lines):
            id, video_path = line.strip('\n').split('\t') 
            video_path = os.path.join(video_root_dir, video_path)

            image_list = extract_video_frames(video_path, frame_num).tolist() # 

            image_base64_list = []
            for image in image_list:
                imagebase64 = image_to_base64(image)
                image_base64_list.append(imagebase64)

            outfile.write("{}\t{}\n".format(id, image_base64_list))
    
    outfile.close()
    
    print("Finished processing {} videos in total.".format(len(lines)))


# save video frames to image files
def extraction_process_save_path(input_file, video_root_dir, frame_num, frame_root_dir, output_file):
    if not os.path.exists(frame_root_dir):
        os.mkdir(frame_root_dir)

    outfile = open(output_file, 'w', encoding='utf-8')

    with open(input_file, 'r', encoding='utf-8') as input:
        lines = input.readlines()
        for line in tqdm(lines):
            id, video_path = line.strip('\n').split('\t') 
            video_path = os.path.join(video_root_dir, video_path)

            image_list = extract_video_frames(video_path, frame_num).tolist() # 

            imagepath_list = []
            for frame_idx in range(len(image_list)):
                image = image_list[frame_idx]
                save_path = '{}_{}.jpeg'.format(id, frame_idx)
                save_path = os.path.join(frame_root_dir, save_path)
                imagepath = image_to_file(image, save_path)
                imagepath_list.append(imagepath)

            outfile.write("{}\t{}\n".format(id, imagepath_list))
    
    outfile.close()
    
    print("Finished processing {} videos in total.".format(len(lines)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--tables', default='', type=str, help='input table names')
    parser.add_argument('--video_root_dir', default='', type=str, help='the root dir of input videos')
    parser.add_argument('--frame_num', default=-1, type=str, help='the number of frames to be extracted')
    parser.add_argument('--enable_frame_path', action='store_true', help='the output format of frames, image_path or image_base64')
    parser.add_argument('--frame_root_dir', default='', type=str, help='the root dir of saved frames')
    parser.add_argument('--outputs', default='', type=str, help='output table names')
    
    args = parser.parse_args()

    frame_num = int(args.frame_num)

    if args.enable_frame_path:
        assert args.frame_root_dir != '', 'frame_root_dir should be assigned.'

    if args.enable_frame_path:
        extraction_process_save_path(args.tables, args.video_root_dir, frame_num, args.frame_root_dir, args.outputs)
    else:
        extraction_process_save_base64(args.tables, args.video_root_dir, frame_num, args.outputs)
