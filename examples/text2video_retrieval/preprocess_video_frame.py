import os
from xmlrpc.client import boolean
from cv2 import idct
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import json
from PIL import Image

# ! pip install decord
from decord import VideoReader

import argparse

import base64
from io import BytesIO

# save image to file
def image_to_file(img_numpy, save_path, size=224, resample=Image.BICUBIC):
    # 将info转成为uint8格式，否则报错"cannot convert this type"
    img = Image.fromarray(np.uint8(img_numpy)).convert('RGB')
    width, height = img.size
    short, long = (width, height) if width <= height else (height, width)
    if short == size:
        img.save(save_path)
        return save_path
    new_short, new_long = size, int(size * long / short)
    new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
    img.resize((new_w, new_h), resample) 
    img.save(save_path)
    return save_path

# extracte several candidate frames from videos (one frame per second)
def extract_video_frames(video_path, frame_num = -1):
    container = VideoReader(video_path, num_threads=16)
    total_frames = len(container)

    if total_frames <= frame_num:
        idx = np.arange(total_frames - 1)
    else:
        idx = np.linspace(0, total_frames - 1, num=frame_num, dtype=int)

    assert len(idx) == frame_num, video_path

    # shape=time*H*W*C 的numpy数组 通道顺序为RGB
    image_list = container.get_batch(idx).asnumpy() 
    
    return image_list

# save video frames to image files
def extraction_process_save_path(csv_dir, json_dir, video_dir, frame_num, frame_dir, output):
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    csv = pd.read_csv(csv_dir)
    data = json.load(open(json_dir, 'r'))
    output_file = []
    video_ids = list(csv['video_id'].values)  
    processed = []
    for itm in tqdm(data['sentences']):  
        if itm['video_id'] in video_ids:
            video_id = itm['video_id']
            output_file.append(itm['caption']+'\t'+os.path.join(frame_dir, video_id))
            
            if video_id in processed:
                continue            
            else:
                processed.append(video_id)

            if os.path.exists(os.path.join(frame_dir, video_id)):
                continue 

            video_path = os.path.join(video_dir, "{}.mp4".format(video_id))
            try:
                image_list = extract_video_frames(video_path, frame_num).tolist() # 
            except:
                print(video_id)
                print(video_path)
                import pdb;pdb.set_trace()

            for frame_idx in range(len(image_list)):
                image = image_list[frame_idx]
                if not os.path.exists(os.path.join(frame_dir, video_id)):
                    os.mkdir(os.path.join(frame_dir, video_id))
                save_path = '{}.jpeg'.format(frame_idx)
                save_path = os.path.join(frame_dir, video_id, save_path)
                imagepath = image_to_file(image, save_path)

    with open(output, 'w') as of:
        of.write('\n'.join(output_file))
    if 'test' in output:
        with open('./msrvtt_subset/MSRVTT_test_part_text.tsv', 'w') as of:
            of.write('\n'.join([o.split('\t')[0] for o in output_file]))
        with open('./msrvtt_subset/MSRVTT_test_part_video.tsv', 'w') as of:
            of.write('\n'.join([o.split('\t')[1] for o in output_file]))
    print("Finished processing {} videos in total.".format(len(video_ids)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--csv_dir', default='', type=str, help='the dir of csv file')
    parser.add_argument('--json_dir', default='', type=str, help='the dir of json file')
    parser.add_argument('--video_dir', default='', type=str, help='the dir of input videos')
    parser.add_argument('--frame_num', default=-1, type=str, help='the number of frames to be extracted')
    parser.add_argument('--frame_dir', default='', type=str, help='the root dir of saved frames')
    parser.add_argument('--output', default='', type=str, help='the root dir of output txt')
    
    args = parser.parse_args()

    frame_num = int(args.frame_num)

    extraction_process_save_path(args.csv_dir, args.json_dir, args.video_dir, frame_num, args.frame_dir, args.output)
