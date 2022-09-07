# encoding=utf-8
# convert video to the format of 'id \t path' 

import os
from decord import VideoReader

valid_video_ext = ['mp4']

def validation(video_path):
    try:
        container = VideoReader(video_path, num_threads=16)
        total_frames = len(container)
        img = container[total_frames-1]
        print('load success')
    except:
        print(f'broken video {video_path}')
        return False
    
    return True

def preprocess(input_dir):
    results = []
    prefix_path_len = len(input_dir)

    for root_dir, subroot_dir, filenames in os.walk(input_dir):
        if subroot_dir != []:
            continue

        if root_dir.split('/')[-1].startswith('.'):
            continue

        print (root_dir, subroot_dir, filenames)

        for file in filenames:
            ext = file.split('.')[-1]
            path = os.path.join(root_dir, file)
            
            if ext in valid_video_ext and validation(path):
                id = file.split('.')[0]
                results.append({'id': id, 'path': path[prefix_path_len:]})

    return results

def write(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in results:
            outfile.write('{}\t{}\n'.format(result['id'], result['path']))

if __name__ == '__main__':
    input_dir = '/home/zhuxiangru.zxr/workspace/tmp_update/EasyNLP/tmp/sample_videos/'
    output_file = '/home/zhuxiangru.zxr/workspace/tmp_update/EasyNLP/tmp/video_id_path.txt'

    result = preprocess(input_dir)
    write(result, output_file)


