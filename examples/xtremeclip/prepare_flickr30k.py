import json
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
import os

split_file = '/home/moming/data/flickr/dataset_flickr30k.json'
image_path = '/home/moming/data/flickr/flickr30k-images'
dest_path = '/home/moming/data/flickr'
contents = json.loads(open(split_file, 'r').read())
contents = contents['images']

data = {"train": {"img": [], "txt": []}, "val": {"img": [], "txt": []}, "test": {"img": [], "txt": []}}

for c in tqdm(contents):
    split = c['split']
    img_name = c['filename']
    imgFilename = os.path.join(image_path, img_name)
    img = Image.open(imgFilename)  # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str

    sentences = [s['raw'] for s in c['sentences']]
    label_id = []
    for s in sentences:
        label_id.append(len(data[split]['txt']))
        data[split]['txt'].append(s)
    data[split]['img'].append({"data": base64_str, "label": label_id})

for split in ["train", "val", "test"]:
    img_data = data[split]['img']
    textual_data = data[split]['txt']
    with open(os.path.join(dest_path + "_" + split + '_img.pt'), 'a') as f:
        for i in img_data:
            print(json.dumps(i), file=f)

    with open(os.path.join(dest_path + "_" + split + '_txt.pt'), 'a') as f:
        for t in textual_data:
            print(t, file=f)


