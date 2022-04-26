import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption

import oss2
import io

class retrieval_train_oss(Dataset):
    def __init__(self, transform, oss_config, image_root, ann_root, filename, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''

        print(oss_config)

        self.oss_bucket = self._oss_setup(oss_config)
        
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            
    def _oss_setup(self, oss_config):

        access_id = oss_config['access_id']
        access_key = oss_config['access_key']
        endpoint = oss_config['endpoint']
        bucket_name = oss_config['bucket_name']

        auth = oss2.Auth(access_id, access_key)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        return bucket
    
    def _read_img_object_from_oss(self, img_path):
        img_str = self.oss_bucket.get_object(img_path)
        img_buf = io.BytesIO()

        img_buf.write(img_str.read())
        img_buf.seek(0)
        img_object = Image.open(img_buf).convert('RGB')
        img_buf.close()

        return img_object

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])

        image = self._read_img_object_from_oss(image_path)  
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
class retrieval_eval_oss(Dataset):
    def __init__(self, transform, oss_config, image_root, ann_root, filename, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.oss_bucket = self._oss_setup(oss_config)
        
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
    
    def _oss_setup(self, oss_config):

        access_id = oss_config['access_id']
        access_key = oss_config['access_key']
        endpoint = oss_config['endpoint']
        bucket_name = oss_config['bucket_name']

        auth = oss2.Auth(access_id, access_key)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        return bucket
    
    def _read_img_object_from_oss(self, img_path):
        img_str = self.oss_bucket.get_object(img_path)
        img_buf = io.BytesIO()

        img_buf.write(img_str.read())
        img_buf.seek(0)
        img_object = Image.open(img_buf).convert('RGB')
        img_buf.close()

        return img_object
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = self._read_img_object_from_oss(image_path)    
        image = self.transform(image)  

        return image, index    