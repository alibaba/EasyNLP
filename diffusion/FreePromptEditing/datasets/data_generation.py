import json
from tqdm import tqdm
import random


color_words = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "gray",
               "brown", "beige", "cyan", "magenta", "teal", "lime", "olive", "navy", "maroon", "silver",
               "gold", "bronze", "peach", "coral", "indigo", "violet", "turquoise","chocolate"]
Car_real_edit = []

with open('StanfordCar.json', 'r') as jsonfile:
    data = json.load(jsonfile)
for row in tqdm(data):
    image_name = row['image']
    for word in color_words:
        if word in row['prompt']:
            continue
        else:
            soure_prompt = row['prompt']
            edit_prompt = 'a '+ word +' car'
            prompt = {}
            
            prompt['image_name']= image_name
            prompt['soure_prompt']= soure_prompt
            prompt['edit_prompt']= edit_prompt
            Car_real_edit.append(prompt)
            
with open("Car_real_edit.json", "w") as jsonl_file:
    json.dump(Car_real_edit, jsonl_file,indent=4)

Car_fake_edit = []
for word in color_words:
        soure_prompt = 'a ' + word + ' car'
        
        color_words_edit = color_words[:]
        color_words_edit.remove(word)

        for edit_word in color_words_edit:
            prompt = {}
            edit_prompt =  'a ' + edit_word + ' car'
            prompt['soure_prompt']= soure_prompt
            prompt['edit_prompt']= edit_prompt
            Car_fake_edit.append(prompt)
            
with open("Car_fake_edit.json", "w") as jsonl_file:
    json.dump(Car_fake_edit, jsonl_file,indent=4)


Car_fake_edit = []
for word in color_words:
        soure_prompt = 'a ' + word + ' car'
        
        color_words_edit = color_words[:]
        color_words_edit.remove(word)

        for edit_word in color_words_edit:
            prompt = {}
            edit_prompt =  'a ' + edit_word + ' car'
            prompt['soure_prompt']= soure_prompt
            prompt['edit_prompt']= edit_prompt
            Car_fake_edit.append(prompt)
            
with open("Car_fake_edit.json", "w") as jsonl_file:
    json.dump(Car_fake_edit, jsonl_file,indent=4)
    
    
json_path = 'ImageNet.json'
with open(json_path, 'r') as jsonfile:
    data = json.load(jsonfile)
    
ImageNet_real_edit = []
ImageNet_fake_edit = []

for row in tqdm(data):

    path = row['path']
    dir_img = path.split('/')
    is_test = row['is_test']


    if is_test =='True':
        source_word = row['source'] 
        target_word = row['target']

        source_prompt = 'a photo of a {}'.format(source_word)
        target_prompt = 'a photo of a {}'.format(target_word)
        prompt = {}
        
        prompt['image_name']= path
        prompt['soure_prompt']= source_prompt
        prompt['edit_prompt']= target_prompt
        ImageNet_real_edit.append(prompt)
        
with open("ImageNet_real_edit.json", "w") as jsonl_file:
    json.dump(ImageNet_real_edit, jsonl_file,indent=4) 
        
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]
word_list = ["dog","giraffe","horse","lion", "rabbit", "sheep", "cat", "monkey", "leopard", "tiger"]


for row in tqdm(data):
    path = row['path']
    dir_img = path.split('/')
    # print(dir_img)
    is_test = row['is_test']


    if is_test =='True':
        source_word = row['source'] 
        target_word = row['target']

        text = random.choice(imagenet_templates_small)
        source_prompt = text.format(source_word)
        target_prompt = text.format(target_word)

        prompt = {}
        
        prompt['soure_prompt']= source_prompt
        prompt['edit_prompt']= target_prompt
        ImageNet_fake_edit.append(prompt)
        
for word in tqdm(word_list):

    source_prompt = 'a '+ word + ' standing in the park'

    words_edit = word_list[:]
    words_edit.remove(word)

    for edit_word in words_edit:
        prompts = []
        edit_prompt =  'a ' + edit_word + ' standing in the park'
        prompt = {}
        prompt['soure_prompt']= source_prompt
        prompt['edit_prompt']= edit_prompt
        ImageNet_fake_edit.append(prompt)
        
with open("ImageNet_fake_edit.json", "w") as jsonl_file:
    json.dump(ImageNet_fake_edit, jsonl_file,indent=4) 