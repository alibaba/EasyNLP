import argparse
import json
import os
import time
import math
import logging
import random

from prettytable import PrettyTable
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel

from beautiful_prompt.utils import (
    read_json,
    save_json,
    init_sd_pipeline,
    set_seed
)
from beautiful_prompt.evaluator import (
    ImageReward,
    CLIPScore,
    AestheticScore,
    PickScore,
    HPS
)

logging.getLogger('transformers').setLevel(logging.ERROR)

def generate_prompts(args, data):
    if args.method == 'raw':
        for d in data:
            d['generated_prompt'] = d['raw_prompt']

    elif args.method == 'magic-prompt':
        model = AutoModelForCausalLM.from_pretrained('Gustavosta/MagicPrompt-Stable-Diffusion')
        tokenizer = AutoTokenizer.from_pretrained('Gustavosta/MagicPrompt-Stable-Diffusion')
        
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)

        raw_prompts = [x['raw_prompt'] for x in data]
        generated_prompts = []
        
        for i in tqdm(range(math.ceil(len(raw_prompts) / args.batch_size)), desc='Generating prompts...', disable=args.disable_tqdm):
            batch_ixs = slice(i * args.batch_size, (i + 1) * args.batch_size)
            
            inputs = raw_prompts[batch_ixs]
            
            model_inputs = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(args.device)
        
            generations = model.generate(
                **model_inputs,
                max_length=args.max_length,
                do_sample=False,
                repetition_penalty=1.2
            )
            
            generated_prompt = tokenizer.batch_decode(generations, skip_special_tokens=True)
            generated_prompts.extend([x.strip() for x in generated_prompt])

        for d, generated_prompt in zip(data, generated_prompts):
            d['generated_prompt'] = generated_prompt

        del model
        torch.cuda.empty_cache()

    elif args.method == 'chatgpt':
        for d in data:
            d['generated_prompt'] = d['chatgpt_prompt']

    elif args.method == 'beautiful-prompt':
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model.to(args.device)

        raw_prompts = [x['raw_prompt'] for x in data]
        generated_prompts = []
        
        for i in tqdm(range(math.ceil(len(raw_prompts) / args.batch_size)), desc='Generating prompts...', disable=args.disable_tqdm):
            batch_ixs = slice(i * args.batch_size, (i + 1) * args.batch_size)
            
            inputs = [f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompts}\nOutput:' for raw_prompts in raw_prompts[batch_ixs]]
            
            model_inputs = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(args.device)
        
            generations = model.generate(
                **model_inputs,
                max_length=args.max_length,
                do_sample=False,
                repetition_penalty=1.2
            )
            
            generated_prompt = tokenizer.batch_decode(generations[:, model_inputs.input_ids.size(1):], skip_special_tokens=True)
            generated_prompts.extend([x.strip() for x in generated_prompt])

        for d, generated_prompt in zip(data, generated_prompts):
            d['generated_prompt'] = generated_prompt

        del model
        torch.cuda.empty_cache()

    else:
        raise NotImplementedError()

    return data

def generate_images(args, data):
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    prompts = [x['generated_prompt'] for x in data]

    os.makedirs(args.imgs_path, exist_ok=True)
    
    sd_pipeline = init_sd_pipeline()
    sd_pipeline.set_progress_bar_config(disable=True)
    
    for i in tqdm(range(math.ceil(len(prompts) / args.batch_size)), desc='Generating images...', disable=args.disable_tqdm):
        batch_ixs = slice(i * args.batch_size, (i + 1) * args.batch_size)
        
        images = sd_pipeline.text2img(
            prompts[batch_ixs],
            negative_prompt='nsfw, ((ugly)), (duplicate), morbid, mutilated, [out of frame], (extra fingers), mutated hands',
            width=512,
            height=512,
            max_embeddings_multiples=6,
            num_inference_steps=args.num_inference_steps,
            generator=generator).images

        for j, img in enumerate(images):
            idx = i * args.batch_size + j
            img_path = os.path.join(args.imgs_path, f'{idx}.png')
            img.save(img_path)
            data[idx]['img_path'] = img_path

    del sd_pipeline
    torch.cuda.empty_cache()

    return data

def compute_score(args, data, score_type):
    if score_type == 'ImageReward':
        evaluator = ImageReward(args.imagereward_path, device=args.device)
    elif score_type == 'CLIPScore':
        evaluator = CLIPScore(device=args.device)
    elif score_type == 'Aesthetic':
        evaluator = AestheticScore(device=args.device)
    elif score_type == 'PickScore':
        evaluator = PickScore(processor_checkpoint='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                              model_checkpoint=args.pickscore_path,
                              device=args.device)
    elif score_type == 'HPS':
        evaluator = HPS(model_checkpoint=args.hps_path, device=args.device)
    else:
        raise NotImplementedError(f'{score_type} is not implemented.')

    for i in tqdm(range(math.ceil(len(data) / args.batch_size)), desc=f'Computing {score_type}...', disable=args.disable_tqdm):
        batch_ixs = slice(i * args.batch_size, (i + 1) * args.batch_size)
        
        # use raw_prompts for computing ImageReward, PickScore, etc...
        prompts = [x['raw_prompt'] for x in data[batch_ixs]]
        imgs = [x['img_path'] for x in data[batch_ixs]]
        
        scores = evaluator(prompts, imgs)
        
        for j, score in enumerate(scores):
            idx = i * args.batch_size + j
            data[idx][score_type] = score
        
    del evaluator
    torch.cuda.empty_cache()

    return data

def main(args):
    data = read_json(args.data_path)
    
    if args.num_items is not None and len(data) > args.num_items:
        data = random.sample(data, args.num_items)
    
    # 1. Generate prompts
    data = generate_prompts(args, data)
    save_json(data, args.result_path)
    
    # 2. Generate images and save them
    data = generate_images(args, data)
    save_json(data, args.result_path)
    
    # 3. Compute Scores
    args.batch_size *= 2 # accelerate
    if args.score_types == 'all':
        scores = ['PickScore', 'Aesthetic', 'HPS', 'CLIPScore']
    else:
        scores = args.score_types

    for score in scores:
        data = compute_score(args, data, score_type=score)
        save_json(data, args.result_path)
    
    # 4. Results
    result = {'All': {}}
    sums = {}
    for score in scores:
        result['All'][score] = round(sum([x[score] for x in data]) / len(data), 4)
    
    if 'type' in data[0]:
        for x in data:
            x_type = x['type']
            if x_type not in sums:
                sums[x_type] = {'count': 0}
                for score in scores:
                    sums[x_type][score] = 0

            sums[x_type]['count'] += 1
            for score in scores:
                sums[x_type][score] += x[score]

        for x_type, score_sum in sums.items():
            for score in scores:
                if x_type not in result:
                    result[x_type] = {}
                result[x_type][score] = round(score_sum[score] / score_sum['count'], 4)
        
    table = PrettyTable()
    table.title = f'Method: {args.method}'
    table.field_names = ['Type'] + scores

    for score, score_average in result.items():
        table.add_row([score] + [score_average[score] for score in scores])

    print(table)
    
    result['items'] = data
    save_json(result, args.result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--method', type=str, default='beautiful-prompt',
                        choices=['raw', 'magic-prompt', 'chatgpt', 'beautiful-prompt'],
                        help='The method to generate prompts.')
    parser.add_argument('--score_types', type=str, nargs='+', default='all')
    
    parser.add_argument('--data_path', type=str, default='data/test.json', help='Path to the data file.')
    parser.add_argument('--result_path', type=str, default='data/eval-results.json', help='Path to the result file.')
    parser.add_argument('--model_path', type=str, default='outputs/ppo', help='Path to the beautiful-prompt model dir.')
    parser.add_argument('--imgs_path', type=str, default='data/imgs', help='Path to the images dir.')
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_inference_steps', type=int, default=20)

    parser.add_argument('--hps_path', type=str)
    parser.add_argument('--pickscore_path', type=str, default='yuvalkirstain/PickScore_v1')
    parser.add_argument('--imagereward_path', type=str, default='ImageReward-v1.0')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_items', type=int, default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    
    
    args = parser.parse_args()
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    main(args)
