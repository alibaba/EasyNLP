import json
import math
import os
import io
import random
import argparse
import time

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from beautiful_prompt.utils import read_json

def create_reward_fn(args):  # noqa:  C901
    if os.environ.get("RANK", "0") == "0":
        class RewardModel(nn.Module):
            def __init__(self, checkpoint_path):
                super().__init__()
                self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=1)

            def forward(self, input_ids: torch.LongTensor, attention_mask = None) -> torch.Tensor:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                value = outputs['logits'].squeeze(-1)
                return value
                    
        aes_model = RewardModel(args.aes_model_path).eval().half()
        ps_model = RewardModel(args.ps_model_path).eval().half()
        
        aes_tokenizer = AutoTokenizer.from_pretrained(args.aes_model_path)
        aes_tokenizer.truncation_side = "left"
        aes_tokenizer.padding_side = "right"
        
        ps_tokenizer = AutoTokenizer.from_pretrained(args.ps_model_path)
        ps_tokenizer.truncation_side = "left"
        ps_tokenizer.padding_side = "right"
        
        reward_device = torch.cuda.device_count() - 1

        aes_model = aes_model.to(reward_device)
        ps_model = ps_model.to(reward_device)

        reward_batch_size = args.reward_batch_size
        delta_reward = True
        
        @torch.no_grad()
        def get_reward(raw_prompts, generated_prompts):
            aes_input = aes_tokenizer(
                [p + aes_tokenizer.eos_token for p in generated_prompts],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )

            ps_input = ps_tokenizer(
                [f"Input: {rp}\nOutput: {p}{ps_tokenizer.eos_token}" for rp, p in zip(raw_prompts, generated_prompts)],
                padding=True,
                truncation=True,
                max_length=400,
                return_tensors="pt",
            )

            mbs = reward_batch_size
            aess = []
            irs = []
            for i in range(math.ceil(len(generated_prompts) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = aes_input.input_ids[batch_ixs].to(reward_device)
                attention_mask = aes_input.attention_mask[batch_ixs].to(reward_device)
                scores = aes_model(input_ids, attention_mask)
                aess.extend(scores)
                
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = ps_input.input_ids[batch_ixs].to(reward_device)
                attention_mask = ps_input.attention_mask[batch_ixs].to(reward_device)
                scores = ps_model(input_ids, attention_mask)
                irs.extend(scores)
            
            prompts_len = [max(len(p), 200) for p in generated_prompts]

            return (1-args.alpha) * torch.hstack(aess) + args.alpha * torch.hstack(irs) + 0.01 * torch.tensor(len(prompts_len))

        def reward_fn(samples, prompts, original_output, **kwargs):
            generated_prompts = [s.replace(p, '').strip().strip('</s>') for p, s in zip(prompts, samples)]

            raw_prompts = [p.split('Input:')[1].split('Output:')[0].strip().strip('</s>') for p in prompts]
            rewards = get_reward(raw_prompts, generated_prompts)
            
            if not delta_reward:
                return rewards

            original_rewards = get_reward(raw_prompts, original_output)

            return rewards - original_rewards
    else:
        reward_fn = True

    return reward_fn

def main(args):
    config = TRLConfig(
        train=TrainConfig(
            seq_length=args.max_length,
            epochs=10000,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            eval_interval=args.eval_interval,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            checkpoint_dir=args.save_path,
            save_optimizer=False,
            tracker="tensorboard",
            logging_dir=args.logging_dir,
            project_name=f'beautilful-prompt ppo [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]',
            save_best=False
        ),
        model=ModelConfig(model_path=args.model_path, num_layers_unfrozen=args.num_layers_unfrozen),
        tokenizer=TokenizerConfig(tokenizer_path=args.model_path, truncation_side="left"),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=args.lr, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=args.weight_decay)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=args.lr)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=64,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.05,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.5,
            scale_reward="running",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=256,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )

    dataset = read_json(args.data_path)
    random.seed(42)

    random.shuffle(dataset)
    dataset = dataset[:40000]

    prompts = [
        {
            "prompt": f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {x["raw_prompt"]}\nOutput:',
            "original_output": x["prompt"]
        }
        for x in dataset[500:]
    ]
    eval_prompts = [
        {
            "prompt": f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {x["raw_prompt"]}\nOutput:',
            "original_output": x["prompt"]
        }
        for x in dataset[:500]
    ]
    reward_fn = create_reward_fn(args)

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=[],
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/sft')
    parser.add_argument('--aes_model_path', type=str, default='outputs/rm_aes')
    parser.add_argument('--ps_model_path', type=str, default='outputs/rm_ps')
    parser.add_argument('--data_path', type=str, required=True)
    
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--alpha', type=float, default=0.7)
    
    parser.add_argument('--save_path', type=str, default='outputs/ppo')
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--reward_batch_size', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=2000)
    parser.add_argument('--checkpoint_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--num_layers_unfrozen', type=int, default=8)

    args = parser.parse_args()
    main(args)
