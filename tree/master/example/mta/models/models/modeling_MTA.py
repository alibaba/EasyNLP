import copy
import math
import warnings
from typing import Optional, Tuple, Union
import sys
sys.path.append('models/switch_transformers')

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,functional
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from models.models.configuration import MTAConfig
from transformers.models.t5.modeling_t5 import T5DenseActDense ,T5DenseGatedActDense,T5LayerNorm

logger = logging.get_logger(__name__)

def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)



ALL_LAYERNORM_LAYERS.append(T5LayerNorm)


class T5DenseActDense(nn.Module):
    def __init__(self, config: MTAConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]   # 激活函数

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: MTAConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class MTA_averagesoftmaxt(nn.Module):  # 每个任务对应一个expert
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_averagesoftmaxt")
        
        self.experts = nn.ModuleDict()
        self.gate_for_classify = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_nli = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_mrc = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_generate = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_anaphora_resolution = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))

        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        hidden_states_list = []
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            next_states = expert(hidden_states)
            hidden_states_list.append(next_states)

        out_list = []
        temp = 0.1
        for i in range(len(type_label)):
            output = []
            temp_list = []
            if type_label[i] == "classify":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_classify/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] == "nli":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_nli/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] == "mrc":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_mrc/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] == "generate":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_generate/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] == "anaphora_resolution":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_anaphora_resolution/temp,dim=0)[k]))
                out_list.insert(i,sum(output))      
        hidden_states = torch.stack(out_list)   # list2tensor

        return hidden_states

class MTA_noisegate(nn.Module):  # 每个任务对应一个expert
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    # def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_noisegate")
        
        self.experts = nn.ModuleDict()
        
        # # weight1
        self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.22,0.2,0.2,0.2)))
        self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.2,0.22,0.22,0.2,0.2)))
        self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.22,0.22)))

        # # weight2
        # self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.2,0.2,0.2,0.2)))
        # self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.2,0.2,0.22,0.2,0.2)))
        # self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.22)))

        # self.gate_for_mrc =                 torch.nn.Parameter(torch.tensor((0.2,0.2,0.22,0.22,0.2)))
        # self.gate_for_anaphora_resolution = torch.nn.Parameter(torch.tensor((0.22,0.2,0.2,0.2,0.22)))

        # self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.22,0.2)))
        # self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.2,0.22,0.22)))
        # self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.22,0.2,0.22)))

        # self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.22,0.2,0.2,0.2,0.21)))
        # self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.2,0.22,0.22,0.2,0.2,0.25)))
        # self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.22,0.22,0.24)))

        # self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.22,0.20,0.20)))
        # self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.20,0.22,0.22,0.20)))
        # self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.20,0.20,0.22,0.22)))

        # self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.2,0.2)))
        # self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.2,0.22,0.2)))
        # self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.2,0.2,0.22)))

        # self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.20)))
        # self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.22,0.20)))
        # self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.20,0.22)))

        print("----",config.num_experts)
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        hidden_states_list = []
        # next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            next_states = expert(hidden_states)
            hidden_states_list.append(next_states)

        out_list = []
        temp = 0.1
        for i in range(len(type_label)):
            output = []
            temp_list = []
            if type_label[i] == "classify":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_classify/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] == "nli":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_nli/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] == "generate":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_generate/temp,dim=0)[k]))
                out_list.insert(i,sum(output))

        hidden_states = torch.stack(out_list)   # list2tensor

        return hidden_states

class MTA_sharemoe(nn.Module):  # 类似的任务同一个experts,总分成三类任务
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_share_moe")
        
        # 定义特定任务专家
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

        # 定义共享专家
        self.share_expert = nn.ModuleDict()
        self.share_expert["expert_0"] = expert_class(config)

        # gating network
        self.gate = nn.Sequential(nn.Linear(config.my_hidden_size*2,config.my_hidden_size,bias=False),    # 随机初始化
                                    nn.ReLU(),
                                    nn.Linear(config.my_hidden_size,2,bias=False),                        # 随机初始化
                                    nn.Softmax(dim=-1)
                                    )
        self.gates = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.gates[f"gate_{idx}"] = self.gate

        
    def forward(self,hidden_states,type_label):
        r"""
        type_label in ["classify","nli","generate","mrc","anaphora_resolution"]
        """
        type_list = ["classify","nli","generate","mrc","anaphora_resolution"]

        share_experts_output = self.share_expert["expert_0"](hidden_states)
        hidden_states_list = []

        for i in range(len(type_label)):
            idx = type_list.index(type_label[i])

            expert_out = self.experts[f"expert_{idx}"](hidden_states[i])
            gate_inputs = torch.cat((expert_out[0],share_experts_output[i][0]),0)
            weights = self.gates[f"gate_{idx}"](gate_inputs)

            hidden_states_list.append((expert_out*weights[0]+share_experts_output[i]*weights[1]))
        
        hidden_states = torch.stack(hidden_states_list)

        return hidden_states

class MTA_sharemoe2(nn.Module):  # 类似的任务同一个experts,总分成三类任务
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    # def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_share_moe2")
        
        # 定义特定任务专家
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        # 定义共享专家
        self.share_expert = nn.ModuleDict()
        for idx in range(2):
            self.share_expert[f"expert_{idx}"] = expert_class(config)

        # # gating network
        # self.gate = nn.Sequential(nn.Linear(config.my_hidden_size*3,config.my_hidden_size,bias=False),    # 随机初始化
        #                             nn.ReLU(),
        #                             nn.Linear(config.my_hidden_size,3,bias=False),                        # 随机初始化
        #                             nn.Softmax(dim=-1)
        #                             )
        self.gate = nn.Sequential(nn.Linear(config.d_model*3,config.d_model,bias=False),    # 随机初始化
                                    nn.ReLU(),
                                    nn.Linear(config.d_model,3,bias=False),                        # 随机初始化
                                    nn.Softmax(dim=-1)
                                    )

        self.gates = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.gates[f"gate_{idx}"] = self.gate

        
    def forward(self, hidden_states,type_label):
        r"""
        type_label in ["classify","nli","generate","mrc","anaphora_resolution"]
        """
        type_list = ["classify","nli","generate","mrc","anaphora_resolution"]

        share_experts_output_0 = self.share_expert["expert_0"](hidden_states)
        share_experts_output_1 = self.share_expert["expert_1"](hidden_states)
        # share_experts_output = share_experts_output_0 + share_experts_output_1
        hidden_states_list = []

        for i in range(len(type_label)):
            idx = type_list.index(type_label[i])

            expert_out = self.experts[f"expert_{idx}"](hidden_states[i])
            gate_inputs = torch.cat((expert_out[0],share_experts_output_0[i][0],share_experts_output_1[i][0]),0)
            weights = self.gates[f"gate_{idx}"](gate_inputs)

            hidden_states_list.append((expert_out*weights[0]+share_experts_output_0[i]*weights[1]+share_experts_output_1[i]*weights[2]))
        
        hidden_states = torch.stack(hidden_states_list)

        return hidden_states



class MTA_sharemoe3(nn.Module):  # 类似的任务同一个experts,总分成三类任务
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    # def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_share_moe3")
        
        # 定义特定任务专家
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        # 定义共享专家
        self.share_expert = nn.ModuleDict()
        for idx in range(3):
            self.share_expert[f"expert_{idx}"] = expert_class(config)

        # gating network
        self.gate = nn.Sequential(nn.Linear(config.my_hidden_size*4,config.my_hidden_size,bias=False),    # 随机初始化
                                    nn.ReLU(),
                                    nn.Linear(config.my_hidden_size,4,bias=False),                        # 随机初始化
                                    nn.Softmax(dim=-1)
                                    )
        self.gates = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.gates[f"gate_{idx}"] = self.gate

        
    def forward(self, hidden_states,type_label):
        r"""
        type_label in ["classify","nli","generate","mrc","anaphora_resolution"]
        """
        type_list = ["classify","nli","generate","mrc","anaphora_resolution"]

        share_experts_output_0 = self.share_expert["expert_0"](hidden_states)
        share_experts_output_1 = self.share_expert["expert_1"](hidden_states)
        share_experts_output_2 = self.share_expert["expert_2"](hidden_states)
        # share_experts_output = share_experts_output_0 + share_experts_output_1 + share_experts_output_2
        hidden_states_list = []

        for i in range(len(type_label)):
            idx = type_list.index(type_label[i])

            expert_out = self.experts[f"expert_{idx}"](hidden_states[i])
            gate_inputs = torch.cat((expert_out[0],share_experts_output_0[i][0],share_experts_output_1[i][0],share_experts_output_2[i][0]),0)
            weights = self.gates[f"gate_{idx}"](gate_inputs)

            hidden_states_list.append((expert_out*weights[0]+share_experts_output_0[i]*weights[1]+share_experts_output_1[i]*weights[2]+share_experts_output_2[i]*weights[3]))
        
        hidden_states = torch.stack(hidden_states_list)

        return hidden_states

class MTA_noisegatev3(nn.Module):  # 类似的任务同一个experts,总分成三类任务
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_noisegatev3")
        
        self.experts = nn.ModuleDict()
        self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.2,0.22)))
        self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.22,0.2)))

        for idx in range(2):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        hidden_states_list = []
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            next_states = expert(hidden_states)
            hidden_states_list.append(next_states)

        out_list = []
        temp = 0.1
        for i in range(len(type_label)):
            output = []
            temp_list = []
            if type_label[i] == "generate":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_generate/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
            elif type_label[i] : #   ["classify","anaphora_resolution","nli","mrc"]
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(functional.softmax(self.gate_for_classify/temp,dim=0)[k]))
                out_list.insert(i,sum(output))
      
        hidden_states = torch.stack(out_list)   # list2tensor

        return hidden_states

class MTA_StableWeight(nn.Module):  # 每个任务对应一个expert
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_StableWeight")
        
        self.gate_for_classify =            torch.tensor((0.6,0.1,0.1,0.1,0.1))
        self.gate_for_nli =                 torch.tensor((0.1,0.6,0.1,0.1,0.1))
        self.gate_for_mrc =                 torch.tensor((0.1,0.1,0.6,0.1,0.1))
        self.gate_for_generate =            torch.tensor((0.1,0.1,0.1,0.6,0.1))
        self.gate_for_anaphora_resolution = torch.tensor((0.1,0.1,0.1,0.1,0.6))

        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        hidden_states_list = []
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            next_states = expert(hidden_states)
            hidden_states_list.append(next_states)

        out_list = []
        for i in range(len(type_label)):
            output = []
            temp_list = []
            if type_label[i] == "classify":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_classify[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "nli":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_nli[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "mrc":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_mrc[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "generate":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_generate[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "anaphora_resolution":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_anaphora_resolution[k],dim=0))
                out_list.insert(i,sum(output))      
        hidden_states = torch.stack(out_list)   # list2tensor

        return hidden_states

class T5SparseMLP(nn.Module):  # 每个任务对应一个expert
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:T5SparseMLP")
        
        self.gate_for_classify =            torch.tensor((0.2,0.2,0.2,0.2,0.2))
        self.gate_for_nli =                 torch.tensor((0.2,0.2,0.2,0.2,0.2))
        self.gate_for_mrc =                 torch.tensor((0.2,0.2,0.2,0.2,0.2))
        self.gate_for_generate =            torch.tensor((0.2,0.2,0.2,0.2,0.2))
        self.gate_for_anaphora_resolution = torch.tensor((0.2,0.2,0.2,0.2,0.2))

        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        hidden_states_list = []
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            next_states = expert(hidden_states)
            hidden_states_list.append(next_states)

        out_list = []
        for i in range(len(type_label)):
            output = []
            temp_list = []
            if type_label[i] == "classify":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(self.gate_for_classify[k]/sum(self.gate_for_classify)))
                out_list.insert(i,sum(output))
            elif type_label[i] == "nli":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(self.gate_for_nli[k]/sum(self.gate_for_nli)))
                out_list.insert(i,sum(output))
            elif type_label[i] == "mrc":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(self.gate_for_mrc[k]/sum(self.gate_for_mrc)))
                out_list.insert(i,sum(output))
            elif type_label[i] == "generate":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(self.gate_for_generate[k]/sum(self.gate_for_generate)))
                out_list.insert(i,sum(output))
            elif type_label[i] == "anaphora_resolution":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*(self.gate_for_anaphora_resolution[k]/sum(self.gate_for_anaphora_resolution)))
                out_list.insert(i,sum(output))      
        hidden_states = torch.stack(out_list)   # list2tensor
        # print(self.gate_for_generate)
        return hidden_states

class MTA_averageweights(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """
    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_averageweights")
        
        self.experts = nn.ModuleDict()
        self.gate_for_classify = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_nli = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_mrc = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_generate = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))
        self.gate_for_anaphora_resolution = torch.nn.Parameter(torch.tensor((0.2,0.2,0.2,0.2,0.2)))

        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        hidden_states_list = []
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            next_states = expert(hidden_states)
            hidden_states_list.append(next_states)

        out_list = []
        for i in range(len(type_label)):
            output = []
            temp_list = []
            if type_label[i] == "classify":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_classify[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "nli":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_nli[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "mrc":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_mrc[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "generate":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_generate[k],dim=0))
                out_list.insert(i,sum(output))
            elif type_label[i] == "anaphora_resolution":
                for j in range(len(hidden_states_list)):
                    temp_list.insert(j,hidden_states_list[j][i])
                for k in range(len(temp_list)):
                    output.insert(k,temp_list[k]*functional.softmax(self.gate_for_anaphora_resolution[k],dim=0))
                out_list.insert(i,sum(output))      
        hidden_states = torch.stack(out_list)   # list2tensor

        return hidden_states

class MTA_sharemoe4(nn.Module):  # 类似的任务同一个experts,总分成三类任务
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseActDense):
    # def __init__(self, config: SwitchTransformersConfig, expert_class: nn.Module = SwitchTransformersDenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_share_moe4")
        
        # 定义特定任务专家
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        # 定义共享专家
        self.share_expert = nn.ModuleDict()
        for idx in range(3):
            self.share_expert[f"expert_{idx}"] = expert_class(config)

        self.gate_for_classify =            torch.nn.Parameter(torch.tensor((0.22,0.22,0.20,0.20)))
        self.gate_for_nli =                 torch.nn.Parameter(torch.tensor((0.20,0.22,0.22,0.20)))
        self.gate_for_generate =            torch.nn.Parameter(torch.tensor((0.20,0.20,0.22,0.22)))

        # gating network
        self.gate = nn.Sequential(nn.Linear(config.my_hidden_size*4,config.my_hidden_size,bias=False),    # 随机初始化
                                    nn.ReLU(),
                                    nn.Linear(config.my_hidden_size,4,bias=False),                        # 随机初始化
                                    nn.Softmax(dim=-1)
                                    )
        self.gates = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.gates[f"gate_{idx}"] = self.gate

        
    def forward(self, hidden_states,type_label):
        r"""
        type_label in ["classify","nli","generate","mrc","anaphora_resolution"]
        """
        type_list = ["classify","nli","generate","anaphora_resolution"]
        type_dict = {0:self.gate_for_classify,1:self.gate_for_nli,2:self.gate_for_generate}
        # type_list = ["classify","nli","mrc","generate","anaphora_resolution"]

        share_experts_output_0 = self.share_expert["expert_0"](hidden_states)
        share_experts_output_1 = self.share_expert["expert_1"](hidden_states)
        share_experts_output_2 = self.share_expert["expert_2"](hidden_states)
        # share_experts_output = share_experts_output_0 + share_experts_output_1 + share_experts_output_2
        hidden_states_list = []
        expert_out = 0
        for i in range(len(type_label)):
            # idx = type_dict[type_label[i]]
            idx = type_list.index(type_label[i])
            
            sum_out = [self.experts[f"expert_{idx}"](hidden_states[i]),self.experts[f"expert_{idx+1}"](hidden_states[i])]
            weight_out = [functional.softmax(type_dict[idx]/0.1,dim=0)[idx],functional.softmax(type_dict[idx]/0.1,dim=0)[idx+1]]

            expert_out = sum_out[0] * weight_out[0] + sum_out[1] * weight_out[1]

            gate_inputs = torch.cat((expert_out[0],share_experts_output_0[i][0],share_experts_output_1[i][0],share_experts_output_2[i][0]),0)
            weights = self.gates[f"gate_{idx}"](gate_inputs)

            hidden_states_list.append((expert_out*weights[0]+share_experts_output_0[i]*weights[1]+share_experts_output_1[i]*weights[2]+share_experts_output_2[i]*weights[3]))
        
        hidden_states = torch.stack(hidden_states_list)

        return hidden_states



class MTA_lineargate(nn.Module):  
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_lineargate")
        
        self.gate = nn.Sequential(nn.Linear(config.my_hidden_size,config.my_hidden_size, bias=False),
                                    nn.Softmax(dim=-1)
                                    )

        self.experts = nn.ModuleDict()

        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        print("num_experts = ",config.num_experts)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        cls_hidden_state = hidden_states[:,0]
        gate_value = self.gate(cls_hidden_state)   # torch.size([4,5])

        expert_output = 0
        for i, expert in enumerate(self.experts.values()):
            expert_output = expert_output + gate_value[:,i].unsqueeze(1).unsqueeze(1) * expert(hidden_states)

        hidden_states = expert_output    
        
        return hidden_states

class MTA_gatenetwork(nn.Module):  # 每个任务对应一个gate_network
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MTAConfig, expert_class: nn.Module = T5DenseGatedActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        # Step 2: Get the experts
        print("MLP_type:MTA_gatenetwork")
        
        self.gate = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.gate[f"gate_{idx}"] = nn.Sequential(nn.Linear(config.my_hidden_size,config.my_hidden_size, bias=False),
                                                        nn.ReLU(),
                                                        nn.Linear(config.my_hidden_size,config.num_experts, bias=False),
                                                        nn.Softmax(dim=-1)
                                                        )
        self.gate_value = {}
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
        print("num_experts = ",config.num_experts)

    def forward(self, hidden_states,type_label):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        cls_hidden_state = hidden_states[:,0]

        expert_output = 0
        for i, expert in enumerate(self.experts.values()):
            expert_output = expert_output + self.gate[f"gate_{i}"](cls_hidden_state)[:,i].unsqueeze(1).unsqueeze(1) * expert(hidden_states)

        hidden_states = expert_output    
        
        return hidden_states

class T5DenseActDense(nn.Module):
    def __init__(self, config: MTAConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: MTAConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class MTALayers(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`MTAConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        is_sparse (`bool`):
            Whether the MLP layer is a `Sparse` layer (contains a Mixture of Experts) or not
    """

    def __init__(self, config: MTAConfig, is_sparse=False):
        super().__init__()

        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)        

        self.is_sparse = is_sparse
        self.mlp = MTA_noisegate(config)   
        '''
        type:
        T5SparseMLP
        MTA_StableWeight
        MTA_noisegate
        MTA_averageweights
        MTA_sharemoe
        MTA_sharemoe2
        MTA_sharemoe3
        MTA_averagesoftmaxt
        MTA_noisegatev3
        MTA_noisegatev2
        MTA_lineargate
        MTA_gatenetwork
        '''
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states,type_label):
        forwarded_states = self.layer_norm(hidden_states)

        # for generate beam search
        temp_list = []
        if forwarded_states.shape[0] == len(type_label):
            forwarded_states = self.mlp(forwarded_states,type_label) 
        else:
            temp_list = copy.deepcopy(type_label)
            for i in range(len(type_label)):
                for j in range(int(forwarded_states.shape[0]/len(type_label))-1):
                    temp_list.insert(i*int(forwarded_states.shape[0]/len(type_label)),temp_list[i*int(forwarded_states.shape[0]/len(type_label))+j])
            forwarded_states = self.mlp(forwarded_states,temp_list) 
        output = hidden_states + self.dropout(forwarded_states)

        return output

