from typing import Optional
from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL
from diffusers.models.attention_processor import Attention, AttnProcessor
from einops import rearrange


class CrossFrameAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block=False,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
        )
    

    @classmethod
    def from_unet_attention(cls, attn: Attention):
        state_dict = attn.state_dict()
        inner_dim, query_dim = state_dict["to_q.weight"].shape
        cross_attention_dim = state_dict["to_k.weight"].shape[1]
        heads = attn.heads
        dim_head = inner_dim // heads
        cross_frame_attn = cls(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head
        )
        cross_frame_attn.load_state_dict(state_dict)
        cross_frame_attn.to(
            device=state_dict["to_q.weight"].device,
            dtype=state_dict["to_q.weight"].dtype,
        )
        return cross_frame_attn
    
    
    @classmethod
    def from_vae_attention(cls, attn: Attention):
        state_dict = attn.state_dict()
        inner_dim, query_dim = state_dict["to_q.weight"].shape
        cross_attention_dim = state_dict["to_k.weight"].shape[1]
        heads = attn.heads
        dim_head = inner_dim // heads
        cross_frame_attn = cls(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            bias=True,
            upcast_softmax=True,
            norm_num_groups=32,
            eps=1e-06,
            residual_connection=True,
            _from_deprecated_attn_block=True
        )
        cross_frame_attn.load_state_dict(state_dict)
        cross_frame_attn.to(
            device=state_dict["to_q.weight"].device,
            dtype=state_dict["to_q.weight"].dtype,
        )
        return cross_frame_attn
        

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        if encoder_hidden_states is not None:
            raise Warning("encoder_hidden_states is not None in CrossFrameAttention")
        B = hidden_states.shape[0]
        if len(hidden_states.shape)==3:
            hidden_states = rearrange(hidden_states, "B N D -> 1 (B N) D")
            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = rearrange(hidden_states, "1 (B N) D -> B N D", B=B)
        elif len(hidden_states.shape)==4:
            hidden_states = rearrange(hidden_states, "B D H W -> 1 D H (B W)")
            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = rearrange(hidden_states, "1 D H (B W) -> B D H W", B=B)
        else:
            raise ValueError(f"The shape of hidden_states is {hidden_states.shape}.")
        return hidden_states


def set_cross_frame_attention_unet(model):
    for module_name, module in model.named_children():
        if isinstance(module, Attention) and module_name == "attn1":
            setattr(model, module_name, CrossFrameAttention.from_unet_attention(module))
        else:
            set_cross_frame_attention_unet(getattr(model, module_name))
            
            
def set_cross_frame_attention_vae(model):
    for module_name, module in model.named_children():
        if isinstance(module, Attention):
            setattr(model, module_name, CrossFrameAttention.from_vae_attention(module))
        else:
            set_cross_frame_attention_vae(getattr(model, module_name))


def set_cross_frame_attention(model):
    if isinstance(model, UNet2DConditionModel) or isinstance(model, ControlNetModel):
        set_cross_frame_attention_unet(model)
    elif isinstance(model, AutoencoderKL):
        set_cross_frame_attention_vae(model)
    else:
        raise Warning("Unsupported model architecture.")
