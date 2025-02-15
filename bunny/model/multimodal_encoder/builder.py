import os

from .clip.onellm_clip_encoder import OneLLMVisionTower
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .clip.clip_encoder import CLIPVisionTower

from dataclasses import dataclass
import json

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)
    # print(f"vision_tower: {vision_tower}")
    # print(f"use_s2: {use_s2}")

    if 'sig' in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'eva' in vision_tower.lower():
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for EVA-CLIP')
        else:
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'clip' in vision_tower.lower():
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for CLIP')
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # Add by zyh
    elif 'onellm' in vision_tower.lower():
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for oneLLM')
        else:
            llama_config = "/cpfs01/projects-HDD/cfff-4a6c654d10ce_HDD/zyh_23210440066/Bunny_version_816/bunny/config/llama2/7B.json"
            with open(llama_config, "r") as f:
                params = json.loads(f.read())
            params: ModelArgs = ModelArgs(
                max_seq_len=2048, max_batch_size=32, **params
            )
            return OneLLMVisionTower(vision_tower, params, **kwargs)

    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
