# 从OneLLM模型中加载OneLLMVisionTower

import torch
from model import UniversalEncoder
from onellm_model.meta import MetaModel
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

# OneLLM初始化

meta_pth_path = ""  # pth文件路径--pretrained_path ${WEIGHTS_DIR}/consolidated.00-of-01.pth
save_path = ""

llama_type = "onellm"
llama_config = "config/llama2/7B.json"
tokenizer_path = "config/llama2/tokenizer.model"


model = MetaModel(llama_type, llama_config, tokenizer_path=tokenizer_path)
print("Loading pretrained weights ...")
checkpoint = torch.load(meta_pth_path, map_location='cpu')
msg = model.load_state_dict(checkpoint, strict=False)
print("load result:\n", msg)

with open(llama_config, "r") as f:
    params = json.loads(f.read())
model_args: ModelArgs = ModelArgs(
    max_seq_len=2048, max_batch_size=32, **params
)

# 提取transformers
state_dict = model.llma.state_dict()

# 加载到UE中
ue = UniversalEncoder(model_args)
msg = ue.load_state_dict(state_dict, strict=False)
print("UE loading result:\n", msg)
