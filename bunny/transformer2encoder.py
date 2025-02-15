# 从OneLLM模型中加载OneLLMVisionTower
import os
import sys
root_path = "/cpfs01/projects-HDD/cfff-4a6c654d10ce_HDD/zyh_23210440066/Bunny/"
sys.path.insert(0, root_path)

import torch
from model import UniversalEncoder
from onellm_model.meta import MetaModel
from dataclasses import dataclass
import json

# 单卡设置fairscale
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist

@dataclass
class TransferConfig:
    gpu_ids = [0]
    master_port = 23863
    master_addr = "127.0.0.1"

args = TransferConfig()
rank = 0
world_size = len(args.gpu_ids)
gpu_id = args.gpu_ids[rank]
dist.init_process_group(
    backend="nccl", rank=rank, world_size=world_size,
    init_method=f"tcp://{args.master_addr}:{args.master_port}",
)
print(f"| distributed init on worker {rank}/{world_size}. "
        f"using gpu: {gpu_id}")
fs_init.initialize_model_parallel(world_size)
torch.cuda.set_device(gpu_id)


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

meta_pth_path = "/cpfs01/projects-HDD/cfff-4a6c654d10ce_HDD/zyh_23210440066/models/OneLLM_full/consolidated.00-of-01.pth"  # pth文件路径--pretrained_path ${WEIGHTS_DIR}/consolidated.00-of-01.pth
save_path = "/cpfs01/projects-HDD/cfff-4a6c654d10ce_HDD/zyh_23210440066/models/OneLLMVisionTower/universal_encoder.pth"

llama_type = "onellm"
llama_config = "/cpfs01/projects-HDD/cfff-4a6c654d10ce_HDD/zyh_23210440066/Bunny_version_816/bunny/config/llama2/7B.json"
tokenizer_path = "/cpfs01/projects-HDD/cfff-4a6c654d10ce_HDD/zyh_23210440066/Bunny_version_816/bunny/config/llama2/tokenizer.model"


model = MetaModel(llama_type, llama_config, tokenizer_path=tokenizer_path)
print("Loading pretrained weights ...")
checkpoint = torch.load(meta_pth_path, map_location='cpu')
# msg = model.load_state_dict(checkpoint, strict=False)
# print("load result to original model:\n", msg)

with open(llama_config, "r") as f:
    params = json.loads(f.read())
model_args: ModelArgs = ModelArgs(
    max_seq_len=2048, max_batch_size=32, **params
)

# 提取transformers
# state_dict = model.llma.state_dict()

# 加载到UE中
ue = UniversalEncoder(model_args)
msg = ue.load_state_dict(checkpoint, strict=False)
print("UE loading result:\n", msg)

# 保存起来
torch.save(ue.state_dict(), save_path)

