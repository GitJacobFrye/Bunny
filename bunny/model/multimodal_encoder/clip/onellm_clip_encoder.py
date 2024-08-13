import logging

import torch
import torch.nn as nn
from dataclasses import dataclass

import open_clip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class UnifiedTower(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=nn.init.normal_,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # load clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        for param in self.clip.parameters():
            param.requires_grad = False
            param.data = param.data.half()
        self.clip.transformer = None

        self.image_words = 30
        self.cache_image_words = 0  # for inference

        clip_width = self.clip.visual.conv1.out_channels
        # create modal shared modules
        self.resample_layers = nn.ModuleDict()
        self.num_experts = 3
        self.num_resample_layers = 8
        for expert in range(self.num_experts):
            expert = str(expert)
            self.resample_layers[expert] = nn.ModuleList()
            resampler_params = copy.deepcopy(params)
            resampler_params.n_heads = 16
            resampler_params.dim = clip_width
            for layer_id in range(self.num_resample_layers):
                self.resample_layers[expert].append(
                    TransformerBlock(layer_id, resampler_params))

        self.conv1 = nn.ModuleDict()
        self.positional_embedding = nn.ParameterDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        self.routers = nn.ModuleDict()
        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()
        self.modals = ['image', 'video', 'audio', 'point', 'rgbd', 'rgbn', 'fmri', 'imu']
        for modal in self.modals:
            if modal in ['image', 'video', 'rgbd', 'rgbn']:
                modal_tokens = 256 + 1
                pass
            elif modal == 'audio':
                self.conv1[modal] = nn.Conv2d(
                    1, clip_width, kernel_size=(16, 16), stride=(10, 10))
                modal_tokens = 1212 + 1
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'point':
                from model.lib.point_utils import PointPatchEmbed
                self.conv1[modal] = PointPatchEmbed(
                    in_channels=6, channels=clip_width)
                modal_tokens = 1024 + 1
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'fmri':
                self.conv1[modal] = nn.Linear(15724, 8192)
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([8+1, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'imu':
                self.conv1[modal] = nn.Conv1d(
                    in_channels=6, out_channels=clip_width, kernel_size=10, bias=False)
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([391+1, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)

            self.routers[modal] = Mlp(
                clip_width, clip_width * 4, self.num_experts)

            self.resample_tokens[modal] = nn.Parameter(
                torch.empty([1, 30, resampler_params.dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)

            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(clip_width, resampler_params.dim),
                nn.LayerNorm(resampler_params.dim))

            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(resampler_params.dim, params.dim),
                nn.LayerNorm(params.dim))

            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
        # TODO: Freeze some parameters at here. Freeze LLM for pretraining and Projection for finetuning.

    # @torch.no_grad()

    def clip_encode_image(self, x, modal='image'):
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # use pretrained pos embeding for rest modalities
        pos_embedding = self.clip.visual.positional_embedding
        if modal in ['audio', 'point', 'fmri', 'imu']:
            pos_embedding = self.positional_embedding[modal]

        x = x + pos_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        # if self.clip.visual.proj is not None:
        #    x = x @ self.clip.visual.proj

        return x

    def encode_image(self, x, modal='image'):
        bsz = x.size(0)
        T = 1
        if modal in ['image']:
            # modified from CLIP
            x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        elif modal in ['audio', 'imu']:
            x = self.conv1[modal](x)
        elif modal == 'point':
            # [B, 16384, 6] -> [B, 1024, 1024, 1]
            x = self.conv1[modal](x.float()).to(x.dtype)
        elif modal in ['video', 'rgbd', 'rgbn']:
            # [B, 15, 3, 224, 224]
            B, T = x.shape[:2]
            bsz = B * T
            x = x.reshape(bsz, *x.shape[2:])
            x = self.clip.visual.conv1(x)
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            # [B, 1, 8196] -> [B, 1024, 8]
            x = x.reshape(x.size(0), self.clip.visual.conv1.out_channels, -1)

        image_feats = self.clip_encode_image(x, modal=modal)
        # take mean on time dimension
        # all inputs are reduced to [B, L, D]
        bsz = int(bsz / T)
        image_feats = image_feats.reshape(
            bsz, T, *image_feats.shape[1:]).mean(dim=1)

        image_feats = self.clip_proj1[modal](image_feats)
        image_feats = torch.cat(
            [self.resample_tokens[modal].repeat(bsz, 1, 1), image_feats], dim=1)

        # routing modalites
        # [B, L, D]->[B, L, N]
        routing_weights = self.routers[modal](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        image_feats_experts = []
        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)

            image_feats_expert = image_feats_expert[:, :self.resample_tokens[modal].size(1)]
            routing_weight = routing_weights[:, :self.resample_tokens[modal].size(
                1), expert_id]
            # [B, L, D] * [B, L, 1]
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]

            image_feats_experts.append(image_feats_expert)

        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2[modal](image_feats)

        return image_feats

    def forward(self, examples, image=None, modal='image'):
        self._destroy_kv_cache()  # training always disables kv cache
        modal = modal[0]
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        start_pos = 0
        prefix_len = 0
        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image, modal)
            h = torch.cat((h_bos, self.start_tag[modal].expand(
                _bsz, -1, -1), image_tokens, self.end_tag[modal].expand(_bsz, -1, -1), h_caption), dim=1)
            # bos + image token + start_tag[modal], end_tag[modal] is used for caption generation
            prefix_len = image_tokens.shape[1] + 1 + 1
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, prefix_len:, :])
        return output

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None, modal='image'):
        modal = modal[0] if isinstance(modal, list) else modal
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            # kv cache will not re-allocate if size is unchanged
            self._allocate_kv_cache(_bsz)
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image, modal)
            self.cache_image_words = image_tokens.shape[1]
            h = torch.cat((h_bos, self.start_tag[modal].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal].repeat(_bsz, 1, 1), h_caption), dim=1)
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
        else:
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(
                max_batch_size, self.params.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()


class UniversalEncoder(nn.Module):
    def __init__(self, tower_path ,params: ModelArgs, delay_load=False):
        super().__init__()
        self.tower_path = tower_path
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=nn.init.normal_,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # load clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        for param in self.clip.parameters():
            param.requires_grad = False
            param.data = param.data.half()
        self.clip.transformer = None

        self.image_words = 30
        self.cache_image_words = 0  # for inference

        clip_width = self.clip.visual.conv1.out_channels
        # create modal shared modules
        self.resample_layers = nn.ModuleDict()
        self.num_experts = 3
        self.num_resample_layers = 8
        for expert in range(self.num_experts):
            expert = str(expert)
            self.resample_layers[expert] = nn.ModuleList()
            resampler_params = copy.deepcopy(params)
            resampler_params.n_heads = 16
            resampler_params.dim = clip_width
            for layer_id in range(self.num_resample_layers):
                self.resample_layers[expert].append(
                    TransformerBlock(layer_id, resampler_params))

        self.conv1 = nn.ModuleDict()
        self.positional_embedding = nn.ParameterDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        self.routers = nn.ModuleDict()
        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()
        self.modals = ['image', 'video', 'audio', 'point', 'rgbd', 'rgbn', 'fmri', 'imu']
        for modal in self.modals:
            if modal in ['image', 'video', 'rgbd', 'rgbn']:
                modal_tokens = 256 + 1
                pass
            elif modal == 'audio':
                self.conv1[modal] = nn.Conv2d(
                    1, clip_width, kernel_size=(16, 16), stride=(10, 10))
                modal_tokens = 1212 + 1
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'point':
                from model.lib.point_utils import PointPatchEmbed
                self.conv1[modal] = PointPatchEmbed(
                    in_channels=6, channels=clip_width)
                modal_tokens = 1024 + 1
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'fmri':
                self.conv1[modal] = nn.Linear(15724, 8192)
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([8+1, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'imu':
                self.conv1[modal] = nn.Conv1d(
                    in_channels=6, out_channels=clip_width, kernel_size=10, bias=False)
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([391+1, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)

            self.routers[modal] = Mlp(
                clip_width, clip_width * 4, self.num_experts)

            self.resample_tokens[modal] = nn.Parameter(
                torch.empty([1, 30, resampler_params.dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)

            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(clip_width, resampler_params.dim),
                nn.LayerNorm(resampler_params.dim))

            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(resampler_params.dim, params.dim),
                nn.LayerNorm(params.dim))

            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))

    def load_from_tf_save(self, state_dict):
        # Loading from state_dict and save to path with .pth
        pass

    def load_model(self):
        logging.info(f"Loading onellm universal encoder from {self.tower_path}")
        msg = self.load_state_dict(self.tower_path, strict=False)
        logging.info(msg)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, x, modal="image"):
        bsz = x.size(0)
        T = 1
        if modal in ['image']:
            # modified from CLIP
            x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        elif modal in ['audio', 'imu']:
            x = self.conv1[modal](x)
        elif modal == 'point':
            # [B, 16384, 6] -> [B, 1024, 1024, 1]
            x = self.conv1[modal](x.float()).to(x.dtype)
        elif modal in ['video', 'rgbd', 'rgbn']:
            # [B, 15, 3, 224, 224]
            B, T = x.shape[:2]
            bsz = B * T
            x = x.reshape(bsz, *x.shape[2:])
            x = self.clip.visual.conv1(x)
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            # [B, 1, 8196] -> [B, 1024, 8]
            x = x.reshape(x.size(0), self.clip.visual.conv1.out_channels, -1)

        image_feats = self.clip_encode_image(x, modal=modal)
        # take mean on time dimension
        # all inputs are reduced to [B, L, D]
        bsz = int(bsz / T)
        image_feats = image_feats.reshape(
            bsz, T, *image_feats.shape[1:]).mean(dim=1)

        image_feats = self.clip_proj1[modal](image_feats)
        image_feats = torch.cat(
            [self.resample_tokens[modal].repeat(bsz, 1, 1), image_feats], dim=1)

        # routing modalites
        # [B, L, D]->[B, L, N]
        routing_weights = self.routers[modal](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        image_feats_experts = []
        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)

            image_feats_expert = image_feats_expert[:, :self.resample_tokens[modal].size(1)]
            routing_weight = routing_weights[:, :self.resample_tokens[modal].size(
                1), expert_id]
            # [B, L, D] * [B, L, 1]
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]

            image_feats_experts.append(image_feats_expert)

        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2[modal](image_feats)

        return image_feats

    def clip_encode_image(self, x, modal='image'):
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # use pretrained pos embeding for rest modalities
        pos_embedding = self.clip.visual.positional_embedding
        if modal in ['audio', 'point', 'fmri', 'imu']:
            pos_embedding = self.positional_embedding[modal]

        x = x + pos_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        # if self.clip.visual.proj is not None:
        #    x = x @ self.clip.visual.proj

        return x

    @property
    def dummy_feature(self):
        pass
    @property
    def dtype(self):
        pass
    @property
    def device(self):
        pass
    @property
    def config(self):
        pass
    @property
    def hidden_size(self):
        pass

    @property
    def num_patches(self):
        pass