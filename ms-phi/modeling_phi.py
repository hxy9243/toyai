from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhiLMConfig:
    block_size: int = 256
    vocab_size: int = 128000
    n_layers: int = 32
    n_embed: int = 3072


class PhiCausalLM(nn.Module):
    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.config = config
        self.model = nn.ModuleDict(
            dict(
                embed_tokens=nn.Embedding(config.vocab_size, config.n_embed),
                layers=nn.ModuleList([DecodeLayer(config) for _ in range(config.n_layers)]),
            )
        )


class DecodeLayer(nn.Module):

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(config.n_embed)
        self.self_attn = None
        self.mlp = None
        self.post_attention_layernorm = nn.LayerNorm(config.n_embed)


class AttentionLayer(nn.Module):

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.qkv_proj = None
        self.o_proj = None


class MLP(nn.Module):

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.gate_up_proj = None
        self.act = None
        self.down_proj = None

