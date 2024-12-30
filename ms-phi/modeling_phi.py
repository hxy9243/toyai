from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhiLMConfig:
    """ Model the configuration from here: https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi/configuration_phi.py
    """

    vocab_size: int = 32064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    hidden_act: str = 'silu'
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: int = 10000.0
    bos_token_id: int = 1
    eos_token_id: int = 32000
    pad_token_id: int = 32000


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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward the input through the transformer layer, return logits.
        input shape (Batch_size, Sequence size)
        return shape (Batch_size, 1)
        """
        B, S = x.size()
        assert S <= self.config.max_position_embeddings

        x = self.model.embed_tokens(x)

        for layer in self.model.layers:
            x = layer(x)

        return self.model.lm_head(x)


class Phi3RMSNorm(nn.Module):
    """ Phi3 uses a Llama RMS Norm: layer normalizes by dividing the stddev of the sequence.
    """

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.config = config
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forwards(self, x):
        # TODO: consider upscaling for better float precision

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.config.rms_norm_eps)

        return self.weight(x)


class DecodeLayer(nn.Module):

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.input_layernorm = Phi3RMSNorm(self.config)
        self.post_attention_layernorm = Phi3RMSNorm(self.config)

        self.self_attn = AttentionLayer(self.config)
        self.mlp = None

    def forward(self, x) -> torch.Tensor:
        # go through self attention layer
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        # go through mlp layer
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

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

