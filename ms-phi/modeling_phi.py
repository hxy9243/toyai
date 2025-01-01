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

        self.max_position_embeddings = config.max_position_embeddings
        self.model = nn.ModuleDict(
            dict(
                embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
                layers=nn.ModuleList([DecodeLayer(config) for _ in range(config.n_layers)]),
                lm_head=nn.Linear(config.hidden_size, config.vocab_size, bias=False),
            )
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward the input through the transformer layer, return logits.
        input shape (Batch_size, Sequence size)
        return shape (Batch_size, 1)
        """
        B, S = x.size()
        assert S <= self.max_position_embeddings

        x = self.model.embed_tokens(x)

        for layer in self.model.layers:
            x = layer(x)

        return self.model.lm_head(x)


class Phi3RMSNorm(nn.Module):
    """ Phi3 uses a Llama RMS Norm: layer normalizes by dividing the stddev of the sequence.
    """
    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.rms_norm_eps = config.rms_norm_eps

        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forwards(self, x):
        # TODO: consider upscaling for better float precision

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)

        return self.weight(x)


class DecodeLayer(nn.Module):

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.input_layernorm = Phi3RMSNorm(config)
        self.post_attention_layernorm = Phi3RMSNorm(config)

        self.self_attn = AttentionLayer(config)
        self.mlp = MLP(config)

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

        self.qkv_proj = nn.Linear(config.hidden_size, 3*config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class MLP(nn.Module):
    """ Phi MLP uses a gated activation that:
    - has a gate signal that controls the flow of information
    - uses silu for the MLP activation
    """

    def __init__(self, config: PhiLMConfig):
        super().__init__()

        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        gate, up_states = x.chunk(2, dim=-1)

        act = nn.SiLU()

        up_states = up_states * act(gate)
        return up_states
