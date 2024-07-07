from dataclasses import dataclass
import torch
import torch.nn as nn


class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    dim: int = 1024


class Attention(nn.Module): ...


class Block(nn.Module): ...


class MLP(nn.Module): ...


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        # embedding layer

        # create positional encoding

        # multi-head attention

        # ffn

        # softmax

        ...

    @classmethod
    def from_pretrained(self, weight):
        ...


    def forward(self): ...
