from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layers: int = 12
    n_head: int = 12
    dim: int = 1024
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # embedding layer, translates input ids to embedding representation
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                # positional encoding layer
                wpe=nn.Embedding(config.block_size, config.n_embed),
                # multi-head attention layers
                h=nn.ModuleList(),
                # layer normalization
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

        # the LM_HEAD, that converts the internal representation to logits
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel

        hf_model = GPT2LMHeadModel.from_pretrained()
        hf_model.state_dict()

    def forward(self): ...


class Block(nn.Module):
    """A block defines the single block of self-attention and mlp.
    In GPT-2 architecture, the attention block is repeated n_layer times.

    input dim: (batch_size, seq_size, n_embed)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # x adds the output of layernorm and attention layer
        # this creates a residual connection
        x = x + self.attn(self.ln_1(x))

        # same here, passing through the mlp layer with layernorm and residual
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """The Causal Self-Attention layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # register buffer for bias?
        # is actually the attention mask to mask future tokens during training process
        self.register_buffer("bias", torch.tril(
            torch.ones(
                (config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
        ))

    def forward(self, x):
        # batchsize, sequence size, number of channels
        # calculate the q, k, v for all heads in batch
        B, T, C = x.size()

        # project the input into attention head space, output of size (B, T, 3C)
        qkv = self.c_attn(x)

        # split the input into three attention heads of Q, K, V
        q, k, v = qkv.split(self.n_embed, dim=2)

        # transform each head to multi-heads, defined by nh:
        # (B, T, C) split to -> (B, T, nh, hs), transpose it to (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # calculate attention with mask
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # reassemble the heads into contiguous memory space as a single matrix
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP works as a classification layer after the attention output.
    The hidden layer is of 4 * n_embed, with a GELU for activation.

    This layer expands to a wider dimension to increase the capacity
    of the network, allowing it to learn more features.

    input dim: (batch_size, seq_size, n_embed)
    output dim: (batch_size, seq_size, n_embed)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # fully-connected layer
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)

        # activation layer
        self.gelu = nn.GELU(approximate="tanh")

        # projection layer
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
