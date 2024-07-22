from typing import List, Generator, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                # layer normalization
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

        # the LM_HEAD, that converts the internal representation to logits
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt: {model_type}")
        config_args = {
            "gpt2": dict(n_layers=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layers=12, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layers=36, n_head=20, n_embed=1280),
            "gpt2-xl": dict(n_layers=48, n_head=25, n_embed=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # init our GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard mask/buff

        # load GPT2 model pretrained weights from huggingface
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()
        # discard the masks
        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if (not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias"))
        ]

        # we need to transpose certain weights to match the GPT-2 weights
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys)

        print("copying model from pretrained to model")
        for k in sd_keys_hf:
            print(f"copying layer {k}")
            if any(k.endswith(w) for w in transposed):
                # assert it's transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"Key {k} has different shape, HF: {sd_hf[k].shape}, model: {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, x: torch.tensor) -> torch.tensor:
        # input is of size (B, T)
        B, T = x.size()
        assert T <= self.config.block_size

        # create embedding from token embedding + positional embedding
        # input size (B, T, internal)
        # output size (B, T, internal)
        pos = torch.arange(0, T, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(x)
        x = pos_emb + tok_emb

        # go through n_layers of blocks of attention + MLP
        # input size (B, T, C)
        # output size (B, T, C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # go through LM head to translate embedding back to vocab
        # input size (B, T, C)
        # output size (B, T, vocab_size)
        logits = self.lm_head(x)

        # return logits of shape (B, T, vocab_size)
        return logits

    def generate(
        self,
        prompt: str,
        num_return_tokens=64,
        temperature=0.7,
        sampling=False,
        topk=10,
    ) -> Generator[str, None, None]:
        from transformers import AutoTokenizer

        enc = AutoTokenizer.from_pretrained("gpt2")
        x = enc.encode(prompt, return_tensors="pt")

        eps = 1e-4
        while x.size(1) < num_return_tokens:
            with torch.no_grad():
                logits = self.forward(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / (temperature + eps), dim=-1)

                if not sampling:
                    argmax = torch.argmax(probs, dim=-1)
                    xcol = argmax.unsqueeze(1)
                    x = torch.cat((x, xcol), dim=1)
                    yield enc.decode(xcol[-1].tolist())
                else:
                    topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)

                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, xcol), dim=1)
                    yield enc.decode(xcol[-1].tolist())

    def __call__(self, x: str) -> str:
        return self.generate(x)


class Block(nn.Module):
    """A block defines the single block of self-attention and mlp.
    In GPT-2 architecture, the attention block is repeated n_layer times.

    input dim: (batch_size, seq_size, n_embed)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.tensor):
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
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            ),
        )

    def forward(self, x: torch.tensor):
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
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # reassemble the heads into contiguous memory space as a single matrix
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
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
