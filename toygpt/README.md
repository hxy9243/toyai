ToyGPT
====

# Implementation

A toy implementation of GPT-2, from Karpathy's lesson: https://www.youtube.com/watch?v=l8pRSuU81PU&t=243s

Much of the code is directly copied from Karpathy's codebase, but still, a good exercise to understand
transformer architecture and implementation.

```python
# initializing
def __init__(self, ...):
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
```

The `forward()` process go through:

- Embedding Layer
- Blocks of (attention + MLP)
- LM_head

As explained in the following code snippet:

```python
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

        return logits
```