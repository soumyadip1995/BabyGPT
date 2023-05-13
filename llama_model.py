### from https://github.com/Lightning-AI/lit-llama partially.

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    num_layers: int = 32
    num_heads: int = 32
    embedded_dim: int = 4096


    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(num_layers=32, num_heads=32, embedded_dim=4096),
    "13B": dict(num_layers=40, num_heads=40, embedded_dim=5120),
    "30B": dict(num_layers=60, num_heads=52, embedded_dim=6656),
    "65B": dict(num_layers=80, num_heads=64, embedded_dim=8192),
}


class AttentionHead(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    assert config.embedded_dim % config.num_heads == 0


    self.atten = nn.Linear(config.embedded_dim, 3 * config.embedded_dim, bias=False)
    self.c_proj = nn.Linear(config.embedded_dim, config.embedded_dim, bias=False)
    self.num_heads = config.num_heads
    self.embedded_dim = config.embedded_dim
    self.block_size = config.block_size
    self.rope_cache: Optional[torch.Tensor] = None
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
    

  def forward(self, x):
    B,T,C = x.size()
    q, k ,v  = self.atten(x).split(self.embedded_dim, dim=2)
    q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)


    if self.rope_cache is None:
      # cache for future forward calls
      self.rope_cache = build_rope_cache(
          seq_len=self.block_size,
          n_elem=self.embedded_dim // self.num_heads, 
          dtype=x.dtype,
          device=x.device,
          )
      q = apply_rope(q, self.rope_cache)
      k = apply_rope(k, self.rope_cache)

      # from karpathy
      # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
      y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

      # output projection
      y = self.c_proj(y)
      return y


class RMSNorm(nn.Module):
  def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
    super().__init__()
    self.scale = nn.Parameter(torch.ones(size))
    self.eps = eps
    self.dim = dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + self.eps)
    return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
  # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

  # Create position indexes `[0, 1, ..., seq_len - 1]`
  seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

  # Calculate the product of position index and $\theta_i$
  idx_theta = torch.outer(seq_idx, theta).float()

  cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

  # this is to mimic the behaviour of complex32, else we will get different results
  if dtype in (torch.float16, torch.bfloat16, torch.int8):
    cache = cache.half()
  return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
  x = x.transpose(1, 2)

  # truncate to support variable sizes
  T = x.size(1)
  rope_cache = rope_cache[:T]

  # cast because the reference does
  xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
  rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
  x_out2 = torch.stack(
        [xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
         xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1)

  x_out2 = x_out2.flatten(3)
  return x_out2.transpose(1, 2).type_as(x)

dropout = 0.2
class FeedForward(nn.Module):
  def __init__(self, config: LLaMAConfig) -> None:
    super().__init__()
    hidden_dim = 4 * config.embedded_dim
    n_hidden = int(2 * hidden_dim / 3)
        

    self.c_fc1 = nn.Linear(config.embedded_dim, n_hidden, bias=False)
    self.c_fc2 = nn.Linear(config.embedded_dim, n_hidden, bias=False)
    self.c_proj = nn.Linear(n_hidden, config.embedded_dim, bias=False)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x


class Transformer(nn.Module):
  def __init__(self, config: LLaMAConfig) -> None:
    super().__init__()
    self.attention = AttentionHead(config)
    self.rms_norm_1 = AttentionHead(config)
    self.rms_norm_2 = RMSNorm(config.embedded_dim)
    self.feedforward = FeedForward(config)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attention(self.rms_norm_1(x))
    x = x + self.feedforward(self.rms_norm_2(x))
    return x

class BabyGPTmodel(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        
        self.config = config     
        self.token = nn.Embedding(config.vocab_size, config.embedded_dim)
        self.positional_embeddings = nn.Embedding(config.block_size, config.embedded_dim)
        self.layers1 = nn.ModuleList([Transformer(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.embedded_dim, eps = 1e-12) # final layer 
        self.lnum_heads = nn.Linear(config.embedded_dim, config.vocab_size)

        # init all weights
        # from karpathy
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
          if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))
            if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
              torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
      device = idx.device
      b, t = idx.size()
      tok_emb = self.token(idx)
      position_ids = torch.arange(0, t, dtype = torch.long).unsqueeze(0)
      pos_emb = self.positional_embeddings(position_ids)
      x = tok_emb + pos_emb
      for layers1 in self.layers1:
        x = layers1(x)
      x = self.ln_f(x)
      logits = self.lnum_heads(x[:, -1, :])
      return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))

config =  LLaMAConfig(block_size = 3,
    vocab_size = 478, num_layers  = 4,
    num_heads = 4,
    embedded_dim = 16)

llama = BabyGPTmodel(config)
