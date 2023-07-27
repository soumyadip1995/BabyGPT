### from https://github.com/Lightning-AI/lit-llama partially.
### from https://github.com/facebookresearch/llama/blob/main/llama/model.py partiallly

import math
from dataclasses import dataclass
from typing import Optional
from typing import List, Optional, Tuple, Any



import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self


words = open(r"C:\Users\Soumyadip Nandi\Downloads\policy\BabyGPT\data\ALL_eminem.txt", 'r', encoding='utf-8').read().split()

chars = sorted(list(set(words)))


@dataclass
class LLaMAConfig:
    
    vocab_size: int = 32000
    num_layers: int = 32
    num_heads: int = 32
    embedded_dim: int = 4096
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps : float = 1e-5
    max_seq_length: int = 2048
    max_batch_size = 32
    dropout : float = 0.1
    batch_size :int = 16



    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


### max_seq_length and block_size are the same as llama- v1

llama_configs = {
    "7B": dict(num_layers=32, num_heads=32, embedded_dim=4096),
    "13B": dict(num_layers=40, num_heads=40, embedded_dim=5120),
    "30B": dict(num_layers=60, num_heads=52, embedded_dim=6656),
    "65B": dict(num_layers=80, num_heads=64, embedded_dim=8192),
    "70B": dict(num_layers=80, num_heads=64, embedded_dim=8192),
}

#### arbitrary value for the 70B



class RMSNorm(torch.nn.Module):
    def __init__(self, embedded_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedded_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight




def precompute_freqs_cis(embedded_dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, embedded_dim, 2)[: (embedded_dim // 2)].float() / embedded_dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )



class AttentionHead(nn.Module):
  def __init__(self, config: LLaMAConfig):
    super().__init__()
    assert config.embedded_dim % config.num_heads == 0


    self.n_kv_heads = config.num_heads if config.n_kv_heads is None else config.n_kv_heads
    model_parallel_size = 1
    self.n_local_heads = config.num_heads // model_parallel_size
    self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
    self.n_rep = self.n_local_heads // self.n_local_kv_heads
    self.head_dim = config.embedded_dim // config.num_heads
    


    self.atten = nn.Linear(config.embedded_dim, 3 * config.embedded_dim, bias=False)
    self.c_proj = nn.Linear(config.embedded_dim, config.embedded_dim, bias=False)
    self.num_heads = config.num_heads
    self.embedded_dim = config.embedded_dim
    self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
                                    .view(1, 1, config.max_seq_length, config.max_seq_length))

  def forward(self, x:torch.Tensor, freqs_cis = torch.Tensor):
    bsz, seq_len, _ = x.shape


    xq, xk ,xv  = self.atten(x).split(self.embedded_dim, dim=2)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)


    xq, xk = apply_rotary_emb()

    # grouped multiquery attention: expand out keys and values
    xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
    xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

    # make heads into a batch dimension
    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    # manual implementation of attention
    # from karpathy
    att = (xq @ xk.transpose(-2, -1)) * (1.0 / math.sqrt(xk.size(-1)))
    att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ xv # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(bsz, seq_len, _) # re-assemble all head outputs side by side

    # output projection
    y = self.c_proj(y)
    return y
    
class FeedForward(nn.Module):
  def __init__(self, embedded_dim: int, n_hidden :int, multiple_of: int, dropout :float) -> None:
    super().__init__()
    
    n_hidden = int(2 * n_hidden / 3)
    n_hidden = multiple_of * ((n_hidden + multiple_of - 1)//multiple_of)
        

    self.c_fc1 = nn.Linear(embedded_dim, n_hidden, bias=False)
    self.c_fc2 = nn.Linear(embedded_dim, n_hidden, bias=False)
    self.c_proj = nn.Linear(n_hidden, embedded_dim, bias=False)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x




class Transformer(nn.Module):
  def __init__(self,  layer_id : int, config: LLaMAConfig) -> None:
    super().__init__()
    self.num_heads = config.num_heads
    self.embedded_dim = config.embedded_dim
    self.head_dim = config.embedded_dim // config.num_heads # multi head
    self.attention = AttentionHead(config)
    self.feedforward = FeedForward(config.embedded_dim, dropout = config.dropout, n_hidden = 4 * config.embedded_dim, multiple_of = config.multiple_of)
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(config.embedded_dim ,  eps = config.norm_eps)
    self.feed_forward_network = RMSNorm(config.embedded_dim, eps = config.norm_eps)


  def forward(self, x: torch.Tensor, freqs_cis) -> torch.Tensor:
    h = x + self.attention.forward(self.attention_norm(x), freqs_cis)
    out = h + self.feedforward.forward(self.feed_forward_network(x))
    return out



class BabyGPTmodel(nn.Module):
  def __init__(self, config : LLaMAConfig):
    super(BabyGPTmodel, self).__init__()
    assert config.vocab_size is not None
    assert config.max_seq_length is not None

    self.config = config
    self.token = nn.Embedding(config.vocab_size, config.embedded_dim)
    self.positional_embeddings = nn.Embedding(config.max_seq_length, config.embedded_dim)
    for layer_id in range(config.num_layers):
      self.blocks = nn.Sequential(*[Transformer(layer_id, config) for _ in range(config.num_layers)])
    self.ln_f = RMSNorm(config.embedded_dim, eps = 1e-12) # final layer norm
    self.lnum_heads = nn.Linear(config.embedded_dim, config.vocab_size)



    ## from karpathy
    # share the unembedding parameters with the embedding parameters
    self.token.weight = self.lnum_heads.weight # https://paperswithcode.com/method/weight-tying

    # some useful precompute for the RoPE relative positional embeddings. TODO why * 2 here? confuse
    freqs_cis = precompute_freqs_cis(self.config.embedded_dim // self.config.num_heads, self.config.max_seq_length * 2)
    self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    ## init all weights
    ## from karpathy
    self.apply(self._init_weights)
    for pn, p in self.named_parameters():
      if pn.endswith('atten.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

  def forward(self, idx):
    device = idx.device
    _bsz, seq_len = idx.size()
    tok_emb = self.token(idx)
    position_ids = torch.arange(0, seq_len, dtype = torch.long).unsqueeze(0)
    pos_emb = self.positional_embeddings(position_ids)
    x = tok_emb + pos_emb
    freqs_cis = self.freqs_cis[:seq_len]
    for block in self.blocks:
      x = self.blocks(x, freqs_cis)
    x = self.ln_f(x)
    logits = self.lnum_heads(x)

    return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
      return cls(LLaMAConfig.from_name(name))



config =  LLaMAConfig(max_seq_length = 64,
    vocab_size = len(chars), num_layers  = 4,
    num_heads = 4,
    embedded_dim = 256)

llama2 = BabyGPTmodel(config)


