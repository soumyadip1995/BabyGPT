import math
import torch
import torch.nn as nn
from torch.nn import functional as F 
from math import sqrt  

class Attention(nn.Module):
  def __init__(self, embedded_dim, num_heads, rank):
    super(Attention, self).__init__()
    self.rank = rank
    self.atten = nn.Linear(embedded_dim, 3 * embedded_dim)
    self.projection = nn.Linear(embedded_dim, embedded_dim)
    self.num_heads = num_heads
    self.embedded_dim = embedded_dim
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B,T,C = x.size()
    q, k ,v  = self.atten(x).split(self.embedded_dim, dim=2)
    q = q.view(B, T, self.rank, self.num_heads, C // self.num_heads).transpose(1, 2)
    k = k.view(B, T, self.rank, self.num_heads, C // self.num_heads).transpose(1, 2)
    v = v.view(B, T, self.rank, self.num_heads, C // self.num_heads).transpose(1, 2)


    # manual implementation of attention
    # from karpathy
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))   *  (self.rank ** 0.5))
    att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = self.projection(y)
    return y


dropout = 0.2
class FeedForward(nn.Module):
  def __init__(self, embedded_dim):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(nn.Linear(embedded_dim, 4 * embedded_dim),
    nn.Linear(4 * embedded_dim, embedded_dim),
    nn.GELU(),
    nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)

### A simple Transformer Block    
class Transformer(nn.Module):
  def __init__(self, embedded_dim, num_heads, rank):
    super(Transformer, self).__init__()
    self.attention = Attention(embedded_dim,  num_heads, rank)
    self.feed_forward = FeedForward(embedded_dim)
    self.layer_norm_1 = nn.LayerNorm(embedded_dim)
    self.layer_norm_2 = nn.LayerNorm(embedded_dim)

  def forward(self, x):
    
    x = x + self.attention(self.layer_norm_1(x))
    x = x + self.feed_forward(self.layer_norm_2(x))
    return x



class   LowTransformer(nn.Module):
  def __init__(self,  vocab_size, block_size,  embedded_dim, rank, num_layers,num_heads):
    super(LowTransformer, self).__init__()
    self.token = nn.Embedding(vocab_size, embedded_dim)
    self.positional_embeddings = nn.Embedding( block_size   , embedded_dim)
    self.layers1 = nn.ModuleList([Transformer(embedded_dim, num_heads, rank) for _ in range(num_heads)])
    self.ln_f = nn.LayerNorm(embedded_dim, eps = 1e-12) # final layer 
    self.ln_head = nn.Linear(embedded_dim, block_size)


    # init all weights
    ## from karpathy
    self.apply(self._init_weights)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith('projection.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx):
    device = idx.device
    b, t = idx.size()
    tok_emb = self.token(idx)
    position_ids = torch.arange(0, t, dtype = torch.long).unsqueeze(0)
    pos_emb = self.positional_embeddings(position_ids)
    x = tok_emb + pos_emb
    for layers1 in self.layers1:
      x = layers1(x)
    x = self.ln_f(x)
    logits = self.ln_head(x[:, -1, :])
    return logits


