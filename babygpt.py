import math
import torch
import torch.nn as nn
from torch.nn import functional as F 
from math import sqrt



torch.manual_seed(1337)
class Attention(nn.Module):
  def __init__(self, embedded_dim, num_heads):
    super(Attention, self).__init__()
    self.atten = nn.Linear(embedded_dim, 3 * embedded_dim)
    self.projection = nn.Linear(embedded_dim, embedded_dim)
    self.num_heads = num_heads
    self.embedded_dim = embedded_dim
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B,T,C = x.size()
    q, k ,v  = self.atten(x).split(self.embedded_dim, dim=2)
    q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)


    # manual implementation of attention
    # from karpathy
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
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
  def __init__(self, embedded_dim, num_heads):
    super(Transformer, self).__init__()
    self.attention = Attention(embedded_dim,  num_heads)
    self.feed_forward = FeedForward(embedded_dim)
    self.layer_norm_1 = nn.LayerNorm(embedded_dim)
    self.layer_norm_2 = nn.LayerNorm(embedded_dim)

  def forward(self, x):
    
    x = x + self.attention(self.layer_norm_1(x))
    x = x + self.feed_forward(self.layer_norm_2(x))
    return x


class BabyGPTmodel(nn.Module):
  def __init__(self, vocab_size, block_size, num_layers, embedded_dim, num_heads):
    super(BabyGPTmodel, self).__init__()
    self.token = nn.Embedding(vocab_size, embedded_dim)
    self.positional_embeddings = nn.Embedding(block_size, embedded_dim)
    self.layers1 = nn.ModuleList([Transformer(embedded_dim, num_heads) for _ in range(num_heads)])
    self.ln_f = nn.LayerNorm(embedded_dim, eps = 1e-12) # final layer 
    self.ln_head = nn.Linear(embedded_dim, vocab_size)


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


words = open(r"C:\Users\Soumyadip Nandi\Downloads\policy\input.txt", 'r', encoding='utf-8').read().split()

chars = sorted(list(set(words)))
string2integer = {ch: i for i, ch in enumerate(chars)}


integer2string = {i:ch for ch,i in string2integer.items()}
encode = lambda s: [string2integer[c] for c in s]

decode = lambda l: ''.join([integer2string[i] for i in l])
data = torch.tensor(encode(words), dtype = torch.long)

batch_size = 16
block_size = 4
embedded_dim = 16
num_heads = 4
num_layers = 4

# generate a small batch of data of inputs x and targets y

ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+block_size] for i in ix])
# print((x, y))

vocab_size = len(chars)
block_size = 4
embedded_dim = 16
num_heads = 4
num_layers = 4

gpt = BabyGPTmodel(vocab_size, block_size, num_layers, embedded_dim, num_heads)
## number of parameters: 860,326


optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=1e-1)

## Training
for i in range(1000):
    logits = gpt(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(i, loss.item())


# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long)
# print(decode((context)[0].tolist()))