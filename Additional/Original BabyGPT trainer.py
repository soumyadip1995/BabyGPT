
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional

# hyperparameters

@dataclass
class GPTConfig:
    # these are default GPT-2 hyperparameters
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias :bool = False


### other hyperparametres
batch_size = 64
max_iters = 11000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 500
eval_iters = 200
dropout = 0.2


torch.manual_seed(1337)
words = open(r"/content/shakespeare.txt", 'r', encoding='utf-8').read()

chars = sorted(list(set(words)))
vocab_size = len(chars)


string2integer = {ch: i for i, ch in enumerate(chars)}
integer2string = {i:ch for ch,i in string2integer.items()}
encode = lambda s: [string2integer[c] for c in s]
decode = lambda l: ''.join([integer2string[i] for i in l])
data = torch.tensor(encode(words), dtype = torch.long)


## train and split the data
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i:i+ config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


### from pytorch GPT tutorial
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


torch.manual_seed(1337)
class Attention(nn.Module):
  def __init__(self, config):
    super(Attention, self).__init__()

    assert config.n_embd % config.n_head == 0

    self.atten = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

  def forward(self, x):
    B,T,C = x.size()
    q, k ,v  = self.atten(x).split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


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
  def __init__(self,config):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
    nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
    nn.GELU(),
    nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)

### A simple Transformer Block
class Transformer(nn.Module):
  def __init__(self,config):
    super(Transformer, self).__init__()
    self.attention = Attention(config)
    self.feed_forward = FeedForward(config)
    self.layer_norm_1 = nn.LayerNorm(config.n_embd)
    self.layer_norm_2 = nn.LayerNorm(config.n_embd)

  def forward(self, x):

    x = x + self.attention(self.layer_norm_1(x))
    x = x + self.feed_forward(self.layer_norm_2(x))
    return x


class BabyGPTmodel(nn.Module):

    def __init__(self, config):
        super(BabyGPTmodel, self).__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Transformer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps = 1e-12) # final layer norm
        self.lnum_heads = nn.Linear(config.n_embd, config.vocab_size)

        ## init all weights
        ## from karpathy
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
          if pn.endswith('projection.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        tok_emb = self.token(idx)
        position_ids = torch.arange(0, T, dtype = torch.long, device = device).unsqueeze(0)
        pos_emb =  self.positional_embeddings(position_ids)
        x = tok_emb + pos_emb
        for block in self.blocks:
          x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lnum_heads(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    ## from karpathy's youtube videos.
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


config = GPTConfig(
    block_size = 4,
    vocab_size = len(chars),
    n_head = 4,
    n_layer = 4,
    n_embd = 16)

model = BabyGPTmodel(config)

m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
