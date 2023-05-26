import torch
import torch.nn as nn
import math

class LowRankAttention(nn.Module):
    def __init__(self, dim, rank):
        super(LowRankAttention, self).__init__()
        self.rank = rank
        self.Wq = nn.Linear(dim, rank, bias=False)
        self.Wk = nn.Linear(dim, rank, bias=False)
        self.Wv = nn.Linear(dim, rank, bias=False)
        self.Wo = nn.Linear(rank, dim, bias=False)

    def forward(self, q, k, v):
        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)

        # Compute the attention scores using low-rank approximation
        A = torch.bmm(Q, K.transpose(-2, -1)) / (self.rank ** 0.5)

        # Softmax along the key dimension
        A = torch.softmax(A, dim=-1)

        # Compute the attention-weighted values using low-rank approximation
        AV = torch.bmm(A, V)

        # Apply the output layer to the attention-weighted values
        out = self.Wo(AV)

        return out

class LowRankTransformerLayer(nn.Module):
    def __init__(self, dim, rank, dropout=0.2):
        super(LowRankTransformerLayer, self).__init__()
        self.attention = LowRankAttention(dim, rank)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Compute the self-attention layer
        attention_out = self.attention(x, x, x)

        # Add residual connection and normalize
        x = self.norm1(x + self.dropout1(attention_out))

        # Feed-forward layer
        ff_out = self.feedforward(x)

        # Add residual connection and normalize
        x = self.norm2(x + self.dropout2(ff_out))

        return x

class LowRankTransformer(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, rank, num_heads, dropout= 0.2):
        super(LowRankTransformer, self).__init__()
        self.layers = nn.ModuleList([LowRankTransformerLayer(dim, rank, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.dim = dim
        self.rank = rank
        self.num_heads = num_heads
        self.pos_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        # init all weights
        ## from karpathy
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
          if pn.endswith('Wo.weight'):
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

    def forward(self, x):
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1)]

        # Apply dropout
        x = self.dropout(x)

        # Apply the transformer layers
        for layer in self.layers:
            x = layer(x)

        return x


words = open(r"\context\ALL_eminem.txt", 'r', encoding='utf-8').read().split()

chars = sorted(list(set(words)))
vocab_size = len(chars)
lrt = LowRankTransformer(vocab_size, 4, 16, 4, 8)

### number of parameters: 15328
