import torch
from torch import nn


words = open(r'C:\Users\Soumyadip Nandi\Downloads\policy\input.txt', 'r').read().split()

chars = sorted(list(set(words)))
string2integer = {ch: i for i, ch in enumerate(chars)}
# print(string2integer)

integer2string = {i:ch for ch,i in string2integer.items()}
encode = lambda s: [string2integer[c] for c in s]
decode = lambda l: ''.join([integer2string[i] for i in l])
data = torch.tensor(encode(words), dtype = torch.long)


## block_size and batch size has been changed from 64 and 512 to 32 and 128
block_size = 32
batch_size = 128
ix = torch.randint(len(data) - block_size, (batch_size,))

## hidden dimensionality has been changed from 512 to 128.

vocab_size = len(chars)
d_k = 128

token_emb = nn.Embedding(vocab_size, d_k)
x = torch.stack([data[i:i + block_size] for i in ix])

input_embeds = token_emb(x)

# Cross Attention Added

class MultiHeadAttention(nn.Module):
    def __init__(self, embedded_dim, num_heads, embedded_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.embedded_dim = embedded_dim
        self.d_k = embedded_dim // self.num_heads
        if embedded_input is None:
            d_xq = d_xk = d_xv = embedded_dim
        else:
            d_xq, d_xk, d_xv = embedded_input
        # Embedding dimension of model is a multiple of number of heads
        assert embedded_dim % self.num_heads == 0

        # These are still of dimension embedded_dim. To split into number of heads
        self.W_q = nn.Linear(d_xq, embedded_dim , bias=False)
        self.W_k = nn.Linear(d_xk, embedded_dim, bias=False)
        self.W_v = nn.Linear(d_xv, embedded_dim, bias=False)
        # Outputs of all sub-layers need to be of dimension embedded_dim
        self.W_o = nn.Linear(embedded_dim, embedded_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
      d_k = K.size(-1)
      scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
      if mask is not None:
          scores = scores.masked_fill(mask == 0, -1e9)
      weights = nn.Softmax(dim=-1)(scores)
      output = torch.matmul(weights, V)
      return output

    def forward(self, Q, K, V, mask=None):
      # Apply linear transformation to Q, K, and V
      Q = self.W_q(Q)
      K = self.W_k(K)
      V = self.W_v(V)
        
      # Split into multiple heads
      Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
      K = K.view(batch_size, -1, self.num_heads, self.d_k)
      V = V.view(batch_size, -1, self.num_heads, self.d_k)
        
      # Transpose to prepare for matrix multiplication
      Q = Q.transpose(1, 2)
      K = K.transpose(1, 2)
      V = V.transpose(1, 2)
        
      # Compute attention scores and weights
      output = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        
      # Concatenate the outputs of the multiple heads
      output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
      output = self.W_o(output)
        
      return output

embedded_dim = 128
num_heads = 8

# print(chars)
seq_len = len(chars) # sequence length



Q = input_embeds
K = input_embeds
V = input_embeds
multihead_attn = MultiHeadAttention(embedded_dim, num_heads)
output = multihead_attn(Q, K, V)
print(output.shape)