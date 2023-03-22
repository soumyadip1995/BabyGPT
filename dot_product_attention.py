import torch
from torch import nn
import math
import numpy as np 
import torch.nn.functional  as F
from math import sqrt


words = open(r"C:\Users\Soumyadip Nandi\Downloads\policy\language\text.txt", 'r' , encoding='utf-8').read().split()
# words[:20]


chars = sorted(list(set(words)))
string2integer = {ch: i for i, ch in enumerate(chars)}
# print(string2integer)

integer2string = {i:ch for ch,i in string2integer.items()}
encode = lambda s: [string2integer[c] for c in s]
# print(encode)

decode = lambda l: ''.join([integer2string[i] for i in l])
# print(decode)

data = torch.tensor(encode(words), dtype = torch.long)
# print(data)
# data.size()

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
# input_embeds.size()


def scaled_dot_product(query, key, value):
  dim_k = query.size(-1)
  scores = torch.bmm(query, key.transpose(-2, -1)) / sqrt(dim_k)
  weights = F.softmax(scores, dim = -1)
  return torch.bmm(weights, value)

key = input_embeds
query = input_embeds
value = input_embeds

sdp = scaled_dot_product(query, key, value)
print(sdp.size())
