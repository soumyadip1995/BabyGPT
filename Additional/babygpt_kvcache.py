import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from math import sqrt


words = open(r"/content/shakespeare.txt", 'r', encoding='utf-8').read().split()

chars = sorted(list(set(words)))
string2integer = {ch: i for i, ch in enumerate(chars)}
integer2string = {i: ch for ch, i in string2integer.items()}
encode = lambda s: [string2integer[c] for c in s]
decode = lambda l: ''.join([integer2string[i] for i in l])
data = torch.tensor(encode(words), dtype=torch.long)

vocab_size = len(chars)

block_size = 8
batch_size = 16
embedded_dim = 16
num_heads = 4
num_layers = 4

torch.manual_seed(1337)


# Rotary Position Embedding (RoPE)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor"""
    # x shape: (B, num_heads, T, head_dim)
    B, nh, T, hd = x.shape

    # Split into even and odd features
    x1 = x[..., 0::2]  # (B, nh, T, hd//2)
    x2 = x[..., 1::2]  # (B, nh, T, hd//2)

    # cos, sin shape: (T, hd)
    # We need them to be (T, hd//2) since we split x
    cos = cos[:T, :hd//2]  # (T, hd//2)
    sin = sin[:T, :hd//2]  # (T, hd//2)

    # Reshape for broadcasting: (1, 1, T, hd//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Interleave back
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # (B, nh, T, hd//2, 2)
    rotated = rotated.flatten(-2)  # (B, nh, T, hd)

    return rotated


# Multi-Query Attention with KV Cache
class MultiQueryAttention(nn.Module):
    def __init__(self, embedded_dim, num_heads):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.embedded_dim = embedded_dim
        self.head_dim = embedded_dim // num_heads

        assert embedded_dim % num_heads == 0, "embedded_dim must be divisible by num_heads"

        # Query has multiple heads, Key and Value have single head (MQA)
        self.q_proj = nn.Linear(embedded_dim, embedded_dim)
        self.k_proj = nn.Linear(embedded_dim, self.head_dim)
        self.v_proj = nn.Linear(embedded_dim, self.head_dim)

        self.projection = nn.Linear(embedded_dim, embedded_dim)
        self.register_buffer('tril', torch.tril(torch.ones(1000, 1000)))  # Large buffer

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        # Store attention weights
        self.attention_weights = None

    def forward(self, x, use_cache=False, past_kv=None):
        B, T, C = x.size()

        # Compute Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, head_dim)
        v = self.v_proj(x)  # (B, T, head_dim)

        # Reshape Q to multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)

        # Reshape K and V to single head
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)  # (B, 1, T, hd)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)  # (B, 1, T, hd)

        # Apply rotary embeddings to Q and K
        cos, sin = self.rotary_emb(T, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Handle KV cache for inference
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v) if use_cache else None

        # Get current sequence lengths
        q_len = q.size(2)
        kv_len = k.size(2)

        # Expand K and V to match number of query heads (MQA)
        k = k.expand(B, self.num_heads, kv_len, self.head_dim)  # (B, nh, kv_len, hd)
        v = v.expand(B, self.num_heads, kv_len, self.head_dim)  # (B, nh, kv_len, hd)

        # Compute attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, q_len, kv_len)

        # Apply causal mask
        mask = self.tril[:q_len, :kv_len]  # (q_len, kv_len)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        self.attention_weights = att

        y = att @ v  # (B, nh, q_len, hd)
        y = y.transpose(1, 2).contiguous().view(B, q_len, C)  # (B, q_len, C)

        # Output projection
        y = self.projection(y)

        if use_cache:
            return y, current_kv
        return y


dropout = 0.1


class FeedForward(nn.Module):
    def __init__(self, embedded_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedded_dim, 4 * embedded_dim),
            nn.GELU(),
            nn.Linear(4 * embedded_dim, embedded_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedded_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = MultiQueryAttention(embedded_dim, num_heads)
        self.feed_forward = FeedForward(embedded_dim)
        self.layer_norm_1 = nn.LayerNorm(embedded_dim)
        self.layer_norm_2 = nn.LayerNorm(embedded_dim)

    def forward(self, x, use_cache=False, past_kv=None):
        if use_cache:
            attn_out, current_kv = self.attention(self.layer_norm_1(x), use_cache=True, past_kv=past_kv)
            x = x + attn_out
            x = x + self.feed_forward(self.layer_norm_2(x))
            return x, current_kv
        else:
            x = x + self.attention(self.layer_norm_1(x))
            x = x + self.feed_forward(self.layer_norm_2(x))
            return x


class BabyGPTmodel(nn.Module):
    def __init__(self, vocab_size, block_size, num_layers, embedded_dim, num_heads):
        super(BabyGPTmodel, self).__init__()
        self.token = nn.Embedding(vocab_size, embedded_dim)
        self.positional_embeddings = nn.Embedding(1000, embedded_dim)  # Large position buffer
        self.layers1 = nn.ModuleList([Transformer(embedded_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedded_dim, eps=1e-12)
        self.ln_head = nn.Linear(embedded_dim, vocab_size)
        self.num_layers = num_layers
        self.block_size = block_size

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, use_cache=False, past_kvs=None):
        device = idx.device
        b, t = idx.size()

        # Get token embeddings
        tok_emb = self.token(idx)  # (B, T, C)

        # Handle position embeddings
        if use_cache and past_kvs is not None and past_kvs[0] is not None:
            # When using cache, calculate position based on cached sequence length
            past_len = past_kvs[0][0].size(2)  # Get length from cached keys
            position_ids = torch.arange(past_len, past_len + t, dtype=torch.long, device=device).unsqueeze(0)
        else:
            # Normal forward pass or first cached pass
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        pos_emb = self.positional_embeddings(position_ids)  # (1, T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        # Pass through transformer layers
        current_kvs = []
        for i, layer in enumerate(self.layers1):
            past_kv = past_kvs[i] if (past_kvs is not None and i < len(past_kvs)) else None
            if use_cache:
                x, current_kv = layer(x, use_cache=True, past_kv=past_kv)
                current_kvs.append(current_kv)
            else:
                x = layer(x)

        # Final layer norm and output projection
        x = self.ln_f(x)  # (B, T, C)
        logits = self.ln_head(x[:, -1, :])  # (B, vocab_size) - only last token

        if use_cache:
            return logits, current_kvs
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text with KV caching for efficiency"""
        past_kvs = None

        for _ in range(max_new_tokens):
            # For first iteration, use full context (up to block_size)
            # For subsequent iterations with cache, only use the last token
            if past_kvs is None:
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
                logits, past_kvs = self(idx_cond, use_cache=True, past_kvs=None)
            else:
                # Only pass the last token when using cache
                logits, past_kvs = self(idx[:, -1:], use_cache=True, past_kvs=past_kvs)

            # Apply temperature
            logits = logits / temperature

            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Generate training data
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+block_size] for i in ix])
print("Training data shapes:", x.shape, y.shape)

# Initialize model
gpt = BabyGPTmodel(vocab_size, block_size, num_layers, embedded_dim, num_heads)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=1e-1)

# Training loop
print("\nStarting training...")
for i in range(1000):
    logits = gpt(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")

print("\nTraining complete!")

# Example generation with KV cache
print("\nGenerating text with KV cache...")
context = torch.tensor([encode(words[:block_size])], dtype=torch.long)
generated = gpt.generate(context, max_new_tokens=20, temperature=0.8, top_k=10)
generated_text = decode(generated[0].tolist())
print(f"Generated: {generated_text}")