import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['input_dim'] % config['n_heads'] == 0
        # key, query, value projections
        self.c_attn = nn.Linear(config['input_dim'], 3 * config['input_dim'], bias=config['bias'])
        # output projection
        self.c_proj = nn.Linear(config['input_dim'], config['input_dim'], bias=config['bias'])

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_heads']
        self.n_embd = config['input_dim']
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                             .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['input_dim'], 4 * config['input_dim'], bias=config['bias'])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config['input_dim'], config['input_dim'], bias=config['bias'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config['input_dim'], bias=config['bias'])
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config['input_dim'], bias=config['bias'])
        self.mlp = MLP(config)

    def forward(self, x):
        scores, att = self.attn(self.ln1(x))
        x = x + scores
        x = x + self.mlp(self.ln2(x))
        return x, att


class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['input_dim'] % config['n_heads'] == 0
        # key, query, value projections
        self.c_attn = nn.Linear(config['input_dim'], 3 * config['input_dim'], bias=config['bias'])
        # output projection
        self.c_proj = nn.Linear(config['input_dim'], config['input_dim'], bias=config['bias'])

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_heads']
        self.n_embd = config['input_dim']

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            mask = mask.unsqueeze(1).float()
            mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))
            att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MaskedAttenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config['input_dim'], bias=config['bias'])
        self.attn = MaskedSelfAttention(config)
        self.ln2 = LayerNorm(config['input_dim'], bias=config['bias'])
        self.mlp = MLP(config)

    def forward(self, x, mask):
        scores, att = self.attn(self.ln1(x), mask)
        x = x + scores
        x = x + self.mlp(self.ln2(x))
        return x, att



