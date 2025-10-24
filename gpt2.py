from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    block_size=1024
    vocab_size=50257
    n_embd=768
    n_head=12
    n_layer=12

class CasualSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig ):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                     .view(1,1,config.block_size,config.block_size) ))
    def forward(self, x:Tensor):
        B,T,C = x.size()
        qkv:Tensor = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)

        q = q.view(B,T, self.n_head, C/self.n_head).transpose(1,2)
        k = k.view(B,T, self.n_head, C/self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C/self.n_head).transpose(1,2)

        att = q @ k.transpose(-2,-1) * k.size(-1)**-0.5 # (B,nh,T,hs) @ (B,nh,hs,T) -> (B,nh,T,T)
        att = att.masked_fill(att[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.c_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config:GPTConfig ):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        config = GPTConfig()

        self.transformer = nn.ModuleDict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd)
        )
        self.lm_head = nn.Linear(config.vocab_size, config.n_embd)