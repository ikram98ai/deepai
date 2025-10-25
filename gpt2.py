from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    block_size:int =1024
    vocab_size:int =50257
    n_layer:int =12
    n_head:int =12
    n_embd:int =768

class CasualSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig ):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                     .view(1,1,config.block_size,config.block_size) ))
    def forward(self, x:Tensor):
        B,T,C = x.size()
        qkv:Tensor = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)

        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)

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
        self.c_proj.GPT_SCALE_INIT = 1

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
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size,self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size,self.config.n_embd),
            h = nn.ModuleList(Block(self.config) for _ in range(self.config.n_layer)),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2*self.config.n_layer) **-0.5 
            nn.init.normal_(module.weight,mean=0,std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight,mean=0,std=0.02)


    def forward(self, idx, target=None):
        B,T = idx.shape
        assert T <= self.config.block_size, f"Can not forward sequence {T}, max block size is {self.config.block_size}"
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x) 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss=None
        if target is not None:
            loss = F.cross_entropy(logits.view(B*T,-1), target.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ['gpt2', 'gpt2-medium','gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained %s" % model_type)

        config_args = {
            "gpt2" :        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium" : dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-larage" : dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl" :     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type,cache_dir='data/hfmodel')
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"{k} shapes are not matching"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"{k}'s shapes {sd_hf[k].shape} != {sd[k].shape} are not matching"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

import tiktoken
enc = tiktoken.get_encoding('gpt2')
class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        with open('data/shakespeares.txt','r') as f:
            text = f.read()

        self.tokens = torch.tensor(enc.encode(text))
        self.current_position = 0

        print(f"Total tokens= {len(self.tokens)}")
        print(f"1 epoch= {len(self.tokens)//(B*T)} batches")

    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B*self.T+1]
        x = buf[:-1].view(self.B,self.T)
        y = buf[1:].view(self.B,self.T)

        self.current_position += self.B*self.T
        if self.current_position + (self.B*self.T+1) > len(self.tokens):
            self.current_position = 0
        return x,y


############################################################################################################
import time 
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

train_loader = DataLoaderLite(B=4,T=32)
model = GPT(GPTConfig())
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(20):
    t0 = time.time()
    x,y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits,loss = model(x,y)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tok_per_sec = (train_loader.B*train_loader.T) / (t1-t0)
    print(f"step {i}, loss: {loss.item()}, dt:{dt:.2f}ms, tok/sec: {tok_per_sec:.2f}")




import sys; sys.exit(0)

num_return_sequences = 5
max_tokens = 30
model = GPT.from_pretrained("gpt2")
model.eval()

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)

x = tokens.to(device)

while x.size(1) < max_tokens:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        ix = torch.multinomial(topk_probs,1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x,xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i,:max_tokens].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)