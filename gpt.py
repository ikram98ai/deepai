import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

with open("data/shakespeares.txt", 'r', encoding="utf-8") as f:
    text = f.read()

chars = list(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[idx] for idx in ids])

data = torch.tensor(encode(text))

n= int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

vocab_size=len(chars)
block_size=8
batch_size=32
learning_rate=1e-3
n_embd=32
max_iters = 5000
eval_iters= 300
eval_interval= 300
device = 'gpu' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size, (block_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return xb.to(device), yb.to(device)

torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb,yb = get_batch(split)
            _, loss = model(xb,yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.ones(block_size,block_size))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        wei:Tensor = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) ->  (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B,T,T) @ (B,T,C)
        return out

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,n_embd)
        self.position_embedding = nn.Embedding(block_size,n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx:Tensor, target:Tensor=None): # (B,T), (B,T)
        B,T = idx.shape
        tok_emb = self.token_embedding(idx) # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T,device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self,idx:Tensor, max_token): # idx: (B,T)
        for _ in range(max_token):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # (B,T,C)
            logits = logits[:,-1,:] # (B,C)
            probs = torch.softmax(logits, dim=1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx,idx_next),dim=1) # (B,T+1)
        return idx
    


model = GPTModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch("train")
    logits, loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context,max_token=500)[0].tolist()))