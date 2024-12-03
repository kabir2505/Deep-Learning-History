import torch
from config import device

with open("../data/paul_graham_essays.txt", "r") as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)

stoi={ch:i for i,ch in enumerate(chars)} # string to index
itos={i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[ch] for ch in s]

def decode(n):
    return ''.join([itos[k] for k in n])

data=torch.tensor(encode(text[:1350000]),dtype=torch.long)

ratio=int(0.9*len(data))
train_data=data[:ratio]
val_data=data[ratio:]

torch.manual_seed(42)
batch_size=64
block_size=257
def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size, ))
    x=torch.stack([data[i:i+block_size] for i in ix])
    #y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x


