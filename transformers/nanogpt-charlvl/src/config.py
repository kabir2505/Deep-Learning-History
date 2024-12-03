import torch
device="cuda" if torch.cuda.is_available() else "cpu"
dropout=0.2
learning_rate=3e-4
n_head=6
n_layer=6
n_embd=384