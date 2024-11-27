import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch.functional as F
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine import train_model,criterion,pad_token,generate_next_words,Batch,make_model
from engine import SimpleLossCompute
from data_loader import TextDataset,dataloader,vocab,word2idx,idx2word


device="cuda" if torch.cuda.is_available() else "mps"
model=make_model(vocab=len(vocab),N=12,d_model=768,d_ff=3072,h=12,dropout=0.1).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss_compute=SimpleLossCompute(model.generator,criterion,pad_token)


train_model(model,dataloader,optimizer,loss_compute,epochs=30,accum_iter=2,device=device)

input_seq = "you know Caius Marcius is chief enemy to the people"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Call the function
output_text = generate_next_words(model, word2idx, input_seq, max_len=50, device=device)
print("Generated Text:", output_text)

torch.save(model.state_dict(), 'model.pth') 