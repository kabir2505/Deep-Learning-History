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
from config import device,learning_rate,n_embd,n_head,n_layer
import numpy,pandas
# Set to False to skip notebook execution (e.g. for debugging)
from data_loader import vocab_size
from model import make_model




model=make_model(vocab=vocab_size,N=6,d_model=384,d_ff=1536,h=6,dropout=0.2).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion=nn.CrossEntropyLoss()
criterion


class Batch:
    def __init__(self, src, pad=0):
        self.src = src


        # Derive tgt and tgt_y from src
        self.tgt = src[:, :-1]  # Input to the decoder (shift right)
        self.tgt_y = src[:, 1:]  # Target for loss computation (shift left
        self.ntokens = (self.tgt_y != pad).data.sum()
class SimpleLossCompute:
    def __init__(self,generator,criterion,pad):
        self.generator=generator
        self.criterion=criterion

    def __call__(self,x,y,norm):
        logits=self.generator(x)
        loss=self.criterion(
            logits.contiguous().view(-1,logits.size(-1)),
            y.contiguous().view(-1)
        ) / norm

        return loss.data*norm,loss
loss_compute=SimpleLossCompute(model.generator,criterion,pad=0)
