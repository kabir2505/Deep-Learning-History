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

import numpy,pandas
from config import device
class GPT(nn.Module):

    def __init__(self,decoder,embed,generator):
        """
        GPT model architecture.
        """
        super(GPT,self).__init__()
        self.decoder=decoder
        self.embed=embed
        self.generator=generator

    def forward(self,x):
        #x-> [batch_size,seq_len]

        embedded=self.embed(x)
        return self.decoder(embedded)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.proj=nn.Linear(in_features=d_model,out_features=vocab)

    def forward(self,x):
        #x -> [batch_size,seq_len,d_model]
        projection=self.proj(x) # [batch_size,seq_len,vocab]
        #pred=log_softmax(projection,dim=-1) # [batch_size,seq_len,vocab]
        return projection #since crossentropy expects raw logits
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)  # layer.size -> [d_model]

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        # print("hello from decoder")


        return self.norm(x)

class LayerNorm(nn.Module):
    #https://tungmphung.com/wp-content/uploads/2020/01/Screenshot-from-2020-01-05-07-00-09.png
    def __init__(self,feature,eps=1e-6): #feature -> [d_model]
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(feature))
        self.beta=nn.Parameter(torch.zeros(feature))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)

        return self.gamma*(x-mean)/(std+self.eps) + self.beta

class SubLayerConnection(nn.Module):
    """Residual connection followed by a layer norm """
    def __init__(self,size,dropout):
        super(SubLayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        # print("hello from sublayer")

        return x + self.dropout(sublayer(self.norm(x)))
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SubLayerConnection(size,dropout),2)
        self.size=size

    def forward(self,x):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x))
        return self.sublayer[1](x,self.feed_forward)


def attention(query,key,value,dropout=None):
    #query,key,value -> [batch_size,h,seq_len,d_k]
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k) #[batch_size,h,seq_len,seq_len]
    # print("Query shape:", query.shape)
    # print("Key shape:", key.shape)
    # print("Value shape:", value.shape)


        #print('mask',mask.shape)
    seq_len=query.size(2)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)
    scores=scores.masked_fill(mask==0,-1e9) # wherever mask==0, fill that up with -inf
        #print('scores',scores.shape[-1])
    p_attn=scores.softmax(dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        #h=12,N=12
        super(MultiHeadedAttention,self).__init__()
        assert d_model%h==0
        self.d_k=d_model//h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value):
        #mask shape-> [batch_size,seq_len,seq_len]

        nbatches=query.size(0)

        query,key,value=[
            lin(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for lin,x in zip(self.linears,(query,key,value))
        ]
        #print('mask fow',mask.shape)
        x,self_attn=attention(query,key,value,dropout=self.dropout)

        x=(
            x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x) #WO * x

#3072
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        #d_ff=3072
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(p=dropout)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )


    def forward(self,x):
        #print("hello from feeforward")
        return self.net(x)

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(num_embeddings=vocab,embedding_dim=d_model)
        self.d_model=d_model

    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the input and output features.
            dropout (float): The dropout rate to be applied after the positional encoding.
            max_len (int, optional): The maximum length of the sequence to be encoded. Defaults to 5000.
        """
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # print('diveterm',div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        #print("hello from posen")
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(vocab,N=6,d_model=384,d_ff=1536,h=6,dropout=0.1):
    c=copy.deepcopy
    attn=MultiHeadedAttention(h,d_model)
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)
    position=PositionalEncoding(d_model,dropout)
    model=GPT(
        Decoder(DecoderLayer(d_model,c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,vocab),c(position)),
        Generator(d_model,vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model