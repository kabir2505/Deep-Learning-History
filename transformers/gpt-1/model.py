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

class GPT(nn.Module):

    def __init__(self,decoder,embed,generator):
        """
        GPT model architecture.
        """
        super(GPT,self).__init__()
        self.decoder=decoder
        self.embed=embed
        self.generator=generator

    def forward(self,x,mask):
        #x-> [batch_size,seq_len]
        """
        Forward pass of the GPT model.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size,seq_len).
            mask (torch.Tensor): Mask tensor of shape (batch_size,seq_len,seq_len) to prevent self-attention to certain positions.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,seq_len,d_model) after applying the GPT model.
        """
        embedded=self.embed(x) #->[batch_size,seq_len,d_model]
        return self.decoder(embedded,mask) # [batch_size,seq_len,d_model]

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        """
        Args:
            d_model (int): The size of the embedding output.
            vocab (int): The size of the vocabulary.
        """
        super(Generator,self).__init__()
        self.proj=nn.Linear(in_features=d_model,out_features=vocab)

    def forward(self,x):
        #x -> [batch_size,seq_len,d_model]
        projection=self.proj(x) # [batch_size,seq_len,vocab]
        #pred=log_softmax(projection,dim=-1) # [batch_size,seq_len,vocab]
        return projection #since crossentropy expects raw logits

def clones(module,N):
    """
    Creates a list of cloned modules.

    Args:
        module (nn.Module): The module to be cloned.
        N (int): The number of times to clone the module.

    Returns:
        nn.ModuleList: A list containing N deep copies of the input module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    def __init__(self,layer,N):
        """
        Initializes the Decoder module.

        Args:
            layer (nn.Module): The DecoderLayer module to be cloned.
            N (int): The number of layers to be stacked.
        """
        super(Decoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)  # layer.size -> [d_model]

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        #print("hello from decoder")


        return self.norm(x)

class LayerNorm(nn.Module):
    #https://tungmphung.com/wp-content/uploads/2020/01/Screenshot-from-2020-01-05-07-00-09.png
    def __init__(self,feature,eps=1e-6): #feature -> [d_model]
        """
        Initializes the LayerNorm module.

        Args:
            feature (int): The dimensionality of the input feature space [d_model].
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
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
        """
        Initializes the SubLayerConnection module.

        Args:
            size (int): The size of the layer to be normalized.
            dropout (float): The dropout rate to apply after the sublayer.
        """
        super(SubLayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        # print("hello from sublayer")

        return x + self.dropout(sublayer(self.norm(x)))

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        """
        Initializes the DecoderLayer module.

        Args:
            size (int): The size of the layer.
            self_attn (nn.Module): The self-attention mechanism.
            feed_forward (nn.Module): The feed-forward network.
            dropout (float): The dropout rate.
"""
        super(DecoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SubLayerConnection(size,dropout),2)
        self.size=size

    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)  # Shape: [1, seq_len, seq_len]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)

    return subsequent_mask  # Mask out future positions (upper triangular)

def attention(query,key,value,mask=None,dropout=None):
    #query,key,value -> [batch_size,h,seq_len,d_k]
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Tensor of shape (batch_size, h, seq_len, d_k) representing the query.
        key (torch.Tensor): Tensor of shape (batch_size, h, seq_len, d_k) representing the key.
        value (torch.Tensor): Tensor of shape (batch_size, h, seq_len, d_k) representing the value.
        mask (torch.Tensor, optional): Optional tensor to mask certain positions; shape should be broadcastable to (batch_size, 1, seq_len, seq_len).
        dropout (nn.Dropout, optional): Dropout layer to apply on attention weights.

    Returns:
        torch.Tensor: The output tensor after applying attention weights to the value, of shape (batch_size, h, seq_len, d_k).
        torch.Tensor: The attention weights after applying softmax, of shape (batch_size, h, seq_len, seq_len).
    """
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k) #[batch_size,h,seq_len,seq_len]
    if mask is not None:
        mask = mask.unsqueeze(1).expand(-1, 12, -1, -1)
        scores=scores.masked_fill(mask==0,-1e9) # wherever mask==0, fill that up with -inf
    p_attn=scores.softmax(dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
    output=torch.matmul(p_attn,value)
    if torch.isnan(output).any():
        print("NaN detected in output.")
    return torch.matmul(p_attn,value),p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        #h=12,N=12
        """
        Initializes the MultiHeadedAttention module.

        Args:
            h (int): The number of attention heads.
            d_model (int): The dimensionality of the input and output features.
            dropout (float, optional): The dropout rate to be applied after the attention. Defaults to 0.1.

        Raises:
            AssertionError: If d_model is not divisible by h.
        """
        super(MultiHeadedAttention,self).__init__()
        assert d_model%h==0
        self.d_k=d_model//h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        #mask shape-> [batch_size,seq_len,seq_len]
        if mask is not None:
            mask.unsqueeze(1) # [batch_size,1,seq_len,seq_len] # same masking across all attention heads h

        nbatches=query.size(0)

        query,key,value=[
            lin(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for lin,x in zip(self.linears,(query,key,value))
        ]
        x,self_attn=attention(query,key,value,mask,dropout=self.dropout)

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
        """
        Initializes the PositionwiseFeedForward module.

        Args:
            d_model (int): The dimensionality of the input and output features.
            d_ff (int): The dimensionality of the inner layer.
            dropout (float, optional): The dropout rate to be applied after the inner layer. Defaults to 0.1.
        """
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(p=dropout)
        self.gelu=nn.GELU()

    def forward(self,x):
        #print("hello from feeforward")
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))

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