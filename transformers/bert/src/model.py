import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchtext
import scipy
import re
import random
import math
import torch.nn as nn
import numpy


def get_attn_pad_mask(seq_q,seq_k):
    """
    Create a padding mask for attention mechanism.

    This function generates a padding mask for the attention mechanism where 
    `seq_k` contains padding tokens. The mask is used to prevent attention 
    from being applied to the padding tokens in the sequence. The mask is 
    expanded to match the dimensions required for the attention operation.

    Args:
        seq_q (torch.Tensor): The query sequence of shape (batch_size, len_q).
        seq_k (torch.Tensor): The key sequence of shape (batch_size, len_k) 
                            which may contain padding tokens.

    Returns:
        torch.Tensor: A padding mask of shape (batch_size, len_q, len_k) where 
    /******  96bf9fa3-608e-4617-bc2b-9ba2a682613e  *******/
                    positions with padding tokens have a value of 1 (True) and 
                    other positions have a value of 0 (False).
    """
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1) # 1->padding token , unsqueeze, 1 here will be a placeholder # batch_size,1,l_k
    return pad_attn_mask.expand(batch_size,len_q,len_k) # expand to match the self_attn multiplication    

def gelu(x):
    """
    Implementation of the gelu activation function.

    The gelu activation function is a variation of the ReLU activation function.
    It can be used as an alternative to the ReLU activation function to avoid dying neurons.
    The gelu activation function maps all negative values to 0 and all positive values to the
    corresponding positive value. It is defined as:



    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model,maxlen,nsegments):
        super(Embedding,self).__init__()
        self.tok_embed=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.pos_embed=nn.Embedding(num_embeddings=maxlen,embedding_dim=d_model)
        self.segment_embed=nn.Embedding(num_embeddings=nsegments,embedding_dim=d_model)
        self.norm=nn.LayerNorm(d_model)
    
    def forward(self,x,seg):
        """
        Forward pass of the embedding layer.

        The embedding layer takes in an input tensor and its corresponding segment tensor.
        The input tensor is embedded using the token embedding layer.
        The position tensor is embedded using the position embedding layer.
        The segment tensor is embedded using the segment embedding layer.
        The three embeddings are added together and passed through a layer normalization layer.
        The output of the layer normalization layer is the output of the embedding layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).
            seg (torch.Tensor): The segment tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len=x.size(1)
        pos=torch.arange(seq_len,dtype=torch.long) # (seq_len,)
        pos=pos.unsqueeze(0).expand_as(x) #(1, seq_len) -> (batch_size, seq_len).
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.segment_embed(seg)
        return self.norm(embedding)
    
    
def attention(query,key,value,attn_mask):
    #query,key,value -> batch_size,h,seq_len,d_k   (d_k is d_model//h)
    
    """
    Compute the attention and context for input queries, keys, and values.

    The attention mechanism calculates a weighted sum of the values based 
    on the similarity between the queries and keys. The scores are masked 
    using the attention mask to prevent attending to certain positions, 
    such as padding tokens.

    Args:
        query (torch.Tensor): The query tensor with shape (batch_size, h, seq_len, d_k).
        key (torch.Tensor): The key tensor with shape (batch_size, h, seq_len, d_k).
        value (torch.Tensor): The value tensor with shape (batch_size, h, seq_len, d_k).
        attn_mask (torch.Tensor): The attention mask tensor where positions to be masked 
                                  are indicated with True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - context: The context tensor after applying attention with shape (batch_size, h, seq_len, d_k).
            - attn: The attention weights with shape (batch_size, h, seq_len, seq_len).
    """
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k) # batch_size,h,seq_len, seq_len
    scores.masked_fill(attn_mask,-1e9)
    attn=scores.softmax(-1) 
    context=torch.matmul(attn,value) # batch_size,h,seq_len,d_k
    return context, attn  


class MultiHeadedAttention(nn.Module):
    def __init__(self,d_model,h):
        super(MultiHeadedAttention,self).__init__()
        self.d_model=d_model
        self.d_k=d_model//h
        self.h=h
        self.W_Q=nn.Linear(in_features=d_model,out_features=d_model)
        self.W_V=nn.Linear(in_features=d_model,out_features=d_model)
        self.W_K=nn.Linear(in_features=d_model,out_features=d_model)
    def forward(self,query,key,value,attn_mask):
        #query,key,value -> [batch_size,seq_len,d_model]
        """
        Compute the output of the multi-headed attention layer.

        The multi-headed attention layer allows the model to jointly attend
        to information from different representation subspaces at different
        positions. The layer takes in three inputs of shape (batch_size, seq_len, d_model)
        and an attention mask of shape (batch_size, seq_len, seq_len) where
        positions to be masked are indicated with True.

        Args:
            query (torch.Tensor): The query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): The key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): The value tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor): The attention mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output: The output tensor after applying multi-headed attention with shape (batch_size, seq_len, d_model).
                - attn: The attention weights with shape (batch_size, seq_len, seq_len).
        """
        batch_size=query.size(0)
        res=query
        query=self.W_Q(query).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        value=self.W_V(value).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        key=self.W_K(key).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
        
        attn_mask= attn_mask.unsqueeze(1).repeat(1,self.h,1,1) #[batch_size x n_heads x len_q x len_k]
        
        context,attn=attention(query,key,value,attn_mask)
        
        context=context.transpose(1,2).contiguous().view(batch_size,-1,self.h*self.d_k)
        output=nn.Linear(self.d_k*self.h,self.d_model)(context)
        return nn.LayerNorm(self.d_model)(output + res),attn
    
    

class PositionwiseFeedForward(nn.Module):
    
    def __init__(self,d_model,d_ff):
        super(PositionwiseFeedForward,self).__init__()
        self.W_1=nn.Linear(in_features=d_model,out_features=d_ff)
        self.W_2=nn.Linear(in_features=d_ff,out_features=d_model)
    
    def forward(self,x):
        return self.W_2(gelu(self.W_1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,d_ff):
        super(EncoderLayer,self).__init__()
        self.attn=MultiHeadedAttention(d_model=d_model,h=h)
        self.ffn=PositionwiseFeedForward(d_model=d_model,d_ff=d_ff)
    
    def forward(self,enc_inputs,enc_self_attn_mask):
        enc_outputs,attn=self.attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs=self.ffn(enc_outputs)

        return enc_outputs,attn

class BERT(nn.Module):
    
    def __init__(self,vocab_size,d_model,max_len,n_segments,h,d_ff,n_layers):
        super(BERT,self).__init__()
        self.embedding=Embedding(vocab_size=vocab_size,d_model=d_model,maxlen=max_len,nsegments=n_segments)
        self.layers=nn.ModuleList([EncoderLayer(d_model=d_model,h=h,d_ff=d_ff) for _ in range(n_layers)])
        self.fcls=nn.Linear(in_features=d_model,out_features=d_model)
        self.act1=nn.Tanh()
        #mlm
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        
        self.norm=nn.LayerNorm(d_model)
        
        self.classifier=nn.Linear(d_model,2) #0,1
        
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    
    def forward(self,input_ids,segment_ids,masked_pos):
        output=self.embedding(input_ids,segment_ids)
        enc_self_attn_mask=get_attn_pad_mask(input_ids,input_ids)
        for layer in self.layers:
            output,enc_self_attn=layer(output,enc_self_attn_mask)
            #output->[batch_size,seq_len,d_model] enc_self_attn-> [batch_size,n_heads,d_model,d_model]
        h_pooled=self.act1(self.fcls(output[:,0])) #[batch_size, d_model]
        logits_clsf=self.classifier(h_pooled) # [batch_size,2]
        
        
        masked_pos=masked_pos[:,:,None].expand(-1, -1, output.size(-1)) #[batch_size, max_pred] -> [batch_size, max_pred,1]
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_clsf
        
        
        