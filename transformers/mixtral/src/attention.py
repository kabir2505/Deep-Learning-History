#Multihead gqa, rolling buffer kv cache, sliding window att
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from config import ModelArgs
device=ModelArgs.device

import math
import torch.nn.functional as F
class AttentionWithKVCache(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, max_seq_len: int = 2048, num_kv_heads:int=2):
        """
        Initialize the MultiHeadedAttention module with KV cache.

        Args:
            dim (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            window_size (int): The size of the window for rolling buffer KV cache.
            max_seq_len (int, optional): The maximum sequence length for initialization of KV cache. Defaults to 2048.
            num_kv_heads (int, optional): The number of attention heads for KV projection. Defaults to 2.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_kv_heads=num_kv_heads
        self.max_seq_len = max_seq_len
        self.repeats=num_heads//num_kv_heads
        self.window_size=window_size
        self.half_window=self.window_size//2
        
        # Projection layers
        self.W_q = nn.Linear(dim,self.num_heads*self.head_dim)
        self.W_k = nn.Linear(dim, self.num_kv_heads*self.head_dim)
        self.W_v = nn.Linear(dim, self.num_kv_heads*self.head_dim)
        self.W_o = nn.Linear(dim, dim)

        # Initialize KV cache
        # self.register_buffer('cache_k', torch.zeros(
        #     (1, max_seq_len, self.num_kv_heads, self.head_dim)  # batch=1 for simplicity
        # ))
        # self.register_buffer('cache_v', torch.zeros(
        #     (1, max_seq_len, self.num_kv_heads, self.head_dim)
        # ))
        self.register_buffer('cache_k', torch.zeros((max_seq_len, self.num_kv_heads, self.head_dim)))
        self.register_buffer('cache_v', torch.zeros((max_seq_len, self.num_kv_heads, self.head_dim)))
        
        # self.cache_k=None
        # self.cache_v=None
        self.cache_pos=0
    
    def update_cache(self,seq_len:int,k:torch.Tensor,v:torch.Tensor):
        """
        Update KV cache with new key-value pairs.

        Args:
            seq_len (int): The sequence length of the new key-value pairs.
            k (torch.Tensor): The new key tensor of shape (batch_size, seq_len, num_heads, head_dim).
            v (torch.Tensor): The new value tensor of shape (batch_size, seq_len, num_heads, head_dim).

        Returns:
            None
        """
        seq_len=k.size(1)
        
        if self.cache_pos + seq_len > self.max_seq_len: #check if cache has enough space
            #roll the cache to make space
            roll_amount=seq_len
            self.cache_k=torch.roll(self.cache_k,shifts=-roll_amount,dim=0)
            self.cache_v=torch.roll(self.cache_v,shifts=-roll_amount,dim=0)
            self.cache_pos-=roll_amount
        
        self.cache_k[self.cache_pos:self.cache_pos+seq_len]=k.squeeze(0)
        self.cache_v[self.cache_pos:self.cache_pos+seq_len]=v.squeeze(0)
        self.cache_pos+=seq_len

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Compute attention with KV cache.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int, optional): The starting position of the sequence. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)


        if self.training:
            # Training mode - full attention
            q=q.transpose(1,2)
            k=k.transpose(1,2)
            v=v.transpose(1,2)
            k=torch.repeat_interleave(k,dim=1,repeats=self.repeats) #batch_size,num_heads,seq_len,head_dim
            v=torch.repeat_interleave(v,dim=1,repeats=self.repeats)#batch_size,num_heads,seq_len,head_dim
           
            pad_k=F.pad(k,(0,0,self.half_window,self.half_window))
            pad_v=F.pad(v,(0,0,self.half_window,self.half_window))

            k_unf=pad_k.unfold(dimension=2,size=self.window_size,step=1).transpose(3,4) #(batch_size,num_heads,seq_len,self.window_size,d_head)
            v_unf=pad_v.unfold(dimension=2,size=self.window_size,step=1).transpose(3,4) #(batch_size,num_heads,seq_len,self.window_size,d_head)
            q=q.unsqueeze(-2)
            
            attn_scores=einops.einsum(q,k_unf,'b h s w d, b h s w d -> b h s w ')
            
            mask=torch.tril(torch.ones(self.window_size,self.window_size,device=q.device))
            mask=mask[-1]
            mask=mask.view(1,1,1,self.window_size).expand(batch_size,self.num_heads,seq_len,self.window_size)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            output=einops.einsum(attn_weights,v_unf,'b h s w, b h s w d ->b h s d')
            
        else:
            #batch_size must be 1 for inference
            assert batch_size==1, "batch size must be 1"
            # Inference mode - use KV cache
            # Update cache
            self.update_cache(seq_len,k,v)
            current_len=min(self.cache_pos,self.max_seq_len)
            valid_cache_len=min(current_len,self.window_size)
            
            start_window=max(0,current_len-seq_len-self.half_window)
                
            if self.cache_k is None or self.cache_v is None:
                self.cache_k=torch.zeros((self.max_seq_len,self.num_kv_heads,self.head_dim))
                self.cache_v=torch.zeros((self.max_seq_len,self.num_kv_heads,self.head_dim))
            
            cached_k=self.cache_k[start_window:current_len].unsqueeze(0)
            cached_v=self.cache_v[start_window:current_len].unsqueeze(0)
            
 
            

            q=q.transpose(1,2)
            cached_k=cached_k.transpose(1,2)
            cached_v=cached_v.transpose(1,2)            # Compute attention
            cached_k=cached_k.repeat_interleave(self.repeats,dim=1)
            cached_v=cached_v.repeat_interleave(self.repeats,dim=1)
           
            attn_scores = torch.matmul(q, cached_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply causal mask for new tokens
        
            mask = torch.ones((seq_len, valid_cache_len), dtype=torch.bool, device=x.device)
            mask = torch.tril(mask, diagonal=valid_cache_len - seq_len)

            # Expand mask to (batch_size, num_heads, seq_len, valid_cache_len)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,valid_cache_len)
            mask = mask.expand(batch_size, self.num_heads, seq_len, valid_cache_len)

            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_probs, cached_v)

        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)

    def reset_cache(self):
        """Reset the KV cache between sequences"""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.cache_pos = 0