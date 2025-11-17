import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..compression import SVDCompression, RandomizedSVDCompression

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # SVD compression configuration
        self.use_svd = getattr(config, 'use_svd', False)
        self.use_randomized_svd = getattr(config, 'use_randomized_svd', False)
        self.svd_rank = getattr(config, 'svd_rank', None)
        
        # Randomized SVD specific parameters (following Tropp's recommendations)
        self.svd_oversampling = getattr(config, 'svd_oversampling', 10)  # p parameter
        self.svd_power_iter = getattr(config, 'svd_power_iter', 1)       # q parameter
        
        # Initialize compression modules
        self.v_compressor = None
        if self.use_svd:
            if self.use_randomized_svd:
                self.v_compressor = RandomizedSVDCompression(
                    rank=self.svd_rank,
                    oversampling=self.svd_oversampling,
                    power_iterations=self.svd_power_iter
                )
            else:
                self.v_compressor = SVDCompression(
                    rank=self.svd_rank,
                    compression_type='standard'
                )
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def apply_svd_to_v(self, v):
        """
        Apply SVD compression to the V (values) matrix
        
        Uses either standard SVD or Tropp's randomized SVD based on configuration.
        
        Args:
            v: Tensor of shape (B, nh, T, hs) where
               B = batch size, nh = number of heads, T = sequence length, hs = head size
        
        Returns:
            Compressed V matrix after SVD decomposition
        """
        if not self.use_svd or self.v_compressor is None:
            return v
        
        return self.v_compressor(v)
    
    def get_compression_info(self, v_shape):
        """
        Get information about the compression being applied
        
        Args:
            v_shape: Shape of the V matrix (B, nh, T, hs)
            
        Returns:
            Dictionary with compression information
        """
        if not self.use_svd or self.v_compressor is None:
            return {'compression': 'none'}
        
        info = {
            'compression': 'randomized_svd' if self.use_randomized_svd else 'standard_svd',
            'rank': self.svd_rank,
            'v_shape': v_shape
        }
        
        if self.use_randomized_svd and hasattr(self.v_compressor, 'get_computational_complexity'):
            # Add Tropp algorithm specific information
            complexity_info = self.v_compressor.get_computational_complexity(v_shape)
            memory_info = self.v_compressor.get_memory_usage(v_shape)
            
            info.update({
                'oversampling': self.svd_oversampling,
                'power_iterations': self.svd_power_iter,
                'complexity': complexity_info,
                'memory': memory_info
            })
        
        return info
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply SVD compression to V (values) matrix if enabled
        # This uses either standard SVD or Tropp's randomized SVD
        v = self.apply_svd_to_v(v)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
