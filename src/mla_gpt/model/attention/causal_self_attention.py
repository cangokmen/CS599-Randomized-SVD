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
        self.svd_rank = getattr(config, 'svd_rank', None)
        
        # Individual matrix compression controls
        self.svd_apply_to_q = getattr(config, 'svd_apply_to_q', False)
        self.svd_apply_to_k = getattr(config, 'svd_apply_to_k', False)
        self.svd_apply_to_v = getattr(config, 'svd_apply_to_v', True)
        
        # Individual randomized SVD controls (with fallback to global setting)
        global_rsvd = getattr(config, 'use_randomized_svd', False)
        self.use_randomized_svd_q = getattr(config, 'use_randomized_svd_q', global_rsvd)
        self.use_randomized_svd_k = getattr(config, 'use_randomized_svd_k', global_rsvd)
        self.use_randomized_svd_v = getattr(config, 'use_randomized_svd_v', global_rsvd)
        
        # Randomized SVD specific parameters (following Tropp's recommendations)
        self.svd_oversampling = getattr(config, 'svd_oversampling', 10)  # p parameter
        self.svd_power_iter = getattr(config, 'svd_power_iter', 1)       # q parameter
        
        # Initialize compression modules
        self.q_compressor = None
        self.k_compressor = None
        self.v_compressor = None
        
        if self.use_svd:
            # Initialize compressors for each matrix type that's enabled
            if self.svd_apply_to_q:
                if self.use_randomized_svd_q:
                    self.q_compressor = RandomizedSVDCompression(
                        rank=self.svd_rank,
                        oversampling=self.svd_oversampling,
                        power_iterations=self.svd_power_iter
                    )
                else:
                    self.q_compressor = SVDCompression(
                        rank=self.svd_rank,
                        compression_type='standard'
                    )
                    
            if self.svd_apply_to_k:
                if self.use_randomized_svd_k:
                    self.k_compressor = RandomizedSVDCompression(
                        rank=self.svd_rank,
                        oversampling=self.svd_oversampling,
                        power_iterations=self.svd_power_iter
                    )
                else:
                    self.k_compressor = SVDCompression(
                        rank=self.svd_rank,
                        compression_type='standard'
                    )
                    
            if self.svd_apply_to_v:
                if self.use_randomized_svd_v:
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

    def apply_svd_to_q(self, q):
        """
        Apply SVD compression to the Q (queries) matrix
        
        Args:
            q: Tensor of shape (B, nh, T, hs) - query matrices
        
        Returns:
            Compressed Q matrix after SVD decomposition
        """
        if not self.use_svd or not self.svd_apply_to_q or self.q_compressor is None:
            return q
        
        return self.q_compressor(q)
    
    def apply_svd_to_k(self, k):
        """
        Apply SVD compression to the K (keys) matrix
        
        Args:
            k: Tensor of shape (B, nh, T, hs) - key matrices
        
        Returns:
            Compressed K matrix after SVD decomposition
        """
        if not self.use_svd or not self.svd_apply_to_k or self.k_compressor is None:
            return k
        
        return self.k_compressor(k)
    
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
        if not self.use_svd or not self.svd_apply_to_v or self.v_compressor is None:
            return v
        
        return self.v_compressor(v)
    
    def get_compression_info(self, matrix_shape):
        """
        Get information about the compression being applied to Q, K, V matrices
        
        Args:
            matrix_shape: Shape of the matrices (B, nh, T, hs)
            
        Returns:
            Dictionary with compression information
        """
        if not self.use_svd:
            return {'compression': 'none'}
        
        info = {
            'rank': self.svd_rank,
            'matrix_shape': matrix_shape,
            'q_compression': self.svd_apply_to_q,
            'k_compression': self.svd_apply_to_k,
            'v_compression': self.svd_apply_to_v,
            'q_compression_type': 'randomized_svd' if (self.svd_apply_to_q and self.use_randomized_svd_q) else 'standard_svd' if self.svd_apply_to_q else 'none',
            'k_compression_type': 'randomized_svd' if (self.svd_apply_to_k and self.use_randomized_svd_k) else 'standard_svd' if self.svd_apply_to_k else 'none',
            'v_compression_type': 'randomized_svd' if (self.svd_apply_to_v and self.use_randomized_svd_v) else 'standard_svd' if self.svd_apply_to_v else 'none'
        }
        
        # Add randomized SVD parameters if any matrix uses it
        if any([self.use_randomized_svd_q and self.svd_apply_to_q,
                self.use_randomized_svd_k and self.svd_apply_to_k,
                self.use_randomized_svd_v and self.svd_apply_to_v]):
            info.update({
                'oversampling': self.svd_oversampling,
                'power_iterations': self.svd_power_iter
            })
            
            # Get complexity analysis from any active randomized compressor
            active_rsvd_compressor = None
            if self.q_compressor and self.use_randomized_svd_q:
                active_rsvd_compressor = self.q_compressor
            elif self.k_compressor and self.use_randomized_svd_k:
                active_rsvd_compressor = self.k_compressor
            elif self.v_compressor and self.use_randomized_svd_v:
                active_rsvd_compressor = self.v_compressor
                
            if active_rsvd_compressor and hasattr(active_rsvd_compressor, 'get_computational_complexity'):
                complexity_info = active_rsvd_compressor.get_computational_complexity(matrix_shape)
                memory_info = active_rsvd_compressor.get_memory_usage(matrix_shape)
                
                info.update({
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

        # Apply SVD compression to Q, K, V matrices as configured
        # Each matrix can be compressed independently based on configuration
        q = self.apply_svd_to_q(q)
        k = self.apply_svd_to_k(k)
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
