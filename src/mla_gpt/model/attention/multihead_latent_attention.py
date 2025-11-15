"""
Multi-Head Latent Attention (MLA) implementation

MLA compresses the key-value representations into a lower-dimensional latent space,
reducing the memory and computational requirements while maintaining attention quality.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_attention import BaseAttention


class MultiHeadLatentAttention(BaseAttention):
    """
    Multi-Head Latent Attention (MLA) mechanism.
    
    This attention variant compresses the key-value representations into a 
    lower-dimensional latent space, which can significantly reduce memory usage
    and computational cost while maintaining good performance.
    
    Key innovations:
    1. Compressed key-value representations in latent space
    2. Per-head latent dimensions that can be smaller than head_size
    3. Learnable compression and decompression matrices
    4. Optional SVD-based compression for further efficiency gains
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # MLA-specific configurations
        mla_latent_dim = getattr(config, 'mla_latent_dim', None)
        self.latent_dim = mla_latent_dim if mla_latent_dim is not None else self.head_size // 2
        self.use_bias = getattr(config, 'bias', True)
        
        # SVD parameters (inherited from config)
        self.use_svd = getattr(config, 'use_svd', False)
        self.svd_rank = getattr(config, 'svd_rank', None)
        
        # Query projection (full size)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_bias)
        
        # Key-Value latent projections (compressed)
        self.kv_latent_proj = nn.Linear(self.n_embd, 2 * self.n_head * self.latent_dim, bias=self.use_bias)
        
        # Key-Value expansion from latent space
        self.k_expand = nn.Linear(self.latent_dim, self.head_size, bias=False)
        self.v_expand = nn.Linear(self.latent_dim, self.head_size, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.use_bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using scaled normal distribution"""
        # Initialize projection layers with scaled normal distribution
        std = 0.02
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.kv_latent_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_expand.weight, mean=0.0, std=std / math.sqrt(2))
        nn.init.normal_(self.v_expand.weight, mean=0.0, std=std / math.sqrt(2))
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=std / math.sqrt(2))
        
        # Initialize biases to zero if they exist
        if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if hasattr(self.kv_latent_proj, 'bias') and self.kv_latent_proj.bias is not None:
            nn.init.zeros_(self.kv_latent_proj.bias)
        if hasattr(self.c_proj, 'bias') and self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)
    
    def apply_svd_compression(self, tensor):
        """
        Apply SVD-based compression to tensors if enabled
        
        Args:
            tensor: Input tensor of shape (B, nh, T, dim)
            
        Returns:
            Compressed tensor of same shape
        """
        if not self.use_svd:
            return tensor
        
        B, nh, T, dim = tensor.shape
        tensor_reshaped = tensor.reshape(B * nh, T, dim)
        tensor_reconstructed = torch.zeros_like(tensor_reshaped)
        
        for i in range(B * nh):
            # Perform SVD
            U, S, Vh = torch.linalg.svd(tensor_reshaped[i], full_matrices=False)
            
            # Determine compression rank
            if self.svd_rank is not None:
                rank = min(self.svd_rank, S.shape[0])
            else:
                rank = S.shape[0]
            
            # Reconstruct with reduced rank
            S_diag = torch.diag(S[:rank])
            tensor_reconstructed[i] = U[:, :rank] @ S_diag @ Vh[:rank, :]
        
        return tensor_reconstructed.reshape(B, nh, T, dim)
    
    def forward(self, x):
        """
        Forward pass of Multi-Head Latent Attention
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()  # batch_size, sequence_length, embedding_dim
        
        # 1. Compute queries (full dimensionality)
        q = self.q_proj(x)  # (B, T, n_embd)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # 2. Compute key-value latent representations (compressed)
        kv_latent = self.kv_latent_proj(x)  # (B, T, 2 * n_head * latent_dim)
        kv_latent = kv_latent.view(B, T, 2, self.n_head, self.latent_dim)
        k_latent, v_latent = kv_latent.unbind(dim=2)  # Each: (B, T, nh, latent_dim)
        
        # Transpose to get (B, nh, T, latent_dim)
        k_latent = k_latent.transpose(1, 2)  # (B, nh, T, latent_dim)
        v_latent = v_latent.transpose(1, 2)  # (B, nh, T, latent_dim)
        
        # 3. Expand latent representations to full head size
        # Reshape for linear projection: (B*nh, T, latent_dim)
        k_latent_flat = k_latent.reshape(B * self.n_head, T, self.latent_dim)
        v_latent_flat = v_latent.reshape(B * self.n_head, T, self.latent_dim)
        
        # Expand to full head size
        k_expanded = self.k_expand(k_latent_flat)  # (B*nh, T, head_size)
        v_expanded = self.v_expand(v_latent_flat)  # (B*nh, T, head_size)
        
        # Reshape back: (B, nh, T, head_size)
        k = k_expanded.view(B, self.n_head, T, self.head_size)
        v = v_expanded.view(B, self.n_head, T, self.head_size)
        
        # 4. Apply SVD compression to values if enabled
        v = self.apply_svd_compression(v)
        
        # 5. Compute attention
        if self.flash:
            # Use flash attention if available
            y = self._apply_flash_attention(q, k, v)
        else:
            # Manual attention computation
            y = self._apply_manual_attention(q, k, v)
        
        # 6. Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        
        return y
    
    def get_compression_ratio(self):
        """
        Calculate the compression ratio achieved by the latent representation
        
        Returns:
            Compression ratio as a float
        """
        original_kv_params = 2 * self.n_embd * self.n_embd  # Original K,V projections
        
        # MLA parameters
        latent_proj_params = self.n_embd * (2 * self.n_head * self.latent_dim)
        expand_params = self.n_head * (self.latent_dim * self.head_size * 2)  # K and V expansion
        mla_kv_params = latent_proj_params + expand_params
        
        return original_kv_params / mla_kv_params
    
    def get_memory_savings(self, sequence_length):
        """
        Estimate memory savings for key-value caching during inference
        
        Args:
            sequence_length: Length of the sequence for KV caching
            
        Returns:
            Memory reduction factor
        """
        original_kv_cache = sequence_length * self.n_head * self.head_size * 2  # K and V
        mla_kv_cache = sequence_length * self.n_head * self.latent_dim * 2  # Latent K and V
        
        return original_kv_cache / mla_kv_cache
