#!/usr/bin/env python3
"""
Example usage of Tropp's Randomized SVD for V matrix compression

This example demonstrates how to use the newly implemented RandomizedSVDCompression
class for compressing V matrices in attention mechanisms.
"""

import torch
import torch.nn as nn
from src.mla_gpt.model.compression import RandomizedSVDCompression
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT model with randomized SVD compression"""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # SVD compression configuration
    use_svd: bool = True
    use_randomized_svd: bool = True
    svd_rank: int = 32  # Target rank for compression
    
    # Tropp's algorithm parameters
    svd_oversampling: int = 10  # Oversampling parameter p (typically 5-10)
    svd_power_iter: int = 1     # Power iterations q (0-2, typically 1)


def demonstrate_randomized_svd():
    """
    Demonstrate Tropp's randomized SVD compression on V matrices
    """
    print("=== Tropp's Randomized SVD Demonstration ===\n")
    
    # Configuration
    config = GPTConfig()
    
    # Create test V matrix (typical attention values shape)
    batch_size = 4
    n_heads = config.n_head
    seq_len = 256
    head_size = config.n_embd // config.n_head
    
    # V matrix: (B, nh, T, hs)
    v_matrix = torch.randn(batch_size, n_heads, seq_len, head_size)
    print(f"Original V matrix shape: {v_matrix.shape}")
    print(f"Original matrix size: {v_matrix.numel()} parameters")
    
    # Initialize randomized SVD compression
    compressor = RandomizedSVDCompression(
        rank=config.svd_rank,
        oversampling=config.svd_oversampling,
        power_iterations=config.svd_power_iter
    )
    
    print(f"\nCompression Configuration:")
    print(f"  Target rank: {config.svd_rank}")
    print(f"  Oversampling: {config.svd_oversampling}")
    print(f"  Power iterations: {config.svd_power_iter}")
    
    # Apply compression
    print(f"\nApplying Tropp's randomized SVD...")
    v_compressed = compressor(v_matrix)
    
    print(f"Compressed V matrix shape: {v_compressed.shape}")
    print(f"Shapes match: {v_matrix.shape == v_compressed.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = torch.norm(v_matrix - v_compressed).item()
    relative_error = reconstruction_error / torch.norm(v_matrix).item()
    
    print(f"\nReconstruction Quality:")
    print(f"  Absolute error: {reconstruction_error:.6f}")
    print(f"  Relative error: {relative_error:.6f}")
    print(f"  Relative error %: {relative_error * 100:.4f}%")
    
    # Get computational complexity analysis
    matrix_shape = v_matrix.shape
    complexity = compressor.get_computational_complexity(matrix_shape)
    memory = compressor.get_memory_usage(matrix_shape)
    
    print(f"\nComputational Analysis:")
    print(f"  Speedup ratio: {complexity['speedup_ratio']:.2f}x")
    print(f"  Standard SVD complexity: {complexity['standard_svd']:,}")
    print(f"  Randomized SVD complexity: {complexity['randomized_svd']:,}")
    
    print(f"\nMemory Analysis:")
    print(f"  Original matrix memory: {memory['original_matrix']:,} elements")
    print(f"  Additional memory needed: {memory['additional_memory']:,} elements") 
    print(f"  Memory overhead: {memory['memory_overhead_ratio']:.2f}x")
    
    # Demonstrate parameter sensitivity
    print(f"\n=== Parameter Sensitivity Analysis ===")
    
    # Test different ranks
    ranks = [8, 16, 32, 64]
    for rank in ranks:
        if rank < min(seq_len, head_size):
            test_compressor = RandomizedSVDCompression(rank=rank)
            v_test = test_compressor(v_matrix)
            error = torch.norm(v_matrix - v_test).item() / torch.norm(v_matrix).item()
            print(f"Rank {rank:2d}: Relative error = {error:.6f} ({error*100:.3f}%)")
    
    # Test different oversampling values
    print(f"\nOversampling sensitivity (rank={config.svd_rank}):")
    oversamplings = [5, 10, 15, 20]
    for p in oversamplings:
        test_compressor = RandomizedSVDCompression(
            rank=config.svd_rank, 
            oversampling=p, 
            power_iterations=1
        )
        v_test = test_compressor(v_matrix)
        error = torch.norm(v_matrix - v_test).item() / torch.norm(v_matrix).item()
        print(f"Oversampling {p:2d}: Relative error = {error:.6f} ({error*100:.3f}%)")
    
    # Test different power iterations
    print(f"\nPower iteration sensitivity (rank={config.svd_rank}):")
    power_iters = [0, 1, 2, 3]
    for q in power_iters:
        test_compressor = RandomizedSVDCompression(
            rank=config.svd_rank,
            oversampling=config.svd_oversampling,
            power_iterations=q
        )
        v_test = test_compressor(v_matrix)
        error = torch.norm(v_matrix - v_test).item() / torch.norm(v_matrix).item()
        print(f"Power iter {q}: Relative error = {error:.6f} ({error*100:.3f}%)")


def demonstrate_integration_with_attention():
    """
    Show how to integrate with the attention mechanism
    """
    print(f"\n=== Integration with Attention Mechanism ===")
    
    from src.mla_gpt.model.attention import CausalSelfAttention
    
    config = GPTConfig()
    attention = CausalSelfAttention(config)
    
    # Test input
    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass with randomized SVD compression
    with torch.no_grad():
        output = attention(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Compression applied: {attention.use_randomized_svd}")
    
    # Get compression information
    v_shape = (batch_size, config.n_head, seq_len, config.n_embd // config.n_head)
    compression_info = attention.get_compression_info(v_shape)
    
    print(f"\nCompression Info:")
    for key, value in compression_info.items():
        if key not in ['complexity', 'memory']:
            print(f"  {key}: {value}")
    
    if 'complexity' in compression_info:
        speedup = compression_info['complexity']['speedup_ratio']
        print(f"  Expected speedup: {speedup:.2f}x")


if __name__ == "__main__":
    # Set random seed for reproducible results
    torch.manual_seed(42)
    
    # Run demonstrations
    demonstrate_randomized_svd()
    demonstrate_integration_with_attention()
    
    print(f"\n=== Summary ===")
    print("✅ Successfully implemented Tropp's Randomized SVD Algorithm 4.1")
    print("✅ Supports power iterations (Algorithm 4.4) for improved accuracy")
    print("✅ Integrated with attention mechanism for V matrix compression")
    print("✅ Provides computational and memory complexity analysis")
    print("✅ Follows theoretical recommendations from Halko, Martinsson, Tropp (2011)")