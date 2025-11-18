#!/usr/bin/env python3
"""
Example usage of Randomized SVD for Q, K, and V matrix compression

This example demonstrates how to use the extended implementation that supports
compression of Query, Key, and Value matrices individually.
"""

import torch
import torch.nn as nn
from src.mla_gpt.model.compression import RandomizedSVDCompression
from src.mla_gpt.model.attention import CausalSelfAttention
from src.mla_gpt.model.model import GPTConfig
from dataclasses import dataclass


def demonstrate_qkv_compression():
    """
    Demonstrate Q, K, V compression with different configurations
    """
    print("=== Q, K, V Matrix Compression with Randomized SVD ===\n")
    
    # Create test matrices (typical attention shapes)
    batch_size = 2
    n_heads = 8
    seq_len = 128
    head_size = 64
    
    q_matrix = torch.randn(batch_size, n_heads, seq_len, head_size)
    k_matrix = torch.randn(batch_size, n_heads, seq_len, head_size) 
    v_matrix = torch.randn(batch_size, n_heads, seq_len, head_size)
    
    print(f"Matrix shapes: Q={q_matrix.shape}, K={k_matrix.shape}, V={v_matrix.shape}")
    
    # Create compressor
    compressor = RandomizedSVDCompression(
        rank=32,
        oversampling=10,
        power_iterations=1
    )
    
    print(f"\\nCompression settings: rank=32, oversampling=10, power_iterations=1")
    
    # Test compression on each matrix
    print(f"\\nCompressing individual matrices...")
    
    q_compressed = compressor(q_matrix)
    k_compressed = compressor(k_matrix)
    v_compressed = compressor(v_matrix)
    
    # Calculate errors
    q_error = torch.norm(q_matrix - q_compressed).item() / torch.norm(q_matrix).item()
    k_error = torch.norm(k_matrix - k_compressed).item() / torch.norm(k_matrix).item()
    v_error = torch.norm(v_matrix - v_compressed).item() / torch.norm(v_matrix).item()
    
    print(f"Reconstruction errors:")
    print(f"  Q matrix: {q_error:.6f} ({q_error*100:.3f}%)")
    print(f"  K matrix: {k_error:.6f} ({k_error*100:.3f}%)")
    print(f"  V matrix: {v_error:.6f} ({v_error*100:.3f}%)")


def demonstrate_attention_configurations():
    """
    Show different attention configurations with Q, K, V compression
    """
    print(f"\\n=== Attention Mechanism Configurations ===\\n")
    
    base_config = GPTConfig(
        n_embd=384,   # Smaller for demo
        n_head=6,
        block_size=256,
        use_svd=True,
        use_randomized_svd=True,
        svd_rank=24,
        svd_oversampling=10,
        svd_power_iter=1
    )
    
    # Configuration 1: V-only compression (recommended starting point)
    print("1. V-only compression (recommended)")
    v_only_config = base_config
    v_only_config.svd_apply_to_q = False
    v_only_config.svd_apply_to_k = False
    v_only_config.svd_apply_to_v = True
    
    attention_v_only = CausalSelfAttention(v_only_config)
    test_compression_quality(attention_v_only, "V-only", base_config)
    
    # Configuration 2: K, V compression (preserve query precision)
    print("\\n2. K, V compression (preserve query precision)")
    kv_config = base_config
    kv_config.svd_apply_to_q = False
    kv_config.svd_apply_to_k = True
    kv_config.svd_apply_to_v = True
    
    attention_kv = CausalSelfAttention(kv_config)
    test_compression_quality(attention_kv, "K,V", base_config)
    
    # Configuration 3: Q, K, V compression (maximum compression)
    print("\\n3. Q, K, V compression (maximum compression)")
    qkv_config = base_config
    qkv_config.svd_apply_to_q = True
    qkv_config.svd_apply_to_k = True
    qkv_config.svd_apply_to_v = True
    
    attention_qkv = CausalSelfAttention(qkv_config)
    test_compression_quality(attention_qkv, "Q,K,V", base_config)


def test_compression_quality(attention_module, config_name, config):
    """
    Test the quality of compression for a given attention configuration
    """
    batch_size = 2
    seq_len = 64
    
    # Test input
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    # Get compression info
    matrix_shape = (batch_size, config.n_head, seq_len, config.n_embd // config.n_head)
    compression_info = attention_module.get_compression_info(matrix_shape)
    
    print(f"  Configuration: {config_name}")
    print(f"    Q compression: {'✓' if compression_info['q_compression'] else '✗'}")
    print(f"    K compression: {'✓' if compression_info['k_compression'] else '✗'}")
    print(f"    V compression: {'✓' if compression_info['v_compression'] else '✗'}")
    
    if 'complexity' in compression_info:
        speedup = compression_info['complexity']['speedup_ratio']
        memory_overhead = compression_info['memory']['memory_overhead_ratio']
        print(f"    Expected speedup: {speedup:.2f}x per matrix")
        print(f"    Memory overhead: {memory_overhead:.2f}x")
    
    # Forward pass to test functionality
    with torch.no_grad():
        output = attention_module(x)
    
    print(f"    Forward pass successful: {output.shape}")


def demonstrate_parameter_effects_on_qkv():
    """
    Show how parameters affect Q, K, V compression differently
    """
    print(f"\\n=== Parameter Effects on Q, K, V Compression ===\\n")
    
    # Create identical test matrices
    torch.manual_seed(42)
    matrix_shape = (2, 4, 64, 32)
    q_matrix = torch.randn(*matrix_shape)
    k_matrix = torch.randn(*matrix_shape)
    v_matrix = torch.randn(*matrix_shape)
    
    # Test different ranks
    print("Rank sensitivity:")
    ranks = [8, 16, 24]
    for rank in ranks:
        compressor = RandomizedSVDCompression(rank=rank, oversampling=10, power_iterations=1)
        
        q_comp = compressor(q_matrix)
        k_comp = compressor(k_matrix)
        v_comp = compressor(v_matrix)
        
        q_err = torch.norm(q_matrix - q_comp).item() / torch.norm(q_matrix).item()
        k_err = torch.norm(k_matrix - k_comp).item() / torch.norm(k_matrix).item()
        v_err = torch.norm(v_matrix - v_comp).item() / torch.norm(v_matrix).item()
        
        print(f"  Rank {rank:2d}: Q={q_err:.4f}, K={k_err:.4f}, V={v_err:.4f}")
    
    # Test different power iterations
    print("\\nPower iteration sensitivity:")
    power_iters = [0, 1, 2]
    for q in power_iters:
        compressor = RandomizedSVDCompression(rank=16, oversampling=10, power_iterations=q)
        
        q_comp = compressor(q_matrix)
        k_comp = compressor(k_matrix)
        v_comp = compressor(v_matrix)
        
        q_err = torch.norm(q_matrix - q_comp).item() / torch.norm(q_matrix).item()
        k_err = torch.norm(k_matrix - k_comp).item() / torch.norm(k_matrix).item()
        v_err = torch.norm(v_matrix - v_comp).item() / torch.norm(v_matrix).item()
        
        print(f"  Power {q}: Q={q_err:.4f}, K={k_err:.4f}, V={v_err:.4f}")


def demonstrate_selective_compression():
    """
    Show practical scenarios for selective Q, K, V compression
    """
    print(f"\\n=== Practical Selective Compression Scenarios ===\\n")
    
    scenarios = [
        {
            'name': 'Memory-constrained inference',
            'description': 'Compress V only to reduce memory with minimal quality loss',
            'q': False, 'k': False, 'v': True,
            'rank': 32, 'oversampling': 10, 'power_iter': 1
        },
        {
            'name': 'Balanced compression',
            'description': 'Compress K, V while preserving query precision',
            'q': False, 'k': True, 'v': True,
            'rank': 24, 'oversampling': 10, 'power_iter': 1
        },
        {
            'name': 'Maximum efficiency',
            'description': 'Compress all matrices for maximum speedup',
            'q': True, 'k': True, 'v': True,
            'rank': 20, 'oversampling': 8, 'power_iter': 0
        },
        {
            'name': 'Quality-focused',
            'description': 'Light compression with high accuracy',
            'q': False, 'k': False, 'v': True,
            'rank': 48, 'oversampling': 15, 'power_iter': 2
        }
    ]
    
    base_config = GPTConfig(
        n_embd=192,
        n_head=6,
        block_size=128,
        use_svd=True,
        use_randomized_svd=True
    )
    
    for scenario in scenarios:
        print(f"{scenario['name']}:")
        print(f"  Description: {scenario['description']}")
        print(f"  Compression: Q={scenario['q']}, K={scenario['k']}, V={scenario['v']}")
        print(f"  Parameters: rank={scenario['rank']}, oversampling={scenario['oversampling']}, power_iter={scenario['power_iter']}")
        
        # Configure attention
        config = base_config
        config.svd_apply_to_q = scenario['q']
        config.svd_apply_to_k = scenario['k'] 
        config.svd_apply_to_v = scenario['v']
        config.svd_rank = scenario['rank']
        config.svd_oversampling = scenario['oversampling']
        config.svd_power_iter = scenario['power_iter']
        
        attention = CausalSelfAttention(config)
        
        # Test forward pass
        x = torch.randn(1, 32, config.n_embd)
        with torch.no_grad():
            output = attention(x)
        
        print(f"  Status: ✓ Working")
        print()


if __name__ == "__main__":
    # Set random seed for reproducible results
    torch.manual_seed(42)
    
    # Run demonstrations
    demonstrate_qkv_compression()
    demonstrate_attention_configurations()
    demonstrate_parameter_effects_on_qkv()
    demonstrate_selective_compression()
    
    print(f"=== Summary ===")
    print("✅ Successfully demonstrated Q, K, V compression with Randomized SVD")
    print("✅ Individual matrix compression controls working")
    print("✅ Multiple configuration scenarios tested")
    print("✅ Parameter sensitivity analysis completed")
    print("\\nRecommendations:")
    print("• Start with V-only compression for best quality/compression balance")
    print("• Add K compression if more memory savings needed") 
    print("• Use Q, K, V compression only when maximum efficiency is required")
    print("• Monitor attention quality when compressing multiple matrices")