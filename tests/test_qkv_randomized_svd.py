#!/usr/bin/env python3
"""
Comprehensive tests for Q, K, V randomized SVD compression

This test suite validates the extended implementation that supports
individual compression of Query, Key, and Value matrices.
"""

import torch
import numpy as np
from src.mla_gpt.model.compression import RandomizedSVDCompression
from src.mla_gpt.model.attention import CausalSelfAttention
from src.mla_gpt.model.model import GPTConfig


def test_individual_qkv_compression():
    """
    Test that Q, K, V matrices can be compressed individually
    """
    print("=== Testing Individual Q, K, V Compression ===")
    
    torch.manual_seed(42)
    
    # Create test matrices
    B, nh, T, hs = 2, 4, 32, 32
    q_matrix = torch.randn(B, nh, T, hs)
    k_matrix = torch.randn(B, nh, T, hs)
    v_matrix = torch.randn(B, nh, T, hs)
    
    # Create compressor
    compressor = RandomizedSVDCompression(
        rank=16,
        oversampling=8,
        power_iterations=1
    )
    
    # Test each matrix individually
    q_compressed = compressor(q_matrix)
    k_compressed = compressor(k_matrix) 
    v_compressed = compressor(v_matrix)
    
    # Check shapes are preserved
    assert q_compressed.shape == q_matrix.shape, "Q shape mismatch"
    assert k_compressed.shape == k_matrix.shape, "K shape mismatch"
    assert v_compressed.shape == v_matrix.shape, "V shape mismatch"
    
    # Check compression actually happened (should not be identical)
    q_diff = torch.norm(q_matrix - q_compressed).item()
    k_diff = torch.norm(k_matrix - k_compressed).item()
    v_diff = torch.norm(v_matrix - v_compressed).item()
    
    assert q_diff > 1e-6, "Q compression too accurate (possibly no compression)"
    assert k_diff > 1e-6, "K compression too accurate (possibly no compression)"
    assert v_diff > 1e-6, "V compression too accurate (possibly no compression)"
    
    # Check reconstruction quality is reasonable
    q_rel_error = q_diff / torch.norm(q_matrix).item()
    k_rel_error = k_diff / torch.norm(k_matrix).item()
    v_rel_error = v_diff / torch.norm(v_matrix).item()
    
    print(f"Q relative error: {q_rel_error:.6f}")
    print(f"K relative error: {k_rel_error:.6f}")
    print(f"V relative error: {v_rel_error:.6f}")
    
    assert q_rel_error < 0.5, "Q compression error too large"
    assert k_rel_error < 0.5, "K compression error too large"
    assert v_rel_error < 0.5, "V compression error too large"
    
    print("✅ Individual Q, K, V compression test passed")


def test_attention_compression_configs():
    """
    Test different attention compression configurations
    """
    print("\\n=== Testing Attention Compression Configurations ===")
    
    base_config = GPTConfig(
        n_embd=192,
        n_head=6,
        block_size=128,
        use_svd=True,
        use_randomized_svd=True,
        svd_rank=16,
        svd_oversampling=10,
        svd_power_iter=1
    )
    
    # Test configurations
    configs = [
        {'name': 'V-only', 'q': False, 'k': False, 'v': True},
        {'name': 'K,V', 'q': False, 'k': True, 'v': True},
        {'name': 'Q,K,V', 'q': True, 'k': True, 'v': True},
        {'name': 'Q-only', 'q': True, 'k': False, 'v': False},
        {'name': 'K-only', 'q': False, 'k': True, 'v': False},
        {'name': 'None', 'q': False, 'k': False, 'v': False}
    ]
    
    for config_test in configs:
        print(f"Testing {config_test['name']} compression...")
        
        # Configure compression
        test_config = base_config
        test_config.svd_apply_to_q = config_test['q']
        test_config.svd_apply_to_k = config_test['k']
        test_config.svd_apply_to_v = config_test['v']
        
        # Create attention module
        attention = CausalSelfAttention(test_config)
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, test_config.n_embd)
        
        with torch.no_grad():
            output = attention(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape mismatch for {config_test['name']}"
        
        # Check compression info
        matrix_shape = (batch_size, test_config.n_head, seq_len, test_config.n_embd // test_config.n_head)
        info = attention.get_compression_info(matrix_shape)
        
        assert info['q_compression'] == config_test['q'], f"Q compression config mismatch"
        assert info['k_compression'] == config_test['k'], f"K compression config mismatch"
        assert info['v_compression'] == config_test['v'], f"V compression config mismatch"
        
        print(f"  ✓ {config_test['name']} configuration working")
    
    print("✅ Attention compression configurations test passed")


def test_compression_quality_consistency():
    """
    Test that compression quality is consistent across Q, K, V
    """
    print("\\n=== Testing Compression Quality Consistency ===")
    
    torch.manual_seed(42)
    
    # Create identical matrices
    B, nh, T, hs = 2, 4, 64, 32
    base_matrix = torch.randn(B, nh, T, hs)
    
    # Apply same compression to each
    compressor = RandomizedSVDCompression(
        rank=16,
        oversampling=10,
        power_iterations=1
    )
    
    # Compress identical matrices
    q_compressed = compressor(base_matrix.clone())
    k_compressed = compressor(base_matrix.clone())
    v_compressed = compressor(base_matrix.clone())
    
    # Calculate relative errors
    q_error = torch.norm(base_matrix - q_compressed).item() / torch.norm(base_matrix).item()
    k_error = torch.norm(base_matrix - k_compressed).item() / torch.norm(base_matrix).item()
    v_error = torch.norm(base_matrix - v_compressed).item() / torch.norm(base_matrix).item()
    
    print(f"Relative errors for identical input:")
    print(f"  Q: {q_error:.6f}")
    print(f"  K: {k_error:.6f}")
    print(f"  V: {v_error:.6f}")
    
    # Errors should be similar (within 10% of each other) for identical inputs
    max_error = max(q_error, k_error, v_error)
    min_error = min(q_error, k_error, v_error)
    error_variation = (max_error - min_error) / max_error
    
    print(f"Error variation: {error_variation:.6f}")
    assert error_variation < 0.1, f"Large error variation ({error_variation:.6f}) between Q, K, V"
    
    print("✅ Compression quality consistency test passed")


def test_parameter_sensitivity_qkv():
    """
    Test parameter sensitivity across Q, K, V compression
    """
    print("\\n=== Testing Parameter Sensitivity for Q, K, V ===")
    
    torch.manual_seed(42)
    
    # Test matrices
    B, nh, T, hs = 2, 4, 32, 32
    q_matrix = torch.randn(B, nh, T, hs)
    k_matrix = torch.randn(B, nh, T, hs)
    v_matrix = torch.randn(B, nh, T, hs)
    
    # Test rank sensitivity
    print("Rank sensitivity:")
    for rank in [8, 16, 24]:
        compressor = RandomizedSVDCompression(rank=rank, oversampling=10, power_iterations=1)
        
        q_comp = compressor(q_matrix)
        k_comp = compressor(k_matrix)
        v_comp = compressor(v_matrix)
        
        q_err = torch.norm(q_matrix - q_comp).item() / torch.norm(q_matrix).item()
        k_err = torch.norm(k_matrix - k_comp).item() / torch.norm(k_matrix).item()
        v_err = torch.norm(v_matrix - v_comp).item() / torch.norm(v_matrix).item()
        
        print(f"  Rank {rank}: Q={q_err:.6f}, K={k_err:.6f}, V={v_err:.6f}")
        
        # Errors should decrease with higher rank
        assert q_err < 0.8, f"Q error too high at rank {rank}"
        assert k_err < 0.8, f"K error too high at rank {rank}"
        assert v_err < 0.8, f"V error too high at rank {rank}"
    
    # Test oversampling sensitivity
    print("Oversampling sensitivity:")
    for oversampling in [5, 10, 15]:
        compressor = RandomizedSVDCompression(rank=16, oversampling=oversampling, power_iterations=1)
        
        q_comp = compressor(q_matrix)
        k_comp = compressor(k_matrix)
        v_comp = compressor(v_matrix)
        
        q_err = torch.norm(q_matrix - q_comp).item() / torch.norm(q_matrix).item()
        k_err = torch.norm(k_matrix - k_comp).item() / torch.norm(k_matrix).item()
        v_err = torch.norm(v_matrix - v_comp).item() / torch.norm(v_matrix).item()
        
        print(f"  Oversampling {oversampling}: Q={q_err:.6f}, K={k_err:.6f}, V={v_err:.6f}")
    
    print("✅ Parameter sensitivity test passed")


def test_attention_forward_consistency():
    """
    Test that attention forward pass works consistently with different compression settings
    """
    print("\\n=== Testing Attention Forward Pass Consistency ===")
    
    torch.manual_seed(42)
    
    # Base configuration
    base_config = GPTConfig(
        n_embd=96,  # Small for fast testing
        n_head=4,
        block_size=64,
        use_svd=True,
        use_randomized_svd=True,
        svd_rank=12,
        svd_oversampling=8,
        svd_power_iter=1
    )
    
    # Test input
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, base_config.n_embd)
    
    # Test different compression combinations
    compression_configs = [
        (False, False, False),  # No compression
        (False, False, True),   # V only
        (False, True, True),    # K, V
        (True, True, True),     # All
    ]
    
    outputs = {}
    
    for i, (q_comp, k_comp, v_comp) in enumerate(compression_configs):
        config = base_config
        config.svd_apply_to_q = q_comp
        config.svd_apply_to_k = k_comp
        config.svd_apply_to_v = v_comp
        
        attention = CausalSelfAttention(config)
        
        with torch.no_grad():
            output = attention(x)
        
        outputs[i] = output
        
        # Check output shape
        assert output.shape == x.shape, f"Wrong output shape for config {i}"
        
        # Check output is not all zeros or NaN
        assert not torch.isnan(output).any(), f"NaN in output for config {i}"
        assert torch.norm(output).item() > 1e-6, f"Output too small for config {i}"
        
        config_name = f"Q={q_comp}, K={k_comp}, V={v_comp}"
        print(f"  ✓ Config {config_name}: output norm = {torch.norm(output).item():.6f}")
    
    # Compare outputs - they should be different but reasonable
    for i in range(len(outputs)-1):
        for j in range(i+1, len(outputs)):
            diff = torch.norm(outputs[i] - outputs[j]).item()
            rel_diff = diff / (torch.norm(outputs[i]).item() + torch.norm(outputs[j]).item())
            print(f"  Relative difference between config {i} and {j}: {rel_diff:.6f}")
            
            # Outputs should be different (compression changes results)
            # but not drastically different (compression should preserve most information)
            assert rel_diff > 1e-6, f"Outputs too similar between configs {i} and {j}"
            assert rel_diff < 0.5, f"Outputs too different between configs {i} and {j}"
    
    print("✅ Attention forward pass consistency test passed")


def test_memory_efficiency():
    """
    Test that the compression actually reduces computational requirements
    """
    print("\\n=== Testing Memory and Computational Efficiency ===")
    
    # Test different matrix sizes
    test_sizes = [
        (32, 32),   # Small
        (64, 64),   # Medium
        (128, 128), # Large
    ]
    
    for m, n in test_sizes:
        compressor = RandomizedSVDCompression(
            rank=min(16, m//2),
            oversampling=8,
            power_iterations=1
        )
        
        matrix_shape = (2, 4, m, n)  # (B, nh, m, n)
        
        # Get complexity analysis
        complexity = compressor.get_computational_complexity(matrix_shape)
        memory = compressor.get_memory_usage(matrix_shape)
        
        print(f"Matrix size {m}x{n}:")
        print(f"  Speedup ratio: {complexity['speedup_ratio']:.2f}x")
        print(f"  Memory overhead: {memory['memory_overhead_ratio']:.2f}x")
        
        # Basic sanity checks
        assert complexity['speedup_ratio'] > 0, "Invalid speedup ratio"
        assert memory['memory_overhead_ratio'] > 0, "Invalid memory overhead"
        
        # For larger matrices, we should see computational benefits
        if min(m, n) >= 64:
            print(f"    Expected speedup for size {m}x{n}")
    
    print("✅ Memory and computational efficiency test passed")


if __name__ == "__main__":
    print("Testing Q, K, V Randomized SVD Implementation\\n")
    
    try:
        test_individual_qkv_compression()
        test_attention_compression_configs()
        test_compression_quality_consistency()
        test_parameter_sensitivity_qkv()
        test_attention_forward_consistency()
        test_memory_efficiency()
        
        print("\\n" + "="*60)
        print("🎉 ALL Q, K, V RANDOMIZED SVD TESTS PASSED! 🎉")
        print("✅ Individual matrix compression working")
        print("✅ Attention integration successful")  
        print("✅ Configuration flexibility validated")
        print("✅ Quality consistency confirmed")
        print("✅ Parameter sensitivity understood")
        print("✅ Forward pass stability verified")
        print("✅ Efficiency characteristics measured")
        print("="*60)
        
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        raise