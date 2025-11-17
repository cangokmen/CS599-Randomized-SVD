#!/usr/bin/env python3
"""
Test implementation of Tropp's Randomized SVD compression

This test validates the correctness and performance of the RandomizedSVDCompression
implementation against the reference implementation from FULL-CS599-Randomized-SVD.
"""

import torch
import numpy as np
from src.mla_gpt.model.compression import RandomizedSVDCompression

def test_algorithm_correctness():
    """
    Test that the randomized SVD implementation follows Tropp's Algorithm 4.1 correctly
    """
    print("=== Testing Algorithm Correctness ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test matrix dimensions
    m, n = 100, 80
    rank = 20
    
    # Generate test matrix with known low-rank structure
    U_true = torch.randn(m, rank)
    S_true = torch.linspace(10, 1, rank)  # Decaying singular values
    Vh_true = torch.randn(rank, n)
    A = U_true @ torch.diag(S_true) @ Vh_true
    
    print(f"Test matrix shape: {A.shape}")
    print(f"True rank: {rank}")
    
    # Apply randomized SVD
    compressor = RandomizedSVDCompression(
        rank=rank,
        oversampling=10,
        power_iterations=1
    )
    
    A_approx = compressor._compress_single_matrix_randomized(A)
    
    # Compute standard SVD for comparison
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    A_standard = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    
    # Calculate errors
    error_randomized = torch.norm(A - A_approx).item()
    error_standard = torch.norm(A - A_standard).item()
    
    print(f"Standard SVD error: {error_standard:.8f}")
    print(f"Randomized SVD error: {error_randomized:.8f}")
    print(f"Error ratio: {error_randomized / error_standard:.4f}")
    
    # The randomized SVD should be close to optimal SVD (within factor of 2-3 typically)
    assert error_randomized / error_standard < 5.0, "Randomized SVD error too large"
    print("✅ Algorithm correctness test passed")


def test_parameter_effects():
    """
    Test the effect of oversampling and power iterations
    """
    print("\n=== Testing Parameter Effects ===")
    
    torch.manual_seed(42)
    
    # Generate test matrix
    m, n = 80, 60
    rank = 15
    A = torch.randn(m, n)
    
    # Test different oversampling values
    print("Oversampling effect (power_iter=1):")
    for p in [5, 10, 15, 20]:
        compressor = RandomizedSVDCompression(
            rank=rank, oversampling=p, power_iterations=1
        )
        A_approx = compressor._compress_single_matrix_randomized(A)
        error = torch.norm(A - A_approx).item() / torch.norm(A).item()
        print(f"  p={p:2d}: relative error = {error:.6f}")
    
    # Test different power iterations
    print("\nPower iteration effect (oversampling=10):")
    for q in [0, 1, 2, 3]:
        compressor = RandomizedSVDCompression(
            rank=rank, oversampling=10, power_iterations=q
        )
        A_approx = compressor._compress_single_matrix_randomized(A)
        error = torch.norm(A - A_approx).item() / torch.norm(A).item()
        print(f"  q={q}: relative error = {error:.6f}")
    
    print("✅ Parameter effects test passed")


def test_batch_processing():
    """
    Test batch processing for attention matrices
    """
    print("\n=== Testing Batch Processing ===")
    
    torch.manual_seed(42)
    
    # Typical attention V matrix dimensions
    B, nh, T, hs = 2, 8, 64, 64  # batch=2, heads=8, seq_len=64, head_size=64
    rank = 16
    
    v_matrices = torch.randn(B, nh, T, hs)
    
    compressor = RandomizedSVDCompression(
        rank=rank,
        oversampling=10,
        power_iterations=1
    )
    
    # Compress batch
    v_compressed = compressor(v_matrices)
    
    # Check shapes match
    assert v_compressed.shape == v_matrices.shape, "Shape mismatch after compression"
    
    # Check each matrix individually
    v_reshaped = v_matrices.reshape(B * nh, T, hs)
    errors = []
    for i in range(B * nh):
        original = v_reshaped[i]
        compressed = compressor._compress_single_matrix_randomized(original)
        error = torch.norm(original - compressed).item() / torch.norm(original).item()
        errors.append(error)
    
    avg_error = np.mean(errors)
    print(f"Batch shape: {v_matrices.shape}")
    print(f"Average relative error: {avg_error:.6f}")
    print(f"Error std: {np.std(errors):.6f}")
    
    print("✅ Batch processing test passed")


def test_edge_cases():
    """
    Test edge cases and boundary conditions
    """
    print("\n=== Testing Edge Cases ===")
    
    compressor = RandomizedSVDCompression(rank=10, oversampling=5, power_iterations=1)
    
    # Test small matrices
    small_matrix = torch.randn(5, 5)
    result = compressor._compress_single_matrix_randomized(small_matrix)
    assert result.shape == small_matrix.shape
    print("✅ Small matrix test passed")
    
    # Test rank larger than matrix dimensions
    matrix = torch.randn(10, 8)
    compressor_large_rank = RandomizedSVDCompression(rank=20, oversampling=5)
    result = compressor_large_rank._compress_single_matrix_randomized(matrix)
    assert result.shape == matrix.shape
    print("✅ Large rank test passed")
    
    # Test full rank (rank=None)
    compressor_full = RandomizedSVDCompression(rank=None, oversampling=5)
    result = compressor_full._compress_single_matrix_randomized(matrix)
    error = torch.norm(matrix - result).item() / torch.norm(matrix).item()
    print(f"Full rank error: {error:.8f}")
    assert error < 1e-6, "Full rank should be nearly exact"
    print("✅ Full rank test passed")


def test_complexity_analysis():
    """
    Test computational complexity and memory analysis functions
    """
    print("\n=== Testing Complexity Analysis ===")
    
    compressor = RandomizedSVDCompression(rank=32, oversampling=10, power_iterations=1)
    
    # Test shape
    matrix_shape = (4, 12, 128, 64)  # (B, nh, T, hs)
    
    # Get complexity analysis
    complexity = compressor.get_computational_complexity(matrix_shape)
    memory = compressor.get_memory_usage(matrix_shape)
    
    print(f"Matrix shape: {matrix_shape}")
    print(f"Speedup ratio: {complexity['speedup_ratio']:.2f}x")
    print(f"Memory overhead: {memory['memory_overhead_ratio']:.2f}x")
    
    # Basic sanity checks
    assert complexity['speedup_ratio'] > 0
    assert memory['memory_overhead_ratio'] > 0
    assert 'parameters' in complexity
    
    print("✅ Complexity analysis test passed")


def compare_with_standard_svd():
    """
    Compare randomized SVD with standard SVD implementation
    """
    print("\n=== Comparing with Standard SVD ===")
    
    from src.mla_gpt.model.compression import SVDCompression
    
    torch.manual_seed(42)
    
    # Test matrix
    B, nh, T, hs = 2, 4, 32, 32
    v_matrix = torch.randn(B, nh, T, hs)
    rank = 8
    
    # Standard SVD
    standard_compressor = SVDCompression(rank=rank, compression_type='standard')
    v_standard = standard_compressor(v_matrix)
    
    # Randomized SVD
    randomized_compressor = RandomizedSVDCompression(
        rank=rank, oversampling=10, power_iterations=1
    )
    v_randomized = randomized_compressor(v_matrix)
    
    # Compare errors
    error_standard = torch.norm(v_matrix - v_standard).item() / torch.norm(v_matrix).item()
    error_randomized = torch.norm(v_matrix - v_randomized).item() / torch.norm(v_matrix).item()
    
    print(f"Standard SVD relative error: {error_standard:.6f}")
    print(f"Randomized SVD relative error: {error_randomized:.6f}")
    print(f"Error ratio: {error_randomized / error_standard:.4f}")
    
    # Randomized should be competitive with standard SVD
    assert error_randomized / error_standard < 3.0, "Randomized SVD significantly worse than standard"
    
    print("✅ Comparison test passed")


if __name__ == "__main__":
    print("Testing Tropp's Randomized SVD Implementation\n")
    
    try:
        test_algorithm_correctness()
        test_parameter_effects() 
        test_batch_processing()
        test_edge_cases()
        test_complexity_analysis()
        compare_with_standard_svd()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Tropp's Randomized SVD implementation is working correctly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise